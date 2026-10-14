"""Single-GPU FP16 generation with synchronized timing and per-call accounting."""
import time
from .core import FEATURES, MODELS, digest


class HFBackend:
    def __init__(self, config, snapshots):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Install the cu128 torch wheel and check nvidia-smi.')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES per worker.')
        free, _ = torch.cuda.mem_get_info()
        if free < 36 * 1024**3:
            raise RuntimeError('Need at least 36 GiB free on the selected GPU.')
        self.torch, self.config = torch, config
        self.models, self.tokenizers = {}, {}
        self.metadata = {'device': torch.cuda.get_device_name(0),
                         'capability': list(torch.cuda.get_device_capability(0)),
                         'torch': torch.__version__, 'cuda': torch.version.cuda,
                         'arch_list': torch.cuda.get_arch_list()}
        import os
        self.metadata['visible_device_selection'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        props = torch.cuda.get_device_properties(0)
        self.metadata['total_memory_bytes'] = props.total_memory
        self.metadata['device_uuid'] = str(getattr(props, 'uuid', 'unavailable'))
        torch.manual_seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        start = time.perf_counter()
        for tier in MODELS:
            path = snapshots[tier]
            self.tokenizers[tier] = AutoTokenizer.from_pretrained(path, local_files_only=True)
            tok = self.tokenizers[tier]
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            tok.truncation_side = 'right'
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float16, attn_implementation='sdpa',
                low_cpu_mem_usage=True, local_files_only=True, use_safetensors=True)
            self.models[tier] = model.to('cuda:0').eval()
            print(f'Loaded {tier}: {MODELS[tier][0]}', flush=True)
        # Warm up every tier. These setup calls are excluded from episode costs.
        for tier in MODELS:
            self.generate(tier, [{'role': 'user', 'content': 'Reply with one word: ready.'}], 'first', 4)
        self.metadata['load_and_warmup_wall_s'] = time.perf_counter() - start

    def begin_episode(self, sample):
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    def peak_memory(self):
        return self.torch.cuda.max_memory_allocated()

    def generate(self, tier, messages, feature_mode, max_new_tokens):
        torch, tok = self.torch, self.tokenizers[tier]
        torch.cuda.synchronize()
        began = time.perf_counter()
        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_ids = tok(formatted, add_special_tokens=False)['input_ids']
        truncated = len(full_ids) > self.config['max_input_tokens']
        enc = tok(formatted, return_tensors='pt', add_special_tokens=False,
                  truncation=True, max_length=self.config['max_input_tokens']).to('cuda:0')
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            output = self.models[tier].generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                output_scores=(feature_mode != 'none'), return_dict_in_generate=True,
                pad_token_id=tok.pad_token_id, use_cache=True)
        end.record()
        end.synchronize()
        device_s = start.elapsed_time(end) / 1000.
        n_input = int(enc['input_ids'].shape[-1])
        generated = output.sequences[0, n_input:]
        response = tok.decode(generated, skip_special_tokens=True).strip()
        feats = {}
        if feature_mode != 'none':
            if not output.scores:
                raise RuntimeError('No scores returned for a feature-dependent policy')
            p = torch.softmax(output.scores[0][0].float(), dim=-1)
            top = torch.topk(p, 5).values
            entropy = -(p * p.clamp_min(1e-12).log()).sum()
            values = [entropy.item(), entropy.item(), top[0].item(),
                      (top[0] - top[1]).item(), top.mean().item()]
            if feature_mode == 'all':
                h = []
                for logits in output.scores:
                    prob = torch.softmax(logits[0].float(), dim=-1)
                    h.append(-(prob * prob.clamp_min(1e-12).log()).sum())
                values[1] = torch.stack(h).mean().item()
            feats = dict(zip(FEATURES, values))
        torch.cuda.synchronize()
        return {'response': response, 'features': feats,
                'input_tokens': n_input, 'untruncated_input_tokens': len(full_ids),
                'input_truncated': truncated, 'output_tokens': int(generated.numel()),
                'hit_output_limit': int(generated.numel()) == max_new_tokens,
                'wall_s': time.perf_counter() - began, 'cuda_s': device_s}


class MockBackend:
    """Synthetic deterministic backend for plumbing tests, never paper results."""
    def __init__(self, config, snapshots=None):
        self.metadata = {'device': 'MOCK - NOT SCIENTIFIC RESULTS'}

    def begin_episode(self, sample):
        self.sample = sample

    def peak_memory(self):
        return 0

    def generate(self, tier, messages, feature_mode, max_new_tokens):
        text = messages[-1]['content']
        value = int(digest(self.sample['id'])[:8], 16)
        h = (value % 400) / 100. if 'Observation:' in text else (value % 100) / 100.
        if 'confidence' in messages[0]['content'].lower():
            response = 'Confidence: 80'
        elif 'Observation:' not in text:
            response = 'Thought: Find evidence.\nAction: Search[evidence]'
        else:
            correct = value % {'small': 3, 'mid': 2, 'large': 5}[tier] == 0
            answer = self.sample['answer'] if correct else 'incorrect'
            response = f'Thought: Answer.\nAction: Finish[{answer}]'
        f = dict(zip(FEATURES, [h, h / 2, 1 / (1 + h), .1, .15])) if feature_mode != 'none' else {}
        cost = {'small': .01, 'mid': .02, 'large': .05}[tier]
        return {'response': response, 'features': f, 'input_tokens': len(text.split()),
                'untruncated_input_tokens': len(text.split()), 'input_truncated': False,
                'output_tokens': len(response.split()), 'hit_output_limit': False,
                'wall_s': cost, 'cuda_s': cost}
