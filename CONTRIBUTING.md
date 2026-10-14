# Contributing

Use Python 3.11 and keep scientific changes small and reviewable. Run the
CPU suite described in the root README before proposing a change to routing,
cost accounting, split construction, checkpoints, or statistical analysis.
Document the units and reference denominator of any new metric.

Never tune a threshold or select a policy on confirmatory evaluation outcomes.
Change the protocol/code in a new run directory and rerun the relevant
baselines. Record differences from the exploratory environment explicitly.

Do not commit authentication tokens, model weights, local caches, virtual
environments, or live checkpoint stores. Mock results must remain labeled as
mock and must not be inserted into the manuscript as experimental evidence.

The downloadable bundle mirrors `page2027_confirmatory/` and excludes runs,
virtual environments, and bytecode. Rebuild it whenever those source files
change and verify that its source contents match the tracked directory.
