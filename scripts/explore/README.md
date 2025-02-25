Explore Utility Scripts
=======================

One can find there two scripts for running exploration on a combination
of backend and strategies.

These scripts are mainly used for regression tests and are subject to
change.

## Run script

The run script run explorations and stores exploration results.

The run script runs the combination of envvars:
- `BACKENDS`: backends to run, default: `mlir tvm`
- `STRATEGIES`: strategies to explore, default: all supported

With the random search and the following envvars:
- `TRIALS`: number of trials, default: `20`

Arguments to the script are in order:
- `outdir`: output results dir, default: `.`

For instance, run:

    ./run-explore-variants.sh tmp.result/
    ...

Note this may take some time, approximately: `2 secs * n_runs`
where `n_runs = n_backends * n_strategies * trials`.

In the current configuration it's approximately:
`2 secs * 2 * 10 * 20`, i.e 800 secs (13 mins) to run.


## Display script

The display script display generated exploration results
per strategy and save the figures.

The display script takes the following envvars:
- `STRATEGIES`: strategies to display, default: all supported
- `NOSHOW`: do not show figure is set, default: not set

Arguments to the script are in order:
- `outdir`: results and generated figures dir, default: `.`

For instance, run (after running the run script above):

    NOSHOW=1 ./display-explore-variants.sh tmp.result/
    ls tmp.results/*.png
    ...



