# #!/usr/bin/env zsh


datasets=(wikipedia reddit simulate_v1 simulate_v2)
models=(tgat tgn)

cd "$ROOT/benchmarks/xgraph"

# Number of parallel processes. Set to an appropriate value based on the available CPU cores.
processes=1

for dataset in ${datasets[@]}
do
    echo "dataset: ${dataset}\n"
    for model in ${models[@]}
    do
        echo "model: ${model}\n"
	python ctm_run.py datasets=${dataset} device_id=0 explainers=ctms5 models=${model} ++explainers.parallel_degree=$processes
    done
done
