#!/bin/bash
:'
python scripts/run.py eval --config-name cifar.pgd_cnet --seed 10
python scripts/run.py eval --config-name cifar.pgd_cnet --seed 11
python scripts/run.py eval --config-name cifar.pgd_cnet --seed 12

python scripts/run.py eval --config-name cifar.apgd_cnet --seed 10
python scripts/run.py eval --config-name cifar.apgd_cnet --seed 11
python scripts/run.py eval --config-name cifar.apgd_cnet --seed 12

python scripts/run.py eval --config-name cifar.vi_cnet --seed 10
python scripts/run.py eval --config-name cifar.vi_cnet --seed 11
python scripts/run.py eval --config-name cifar.vi_cnet --seed 12
'
python scripts/run.py eval --config-name mnist.sr_mlp --seed 10
python scripts/run.py eval --config-name mnist.abp_mlp --seed 11
python scripts/run.py eval --config-name mnist.sr_mlp --seed 11
python scripts/run.py eval --config-name mnist.sr_mlp --seed 12

python scripts/run.py eval --config-name mnist.abp_mlp --seed 10
python scripts/run.py eval --config-name mnist.abp_mlp --seed 12
