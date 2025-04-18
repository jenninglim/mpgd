#!/bin/bash
python scripts/run.py eval --config-name cifar.pgd_cnet --seed 10
python scripts/run.py eval --config-name cifar.apgd_cnet --seed 10
python scripts/run.py eval --config-name cifar.vi_cnet --seed 10

python scripts/run.py eval --config-name cifar.pgd_cnet --seed 11
python scripts/run.py eval --config-name cifar.pgd_cnet --seed 12

python scripts/run.py eval --config-name cifar.apgd_cnet --seed 11
python scripts/run.py eval --config-name cifar.apgd_cnet --seed 12

python scripts/run.py eval --config-name cifar.vi_cnet --seed 11
python scripts/run.py eval --config-name cifar.vi_cnet --seed 12
:'
python scripts/run.py eval --config-name cifar.sr_cnet --seed 10
python scripts/run.py eval --config-name cifar.abp_cnet --seed 11
python scripts/run.py eval --config-name cifar.sr_cnet --seed 11
python scripts/run.py eval --config-name cifar.sr_cnet --seed 12

python scripts/run.py eval --config-name cifar.abp_cnet --seed 10
python scripts/run.py eval --config-name cifar.abp_cnet --seed 12
'
