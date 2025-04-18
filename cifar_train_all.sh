#!/bin/bash
python scripts/run.py train --config-name cifar.pgd_cnet --seed 10
python scripts/run.py train --config-name cifar.pgd_cnet --seed 11
python scripts/run.py train --config-name cifar.pgd_cnet --seed 12

python scripts/run.py train --config-name cifar.mpd_cnet --seed 10
python scripts/run.py train --config-name cifar.mpd_cnet --seed 11
python scripts/run.py train --config-name cifar.mpd_cnet --seed 12

python scripts/run.py train --config-name cifar.vi_cnet --seed 10
python scripts/run.py train --config-name cifar.vi_cnet --seed 11
python scripts/run.py train --config-name cifar.vi_cnet --seed 12

python scripts/run.py train --config-name cifar.sr_cnet --seed 10
python scripts/run.py train --config-name cifar.sr_cnet --seed 11
python scripts/run.py train --config-name cifar.sr_cnet --seed 12

python scripts/run.py train --config-name cifar.abp_cnet --seed 10
python scripts/run.py train --config-name cifar.abp_cnet --seed 11
python scripts/run.py train --config-name cifar.abp_cnet --seed 12