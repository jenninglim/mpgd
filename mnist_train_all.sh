#!/bin/bash
python scripts/run.py train --config-name mnist.pgd_mlp --seed 10
python scripts/run.py train --config-name mnist.pgd_mlp --seed 11
python scripts/run.py train --config-name mnist.pgd_mlp --seed 12

python scripts/run.py train --config-name mnist.mpd_mlp --seed 10
python scripts/run.py train --config-name mnist.mpd_mlp --seed 11
python scripts/run.py train --config-name mnist.mpd_mlp --seed 12

python scripts/run.py train --config-name mnist.vi_mlp --seed 10
python scripts/run.py train --config-name mnist.vi_mlp --seed 11
python scripts/run.py train --config-name mnist.vi_mlp --seed 12

python scripts/run.py train --config-name mnist.sr_mlp --seed 10
python scripts/run.py train --config-name mnist.sr_mlp --seed 11
python scripts/run.py train --config-name mnist.sr_mlp --seed 12

python scripts/run.py train --config-name mnist.abp_mlp --seed 10
python scripts/run.py train --config-name mnist.abp_mlp --seed 11
python scripts/run.py train --config-name mnist.abp_mlp --seed 12
