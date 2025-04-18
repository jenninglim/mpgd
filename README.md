# Momentum Particle Maximum Likelihood

Code used in [MPD Paper](https://arxiv.org/abs/2312.07335).

Activate the environment with:

```
pipenv sync
pipenv shell
```

### ToyHM

For relevant scripts, see `scripts/toyhmm`.

### CIFAR and MNIST

Requires: [Weights and Biases](https://wandb.ai/site).

The configurations can be found in `scripts/configs`.

Train with the bash scripts `mnist_train_all.sh` and `cifar_train_all.sh` and evaluate with `mnist_eval_all.sh` and `mnist_train_all.sh`.
