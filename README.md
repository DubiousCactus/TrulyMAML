# Model-Agnostic Meta-Learning

This is a custom implementation of the paper [Model-Agnostic Meta-Learning (Finn et
al.)](https://arxiv.org/abs/1703.03400), using [Higher](https://github.com/facebookresearch/higher)
for second-order optimization, thus making this framework **truly** model-agnostic. Compared to
other implementations, the optimizee does not need to be constructed for MAML, you can just plug in
any PyTorch model into `MAML`!


## Usage

```
usage: main.py [-h] [--checkpoint_path CHECKPOINT_PATH] [--load LOAD] [--eval] [--samples SAMPLES] [-k K] [-q Q] [-n N] [-s S]
               [--dataset {omniglot,sinusoid,harmonic}] [--meta-batch-size META_BATCH_SIZE] [--iterations ITERATIONS]

Model-Agnostic Meta-Learning

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
                        path to checkpoint saving directory
  --load LOAD           path to model checkpoint
  --eval                Evaluation moed
  --samples SAMPLES     Number of samples per task. The resulting number of test samples will be this value minus <K>.
  -k K                  Number of shots for meta-training
  -q Q                  Number of meta-testing samples
  -n N                  Number of classes for n-way classification
  -s S                  Number of inner loop optimization steps during meta-training
  --dataset {omniglot,sinusoid,harmonic}
  --meta-batch-size META_BATCH_SIZE
                        Number of tasks per meta-update
  --iterations ITERATIONS
                        Number of outer-loop iterations
```