## Mixture of Experts - Pytorch

A Pytorch implementation of <a href="https://arxiv.org/abs/2006.16668">Mixture of Experts</a>, for enhancing attention networks. It will pretty much be a line-by-line transcription of the tensorflow implementation <a href="https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/moe.py">here</a>.

## Install

```bash
$ pip install mixture_of_experts
```

## Usage

```python
import torch
from torch import nn
from mixture_of_experts import MoE

inputs = torch.randn(4, 1024, 512)

experts = MoE(
    dim = 512,
    num_experts = 16,           # increase the experts (# parameters) of your model without increasing computation
    hidden_dim = 512 * 4,       # size of hidden dimension in each expert, defaults to 4 * dimension
    activation = nn.LeakyReLU   # use your preferred activation, will default to ReLU
)

out, aux_loss = experts(inputs) # (4, 1024, 512), (1,)
```

## Citation

```bibtex
@misc{lepikhin2020gshard,
    title   = {GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding},
    author  = {Dmitry Lepikhin and HyoukJoong Lee and Yuanzhong Xu and Dehao Chen and Orhan Firat and Yanping Huang and Maxim Krikun and Noam Shazeer and Zhifeng Chen},
    year    = {2020},
    eprint  = {2006.16668},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```
