# KSDAgg package

This package implements the KSDAgg test for goodness-of-fit testing, as proposed in our paper [KSD Aggregated Goodness-of-fit Test](https://arxiv.org/pdf/2202.00824.pdf).
The experiments of the paper can be reproduced using the [ksdagg-paper](https://github.com/antoninschrab/ksdagg-paper/) repository.
The package contains implementations both in Numpy and in Jax, we recommend using the Jax version as it runs more than 500 times faster after compilation (results from the notebook [demo_speed.ipynb](https://github.com/antoninschrab/ksdagg-paper/blob/master/demo_speed.ipynb) in the [ksdagg-paper](https://github.com/antoninschrab/ksdagg-paper/) repository). 
The notebook also contains a demo showing how to use our KSDAgg test.
We also provide installation instructions and example code below.

| Speed in ms | Numpy (CPU) | Jax (CPU) | Jax (GPU) | 
| -- | -- | -- | -- |
| KSDAgg | 12500 | 1470 | 22 | 

## Requirements

The requirements for the Numpy version are:
- `python 3.9`
  - `numpy`
  - `scipy`

The requirements for the Jax version are:
- `python 3.9`
  - `jax`
  - `jaxlib`

## Installation

First, we recommend creating a conda environment:
```bash
conda create --name ksdagg-env python=3.9
conda activate ksdagg-env
# can be deactivated by running:
# conda deactivate
```

We then install the required depedencies by running either:
- for GPU:
  ```bash
  conda install -c conda-forge -c nvidia pip numpy scipy cuda-nvcc "jaxlib=0.4.1=*cuda*" jax
  ```
- or, for CPU:
  ```bash
  conda install -c conda-forge -c nvidia pip numpy scipy cuda-nvcc jaxlib=0.4.1 jax
  ```
  
Our `ksdagg` package can then be installed as follows:
```bash
pip install git+https://github.com/antoninschrab/ksdagg.git
```

## KSDAgg

**Goodness-of-fit testing:** Given arrays X and score_X both of shape $(N, d)$, where score_X is the score of X (i.e. $\nabla p(x)$ where $p$ is the model density), our KSDAggInc test `ksdagg(X, Y)` returns 0 if the samples X are believed to have been drawn from the density $p$, and 1 otherwise.

**Jax compilation:** The first time the function is evaluated, Jax compiles it. 
After compilation, it can fastly be evaluated at any other X and score_X of the same shape. 
If the function is given arrays with new shapes, the function is compiled again.
For details, check out the [demo_speed.ipynb](https://github.com/antoninschrab/ksdagg-paper/blob/master/demo_speed.ipynb) notebook in the [ksdagg-paper](https://github.com/antoninschrab/ksdagg-paper/) repository.

```python
# import modules
>>> import numpy as np 
>>> import jax.numpy as jnp
>>> from ksdagg import ksdagg, human_readable_dict # jax version
>>> # from ksdagg.np import ksdagg

# generate data for goodness-of-fit test
>>> perturbation = 0.5
>>> rs = np.random.RandomState(0)
>>> X = rs.gamma(5 + perturbation, 5, (500, 1))
>>> score_gamma = lambda x, k, theta : (k - 1) / x - 1 / theta
>>> score_X = score_gamma(X, 5, 5)
>>> X = jnp.array(X)
>>> score_X = jnp.array(score_X)

# run KSDAggInc test
>>> output = ksdagg(X, score_X)
>>> output
Array(1, dtype=int32)
>>> output.item()
1
>>> output, dictionary = ksdagg(X, score_X, return_dictionary=True)
>>> output
Array(1, dtype=int32)
>>> human_readable_dict(dictionary)
>>> dictionary
{'KSDAgg test reject': True,
 'Single test 1': {'Bandwidth': 1.0,
  'KSD': 5.788900671177544e-05,
  'KSD quantile': 0.0009193826699629426,
  'Kernel IMQ': True,
  'Reject': False,
  'p-value': 0.41079461574554443,
  'p-value threshold': 0.01699146442115307},
  ...
}
```

## KSDAggInc

For a computationally efficient version of KSDAgg which can run in linear time, check out our package `agginc` in the [agginc](https://github.com/antoninschrab/agginc) repository. 
This package implements the KSDAggInc test (together with MMDAggInc and HISCAggInc) proposed in our paper [Efficient Aggregated Kernel Tests using Incomplete U-statistics](https://arxiv.org/pdf/2206.09194.pdf) with reproducible experiments in the [agginc-paper](https://github.com/antoninschrab/agginc-paper) repository. 

## Contact

If you have any issues running our KSDAgg test, please do not hesitate to contact [Antonin Schrab](https://antoninschrab.github.io).

## Affiliations

Centre for Artificial Intelligence, Department of Computer Science, University College London

Gatsby Computational Neuroscience Unit, University College London

Inria London

## Bibtex

```
@inproceedings{schrab2022ksd,
  author    = {Antonin Schrab and Benjamin Guedj and Arthur Gretton},
  title     = {KSD Aggregated Goodness-of-fit Test},
  booktitle = {Advances in Neural Information Processing Systems 35: Annual Conference
               on Neural Information Processing Systems 2022, NeurIPS 2022},
  editor    = {Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year      = {2022},
}
```

## License

MIT License (see [LICENSE.md](LICENSE.md)).
