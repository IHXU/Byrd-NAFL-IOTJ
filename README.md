# A Byzantine-Robust Federated Learning with Nesterov Momentum

Lihan Xu, Xiaoyi Fan, Gang Wang, Runhao Zeng, Xiping Hu, and Yanjie Dong.

[Paper (arXiv)](https://arxiv.org/abs/2511.02657v4) 

## Abstract

We investigate robust federated learning, where a group of workers collaboratively train a shared model under the orchestration of a central server in the presence of Byzantine adversaries capable of arbitrary and potentially malicious behaviors. To improve communication efficiency while retaining resilience against such adversaries, we propose a Byzantine-resilient Nesterov-augmented federated learning (Byrd-NAFL) algorithm. Byrd-NAFL applies Nesterov’s momentum at the server to the output of any soft $\gamma$-Byzantine resilient aggregation rule, while each worker only computes and uploads a stochastic gradient, which keeps the workers stateless, as desired for intermittently participating Internet of Things devices. For smooth non-convex losses, we establish a finite-time convergence guarantee for all momentum factors $\beta \in [0,1)$ without requiring the aggregated gradient to be unbiased or bounded; the bound reduces to that of Byzantine-resilient stochastic gradient descent at $\beta=0$ and is tighter than that of server-side heavy-ball momentum. Experiments on MNIST, CIFAR-10, and Sentiment140 show that Byrd-NAFL converges faster and attains higher accuracy than existing benchmarks in most i.i.d. and non-i.i.d. setups under diverse attacks, while its resilience is bounded by that of the underlying aggregation rule.

## Installation

Python 3.10 or newer is required.

```bash
cd /path/to/byrd-nafl
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

```bash
cd /path/to/byrd-nafl
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python run_experiment.py --config "<path/to/config.yaml>"
```

IID and non-IID configurations are available in `configs/config_iid/` and `configs/config_noniid/`.

To run a group of configurations:

```bash
cd /path/to/byrd-nafl
source .venv/bin/activate
python pool_run.py --config-dir "<path/to/config-directory>" --gpus 0,1
```

MNIST and CIFAR-10 download automatically. For Sentiment140, obtain the CSV files from the [dataset page](https://help.sentiment140.com/for-students) and place them in `data_cache/sentiment140/`:

```text
training.1600000.processed.noemoticon.csv
testdata.manual.2009.06.14.csv
```

## Supported attacks and aggregation rules

| Attack | Class |
| --- | --- |
| Fall of Empires (FOE) | `foe.FOE` |
| Label-Flipping (LF) | `label_flipping.LabelFlipping` |
| Sign-Flipping (SF) | `sign_flipping.SignFlipping` |
| Mimic (MC) | `mimic.Mimic` |
| A Little Is Enough (ALIE) | `alie.ALIE` |
| Inner Product Manipulation (IPM) | `ipm.IPM` |
| Min-Max | `minmax.MinMax` |
| Min-Sum | `minsum.MinSum` |

| Aggregation rule | Class |
| --- | --- |
| Mean | `mean.MeanAggregator` |
| Coordinate-wise Median (CwMed) | `median.MedianAggregator` |
| Geometric Median (GeoMed) | `geometric_median.GeometricMedianAggregator` |
| Krum | `krum.KrumAggregator` |
| Bulyan | `bulyan.BulyanAggregator` |
| Centered Clipping (CC) | `centered_clipping.CenteredClippingAggregator` |
| Trimmed Mean | `trimmed_mean.TrimmedMeanAggregator` |

## Citation

```bibtex
@misc{xu2026byrdnafl,
  title={A Byzantine-Robust Federated Learning with Nesterov Momentum},
  author={Xu, Lihan and Fan, Xiaoyi and Wang, Gang and Zeng, Runhao and Hu, Xiping and Dong, Yanjie},
  year={2026},
  eprint={2511.02657},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2511.02657v4}
}
```
