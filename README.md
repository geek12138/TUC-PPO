# TUC-PPO
## TUC-PPO: Team Utility-Constrained Proximal Policy Optimization for spatial public goods games

We introduce Team Utility-Constrained PPO (TUC-PPO), a new deep reinforcement learning framework. It extends Proximal Policy Optimization (PPO) by integrating team welfare objectives specifically for spatial public goods games. Unlike conventional approaches where cooperation emerges indirectly from individual rewards, TUC-PPO instead optimizes a bi-level objective integrating policy gradients and team utility constraints. Consequently, all policy updates explicitly incorporate collective payoff thresholds. The framework preserves PPO’s policy gradient core while incorporating constrained optimization through adaptive Lagrangian multipliers. Therefore, decentralized agents dynamically balance selfish and cooperative incentives. The comparative analysis demonstrates superior performance of this constrained deep reinforcement learning approach compared to unmodified PPO and evolutionary game theory baselines. It achieves faster convergence to cooperative equilibria and greater stability against invasion by defectors. The framework formally integrates team objectives into policy updates. This work advances multi-agent deep reinforcement learning for social dilemmas while providing new computational tools for evolutionary game theory research.

## Requirements
It is worth mentioning that because python runs slowly, we use cuda library to improve the speed of code running.

```
* Python Version 3.12.2
* CUDA Version: 12.8
* torch Version: 2.2.1
* numpy Version: 1.26.4
* pandas Version: 2.2.3
```

## Installation
```bash
conda env create -f environment.yaml
```

## Usage
run scripts of run_one_PPO_TUC.sh:
```bash
sh scripts/run_one_PPO.sh
```

## Citation

If you use our codebase or models in your research, please cite this work.

```
@article{YANG_2025_116928,
	title = {TUC-PPO: Team Utility-Constrained Proximal Policy Optimization for spatial public goods games},
	journal = {Chaos, Solitons \& Fractals},
	volume = {199},
	pages = {116928},
	year = {2025},
	issn = {0960-0779},
	doi = {https://doi.org/10.1016/j.chaos.2025.116928},
	author = {Zhaoqilin Yang and Xin Wang and Ruichen Zhang and Chanchan Li and Youliang Tian}
}
```
