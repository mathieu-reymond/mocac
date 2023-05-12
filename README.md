# Actor-critic multi-objective reinforcement learning for non-linear utility functions

This is the code repository for:

> Reymond, M., Hayes, C. F., Steckelmacher, D., Roijers, D. M., & Nowé, A. (2023). Actor-critic multi-objective reinforcement learning for non-linear utility functions. Autonomous Agents and Multi-Agent Systems, 37(2), 23.

- The algorithm implementation is in `agents/mocac.py`

The code is written in Python and uses Pytorch.

> :warning: **This repository uses an old version of Gym, before being part of the Farama foundation**. The additional multi-objective environments (in `envs/`) are based on the old API.

## Running the code

This repository contains run-scripts for all experiments of the paper.
For example, you can run MOCAC on the Split-environment as follows:

```bash
python train_split.py --algo mocac
```

(the `--algo` flag can be set to `moreinforce|mocac|moac`).

The logs are then put in a `runs/` folder.

You can plot the results, based on the logs as follows:

```bash
python plot_split.py runs/split_env/tunnel_10 moreinforce runs/split_env/tunnel_10/mocac runs split_env/tunnel_10/moac
```

Where you append, for each algorithm, the folder containing all this algorithm's logs.