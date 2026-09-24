# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/franka_rl

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/franka_rl/franka_rl/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Robustness evaluation suite

Use the one-seed calibration sweep first to identify perturbation levels that
produce measurable degradation without making the task universally impossible.
It runs 21 isolated Isaac Sim jobs with paired target sets.

```bash
export FRANKA_RL_DATA_ROOT=/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data

direnv exec /home/chen-lab/isaac/franka-rl \
  /home/chen-lab/isaac/.venv/bin/python -u \
  scripts/experiments/run_evaluation_suite.py \
  --suite source/franka_rl/franka_rl/config/experiments/robustness_calibration.yaml
```

After the broad sweep, run the targeted final calibration to resolve stronger
stiffness, mass, observation-bias, and task-space-noise levels. Its artifacts
also include continuous position-error, action-magnitude, overshoot, and soft
joint-limit-margin metrics.

```bash
direnv exec /home/chen-lab/isaac/franka-rl \
  /home/chen-lab/isaac/.venv/bin/python -u \
  scripts/experiments/run_evaluation_suite.py \
  --suite source/franka_rl/franka_rl/config/experiments/robustness_calibration_final.yaml
```

The full nominal-policy protocol evaluates seven fixed scenarios over five
paired seeds (35 isolated Isaac Sim jobs). Targets are replayed exactly across
scenarios, failed jobs are resumable, and compiled tables include both
cross-policy comparisons and scenario-minus-nominal degradation.

```bash
export FRANKA_RL_DATA_ROOT=/media/chen-lab/84BABCB7BABCA6D81/Yifan/franka-rl-data

direnv exec /home/chen-lab/isaac/franka-rl \
  /home/chen-lab/isaac/.venv/bin/python -u \
  scripts/experiments/run_evaluation_suite.py \
  --suite source/franka_rl/franka_rl/config/experiments/robustness_nominal.yaml
```

The output directory contains `jobs/`, `target_sets/`, and these compiled
tables: `jobs.csv`, `scenario_summary.csv`, `robustness_degradation.csv`,
`policy_comparison.csv`, and `target_pairing.csv`.

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/franka_rl"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
