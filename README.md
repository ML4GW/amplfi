# AMPLFI
**Accelerated Multi-messenger Parameter estimation with Likelihood Free Inference**

Framework for performing rapid parameter estimation of gravitational wave events using likelihood free inference

## Environment Setup
> **_Note: this repository is a WIP. Please open up an issue if you encounter bugs, quirks, or any undesired behavior_**

> **_Note: Running AMPLFI out-of-the-box requires access to an enterprise-grade GPU(s) (e.g. P100, V100, T4, A[30,40,100], etc.). There are several nodes on the LIGO Data Grid which meet these requirements_**.

Please see the [ml4gw quickstart](https://github.com/ml4gw/quickstart/) for help on setting up your environment 
on the [LIGO Data Grid](https://computing.docs.ligo.org/guide/computing-centres/ldg/) (LDG) and for configuring access to [Weights and Biases](https://wandb.ai), and the [Nautilus hypercluster](https://ucsd-prp.gitlab.io/). 
This quickstart includes a Makefile and instructions for setting up all of the necessary software, environment variables, and credentials required to run `AMPLFI`. 

Once setup, create a [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of this repository, and clone it.

> **_Note: Ensure that you have added a [github ssh key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to your account_**

```bash
git clone git@github.com:albert-einstein/pe.git
```

`AMPLFI` utilizes `git` [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Make sure to initialize and update those

```bash
git submodule update --init
```

When pulling changes from this repository, it's recommended to use the `--recurse-submodules` flag to pull any updates from the submodules as well.

Next, install the `amplfi.law` submodule, which is used for launching `amplfi` workflows with [`law`](https://github.com/riga/law)

```console
cd amplfi/law
poetry install
```

Finally, build the `train` and `data` project apptainer images. Set the `$AMPLFI_CONTAINER_ROOT` environment variable
to a location where you want the images to be stored, e.g. `~/amplfi/images`

```console
apptainer build $AMPLFI_CONTAINER_ROOT/data.sif projects/data/apptainer.def
apptainer build $AMPLFI_CONTAINER_ROOT/train.sif projects/train/apptainer.def
```

## Generating Data
Training and testing background strain data can be generated with the `amplfi.law.DataGeneration` workflow.

```console
cd amplfi/law/

export AMPLFI_DATADIR=~/amplfi/my-first-run/data/
export AMPLFI_CONDORDIR=~/amplfi/my-first-run/condor

LAW_CONFIG_FILE=config.cfg poetry run law run amplfi.law.DataGeneration --workers 2
```

An example configuration file for the strain data generation can be found at `amplif/law/config.cfg`


## Training
With training data in hand, a flow training run can be launched via the `train` poetry environment:

```console
cd amplfi/projects/train/
export AMPLFI_OUTDIR=~/amplfi/my-first-run/training/
poetry run python train/cli/flow.py --config configs/flow/cbc.yaml
```
