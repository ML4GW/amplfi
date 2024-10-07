Tuning
======

Hyperparameter tuning is powered by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html). We utilize a wrapper library, [lightray](https://github.com/ethanmarx/lightray), that simplifies utilizing Ray Tune with PyTorch Lightning `LightningCLI`'s. 


# Initialize a Tune Experiment
A new tuning experiment can be initialized using the `amplfi-init` command. 
For example, to initialize a directory to train a flow, run

```console
poetry run amplfi-init --mode flow --pipeline tune --directory ~/amplfi/my-first-tune/ 
```

This will create a directory at `~/amplfi/my-first-tune/`, and populate it with 
configuration files for the run. The `train.yaml` contains the main configuration for the training.
`datagen.cfg` controls the configuration for querying training and testing strain data. 
`tune.yaml` configure parameters that control how the hyperparameter tuning is performed. Finally,
`search_space.py` constructs the space of parameters that will searched over during tuning. 


# Configuring an Experiment
The search space of parameters to tune over can be set in the `search_space.py` file. 
For example, the below parameter space will search over the models learning rate 
and the kernel length of the data.

```
# search_space.py
from ray import tune

space = {
    "model.learning_rate": tune.loguniform(1e-4, 1e-1),
    "data.kernel_length": tune.choice([1, 2])
}
```

the parameter names correspond to attributes in the `train.yaml`. Any
parameters set in the search space will be sampled from the distribution
when each trial is launched, and override the value set in `train.yaml`.

The `tune.yaml` file configures parameters of the tuning. You can see a full list of configuration by running 

```
poetry run --directory /home/albert.einstein/path/to/amplfi/projects/train python /home/albert.einstein/path/to/amplfi/projects/train/train/tune/tune.py --help
```

Currently, the `lightray` library uses the `Asynchronous Hyper Band algorithm`(https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.AsyncHyperBandScheduler.html#ray.tune.schedulers.AsyncHyperBandScheduler), which will kill under performing trials after a certain amount of epochs, controlled by the `min_epochs` parameter.


# Launching a Run
The entrypoint to the tuning pipeline is the `run.sh` file:

```
# run.sh

#!/bin/bash
# Export environment variables
export AMPLFI_DATADIR=/home/ethan.marx/amplfi/my-first-tune
export AMPLFI_OUTDIR=/home/ethan.marx/amplfi/my-first-tune/runs/
export AMPLFI_CONDORDIR=/home/ethan.marx/amplfi/my-first-tune/condor

CUDA_VISIBLE_DEVICES=0

# launch the data generation pipeline
LAW_CONFIG_FILE=/home/albert.einstein/amplfi/my-first-tune/datagen.cfg poetry run --directory /home/albert.einstein/path/to/amplfi/law law run amplfi.law.DataGeneration --workers 5

# launch training or tuning pipeline
poetry run --directory /home/albert.einstein/path/to/amplfi/projects/train python /home/albert.einstein/path/to/amplfi/projects/train/train/tune/tune.py --config tune.yaml
```

If you've run the [training pipeline](first_pipeline.md) this should look familiar: environment variables control the location where 
data is stored and where the tuning runs will be stored. There's a command to launch the data generation pipeline, followed by a command to launch the tuning job.


## Local Tuning
If the `address` parameter in the `tune.yaml` is set to `null` (it is by default), then a local Ray cluster will be initialized.
The tuning will then use local resources. The amount of resources to be alloated per trial can be controlled by the 
`gpus_per_worker`, and `cpus_per_gpu` arguments. The `CUDA_VISIBLE_DEVICES` environment variable will control the available GPU resources
exposed to the job.

## Remote Tuning
The tuning can also be performed via a remote Ray cluster. 
