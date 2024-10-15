Tuning
======

Hyperparameter tuning is powered by [Ray Tune](https://docs.ray.io/en/latest/tune/index.html). We utilize a wrapper library, [lightray](https://github.com/ethanmarx/lightray), that simplifies the use of Ray Tune with PyTorch Lightning `LightningCLI`'s. 


## Initialize a Tune Experiment
A new tuning experiment can be initialized using the `amplfi-init` command. 
For example, to initialize a directory to train a flow, run

```console
amplfi-init --mode flow --pipeline tune --directory ~/amplfi/my-first-tune/ 
```

This will create a directory at `~/amplfi/my-first-tune/`, and populate it with 
configuration files for the run. The `train.yaml` contains the main configuration for the training.
`datagen.cfg` controls the configuration for querying training and testing strain data. 
`tune.yaml` configure parameters that control how the hyperparameter tuning is performed. Finally,
`search_space.py` constructs the space of parameters that will searched over during tuning. 


## Configuring an Experiment
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

the parameter names should be python "dot path" to attributes in the `train.yaml`. Any
parameters set in the search space will be sampled from the distribution
when each trial is launched, and override the value set in `train.yaml`.

The `tune.yaml` file configures parameters of the tuning. You can see a full list of configuration by running 

```
amplfi-tune --help
```

```{eval-rst}
.. note::
    Currently, the `lightray` library automatically uses the `Asynchronous Hyper Band algorithm <https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.AsyncHyperBandScheduler.html#ray.tune.schedulers.AsyncHyperBandScheduler>`_, which will kill under performing trials after a certain amount of epochs this is
    controlled by the :code:`min_epochs` parameter.
```

## Launching a Run
The entrypoint to the tuning pipeline is the `run.sh` file generated in the experiment directory.

```bash
# run.sh

#!/bin/bash
# Export environment variables
export AMPLFI_DATADIR=/home/albert.einstein/amplfi/my-first-tune
export AMPLFI_OUTDIR=/home/albert.einstein/amplfi/my-first-tune/runs/
export AMPLFI_CONDORDIR=/home/albert.einstein/amplfi/my-first-tune/condor

CUDA_VISIBLE_DEVICES=0

# launch the data generation pipeline
LAW_CONFIG_FILE=/home/albert.einstein/amplfi/my-first-tune/datagen.cfg law run amplfi.law.DataGeneration --workers 5

# launch training or tuning pipeline
amplfi-tune --config tune.yaml
```

If you've run the [training pipeline](first_pipeline.md) this should look familiar: environment variables control the location where 
data is stored and where the tuning runs will be stored. There's a command to launch the data generation pipeline, followed by a command to launch the tuning job.


## Local Tuning
If the `address` parameter in the `tune.yaml` is set to `null` (the default), then a local Ray cluster will be initialized.
The tuning will then use local resources. The amount of resources to be alloated per trial can be controlled by the 
`gpus_per_worker`, and `cpus_per_gpu` arguments. The `CUDA_VISIBLE_DEVICES` environment variable will control the available GPU resources
exposed to the job.

## Remote Tuning
Tuning can also be performed via a remote Ray cluster. Assuming you have properly set up your cluster worker nodes with access
to a remote data directory on `s3`, and weights and biases (more on this below), then launching a remote tuning job is as simple as passing the
ip address of your Ray clusters head node to the `address` variable. 

Running tuning remotely will require that your data directory live on an `s3` storage system. To generate data
that is autmoatically moved to an `s3` bucket, you can simply set the `AMPLFI_DATADIR` environment variable to an `s3` path
in the `run.sh`! You'll also need to set the `AMPLFI_OUTDIR` to an `s3` location.

```bash
# run.sh
export AMPLFI_DATADIR=s3://my-bucket/my-first-tune/data
export AMPLFI_OUTDIR=s3://my-bucket/my-first-tune/runs
...
```

### Kubernetes Ray Cluster
```{eval-rst}
    .. note::
        Please see the `ml4gw quickstart <https://github.com/ml4gw/quickstart/>`_ for help installing the necessary tools ( :code:`helm`,  :code:`kubernetes`,  :code:`s3cmd`) and configuration (weights and biases, s3 credentials) to run remote tuning. This quickstart includes a comprehensive Makefile to install this 
        tooling in a fresh conda environment, and instructions on settting up necessary credentials.
```

`lightray` ships a `helm` chart that can be used to launch a ray head and worker nodes on a remote kubernetes cluster.

First, add the helm repository

```console
helm repo add lightray https://ethanmarx.github.io/lightray/
```

The helm chart comes with some configuration you'll need to set. To pull the "values" configuration template, run

```console
helm show values lightray/ray-cluster >> values.yaml
```

Specifically, you'll need to set the container to the remote `amplfi` image

```yaml
image: ghcr.io/ml4gw/amplfi/amplfi:main
```

And you'll also need to set your `WANDB_API_KEY`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`
to the corresponding variable so that the remote cluster can access your data on s3, and upload to weights and biases.


Then, you can install the cluster. You can name the installation anything. Here we name it `my-ray-cluster`

```console
helm install my-ray-cluster lightray/ray-cluster -f values.yaml
```

To monitor the status of your pods, run 

```console
kubectl get pods
```

You should see something like 

```console
NAME                                   READY   STATUS              RESTARTS   AGE
my-ray-cluster-head-7b9597fdd8-brrlm    0/1     ContainerCreating   0          2s
my-ray-cluster-worker-bd6698d67-49p6x   0/1     ContainerCreating   0          2s
```

Once the head and at least one worker pod are in the `RUNNING` state, you can query the 
kubernetes Service corresponding to the head node for it's ip address:

```console
$ kubectl get service my-ray-cluster-head-loadbalancer -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

pass this ip address, to the `address` parameter in `tune.yaml` with the format `ray://{ip}:10001`. For example,
if the ip address was `11.22.10.27` you would set

```yaml
address = ray://11.22.10.27:10001
``` 

Now, launch the run!

```console
amplfi-tune --tune.yaml
```

```{eval-rst}
.. note::
   Remember to clean up your kubernetes jobs! You can uninstall all resources
   created by the helm chart with :code:`helm uninstall {chart-name}`
```

#### Syncing Remote Code 
In some cases, it is necessary to launch a tuning job with code changes that haven't been integrated into the `AMPLFI` `main` branch,
and thus have not been pushed to the remote container.

To allow this, the `lightray/ray-cluster` chart supports an optional [git-sync](https://github.com/kubernetes/git-sync) `initContainer`
that will clone and mount remote code inside the kubernetes pods.

To use this with `AMPLFI`, you will need to configure the following in the charts `values.yaml` file

```yaml
# set dev to true
dev: true

gitRepo:
    # name must be set to amplfi
    name: amplfi
    # set to repo you want to mount
    url: git@github.com:albert.einstein/amplfi.git
    # set ref to branch name or commit hash
    ref: my-branch
    # mountPath must be set to /opt
    mountPath: /opt
```
