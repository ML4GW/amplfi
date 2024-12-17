Remote Training
===============

```{eval-rst}
    .. note::
        Please see the `ml4gw quickstart <https://github.com/ml4gw/quickstart/>`_ for help installing the necessary tools ( :code:`helm`,  :code:`kubernetes`,  :code:`s3cmd`) and configuration (weights and biases, s3 credentials) to run remote tuning. This quickstart includes a comprehensive Makefile to install this 
        tooling in a fresh conda environment, and instructions on settting up necessary credentials.
```

## Initialize a Remote Training Experiment
A remote training experiment can be initialized with the `amplfi-init` command
by supplying the optional `--remote-train` and `--s3-bucket` flags.

For example, to initialize a directory to train a flow, run

```console
> amplfi-init --mode flow --pipeline train --directory ~/amplfi/ -n my-first-remote-train --remote-train true --s3-bucket s3://my_bucket/my-first-remote-train/
INFO - Initialized a flow train pipeline at /home/albert.einstein/amplfi/my-first-remote-train
```

The directory contents will look similar to those created for local training jobs. 
For example you will see a `train.yaml` training configuration file, and a `run.sh` file 
for launching the job.

You will also now see a `kubernetes.yaml` file that contains the kuberenetes configuration
for launching the kubernetes pod on nautilus. This file will be filled out with configuration based on
the `s3_bucket` specified. For example, `AWS_ENDPOINT_URL` and `WANDB_API_KEY` will be set inside the remote 
container based on your local environment variables. 

In addition, `AMPLFI_OUTDIR` and `AMPLFI_DATADIR` environment variables will be set inside the container
based on the specified `s3_bucket`:

```yaml 
# snip
  env:
  # snip
  - name: AMPLFI_OUTDIR
      value: s3://my_bucket/my-first-remote-train
  - name: AMPLFI_DATADIR
      value: s3://my_bucket/my-first-remote-train/data
```

```{eval-rst}
    .. note::
        If you already have a remote data directory you wish to train with, you can specify the 
        :code:`AMPLFI_DATADIR` environment variable in the :code:`run.sh` and :code:`kubernetes.yaml` 
        to point to your data directory.
```


The `run.sh` file will look slightly different than the local training job:

```bash
# run.sh

#!/bin/bash
export AMPLFI_DATADIR=s3://my_bucket/my-first-remote-train/data

# launch data generation pipeline
LAW_CONFIG_FILE=/home/ethan.marx/amplfi/my-first-remote-train/datagen.cfg law run amplfi.data.DataGeneration --workers 5

# move config file to remote s3 location
s3cmd put /home/ethan.marx/amplfi/my-first-remote-train/cbc.yaml s3://my_bucket/my-first-remote-train/cbc.yaml

# launch job
kubectl apply -f /home/ethan.marx/amplfi/kubernetes.yaml
```

The first step is generating strain data for training and testing. As usual, if data at the specified `AMPLFI_DATADIR` already exists,
this step will be automatically skipped. Next, the training configuration file will be moved to remote storage 
so that it can be accessed by the kubernetes job. Finally, the kubernetes job will be launched.

To monitor the job, you can run 

```console
kubectl get pods
```

to get the pod name and inspect the status of the pod, and 

```console
kubectl logs <pod-name>
```

to inspect any logs from the pod once its running.

## Configuring Kubernetes Job
By default, the job will utilize the remote `AMPLFI` image at `ghcr.io/ml4gw/amplfi/amplfi:main`.
If for some reason you wish to utilize another image with `AMPLFI` installed, you can change 
the `image` parameter.

The amount of GPUs and CPUs available in the pod can also be configured by editing the `kubernetes.yaml` file"

```yaml
# kubernetes.yaml

# snip
...
    image: ghcr.io/ml4gw/amplfi/amplfi:main
    imagePullPolicy: Always
    name: train
    resources:
        limits:
        cpu: "96"
        memory: 416G
        nvidia.com/gpu: "8"
        requests:
        cpu: "96"
        memory: 416G
        nvidia.com/gpu: "8"
```

By default, 8 gpus are requested. Sometimes it can take a little while for jobs with 8 gpus to be scheduled. 
Decreasing the number of requested GPUs will speed up scheduling time. 
