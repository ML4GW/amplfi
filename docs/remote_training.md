Remote Training
===============

## Initialize a Remote Training Experiment
A remote training experiment can be initialized with the `amplfi-init` command
by supplying the `--remote-train` and `--s3-bucket` flags.

For example, to initialize a directory to train a flow, run

```console
> amplfi-init --mode flow --pipeline train --directory ~/amplfi/ -n my-first-remote-train --remote-train true --s3-bucket s3://my_bucket/my-first-remote-train/
INFO - Initialized a flow train pipeline at /home/albert.einstein/amplfi/my-first-remote-train
```

The directory contents will look similar to those created for local training jobs. 
For example you will see a `train.yaml` training configuration file, and a `run.sh` file 
for launching the job.
