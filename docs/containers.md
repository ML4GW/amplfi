Running in Containers
=====================
`AMPLFI` also provides a container build available via the github container repository

You can pull the container locally with either docker or apptainer

```{eval-rst}
.. tabs::

  .. tab:: apptainer

    .. code-block:: console

        $ apptainer pull $AMPLFI_CONTAINER_ROOT/amplfi.sif docker://ghcr.io/ml4gw/amplfi/amplfi:main

    Supported python versions: 3.9-3.12.

  .. tab:: docker

    .. code-block:: console

      $ docker pull ghcr.io/ml4gw/amplfi/amplfi:main

    Supported python versions: 3.9-3.12.
```

`AMPLFI` train and tune commands can now be run inside the container. For example, 
to train a flow inside the container, you can run 

```console
APPTAINERENV_AMPLFI_OUTDIR=/path/to/outdir APPTAINERENV_AMPLFI_OUTDIR=/path/to/datadir/ \
    apptainer run $AMPLFI_CONTAINER_ROOT/amplfi.sif --nv \ 
    amplfi-flow-cli fit --config /path/to/config.yaml
```

Note that we must map in the `APPTAINERENV_AMPLFI_OUTDIR` and `APPTAINERENV_AMPLFI_DATADIR` environment variables
so they can be accessed inside the container.


## Mounting Local Code
During development, often times you will make code changes and want to test them inside the container.
To do so, you *could* rebuild the entire container locally so that your changes are reflected in the container,

```
apptainer build $AMPLFI_CONTAINER_ROOT/amplfi.sif apptainer.def
```

but it can be quite cumbersome to do this each time you make a minor tweak to the code base.

Fortunately, `AMPLFI` is installed editably inside the container at the path `/opt/amplfi`. So,
it is possible to map local changes into the container at runtime by bind mounting your local repository. As an example, the above training command can be modified to do so using the `-B` flag (replace `/path/to/local/amplfi` with the location where your local amplfi lives).

```console 
APPTAINERENV_AMPLFI_OUTDIR=/path/to/outdir APPTAINERENV_AMPLFI_OUTDIR=/path/to/datadir/ \
    apptainer run -B /path/to/local/amplfi:/opt/amplfi --nv \
    $AMPLFI_CONTAINER_ROOT/amplfi.sif amplfi-flow-cli fit \
    --config /path/to/config.yaml
```

```{eval-rst}
.. note::
   If you make modifications the the environment like adding python dependencies you will
   have to rebuild the containers!
```
