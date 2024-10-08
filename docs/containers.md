Running in Containers
=====================
`AMPLFI` also provides a container build available at ghcr.io/ml4gw/amplfi.

You can pull the container locally with either docker or apptainer

```{eval-rst}
.. tabs::

  .. tab:: apptainer

    .. code-block:: console

        $ apptainer pull docker://ghcr.io/ml4gw/amplfi/data:main $AMPLFI_CONTAINER_ROOT/amplfi.sif

    Supported python versions: 3.9-3.12.

  .. tab:: docker

    .. code-block:: console

      $ docker pull ghcr.io/ml4gw/amplfi:main $AMPLFI_CONTAINER_ROOT/amplfi.sif

    Supported python versions: 3.9-3.12.
```

`AMPLFI` train and tune commands can now be run inside the container. For example, 
to train a flow inside the container, you can run 

```console
APPTAINERENV_AMPLFI_OUTDIR=/path/to/outdir APPTAINERENV_AMPLFI_OUTDIR=/path/to/datadir/ \
    apptainer run $AMPLFI_CONTAINER_ROOT/amplfi.sif --nv amplfi-flow-cli fit --config /path/to/config.yaml
```

Note that we must map in the `APPTAINERENV_AMPLFI_OUTDIR` and `APPTAINERENV_AMPLFI_DATADIR` environment variables
so they can be accessed inside the container.
