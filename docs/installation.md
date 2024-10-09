Installation
============

`AMPLFI` can be installed with `pip`

```console
pip install amplfi
```

and directly from source with either pip or poetry

```{eval-rst}
.. tabs::
  
  .. tab:: Pip
    
    .. code-block:: console

      $ git clone git@git.ligo.org:ml4gw/amplfi.git
      $ cd amplfi
      $ pip install .

    Supported python versions: 3.9-3.12.

  .. tab:: Poetry

    .. code-block:: console

      $ git clone git@git.ligo.org:ml4gw/amplfi.git
      $ cd amplfi
      $ poetry install

    Supported python versions: 3.9-3.12.
```

It is highly recommended that you install `AMPLFI` in a virtual environment like `conda` or `venv`.

## Data Generation
Currently, running the data generation workflow that queries strain data requires utilizing the `AMPLFI` container.
First, set the `$AMPLFI_CONTAINER_ROOT` where you would like the image stored.

```console
export $AMPLFI_CONTAINER_ROOT=~/amplfi/images
```

Then you can either pull the container from the remote github repository

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

Or build the container locally

```cnosle
$ git clone git@git.ligo.org:ml4gw/amplfi.git
$ cd amplfi
$ apptainer build $AMPLFI_CONTAINER_ROOT/amplfi.sif apptainer.def
```

This container can be also be used for launching the training
and tuning pipelines, but is not strictly required. See the [container
documentation](./containers.md) for more information.
