Installation
============

`AMPLFI` can be installed with `pip` from PyPI 

```console
$ pip install amplfi
```

or directly from source


```console
$ git clone git@git.ligo.org:ml4gw/amplfi.git
$ cd amplfi
$ pip install .  
```

It is highly recommended that you install `AMPLFI` in a virtual environment using tools
like `conda`, `venv`, `uv`, or `poetry`. `AMPLFI` is managed using `uv`, which is the tool
we recommend.

```{eval-rst}
  .. tab:: uv

      .. code-block:: console

        $ git clone git@git.ligo.org:ml4gw/amplfi.git
        $ cd amplfi
        $ uv sync

  .. tab:: Poetry

      .. code-block:: console

        $ git clone git@git.ligo.org:ml4gw/amplfi.git
        $ cd amplfi
        $ poetry install

  .. tab:: venv

      .. code-block:: console

        $ git clone git@git.ligo.org:ml4gw/amplfi.git
        $ cd amplfi
        $ python -m venv ./venv 
        $ source ./venv/bin/activate 

      Supported python versions: 3.9-3.12.
```

## Data Generation
Currently, running the data generation workflow that queries strain data requires utilizing the `AMPLFI` container.
First, set the `$AMPLFI_CONTAINER_ROOT` where you would like the image stored.

```console
export AMPLFI_CONTAINER_ROOT=~/amplfi/images
```

Then you can either pull the container from the remote github repository

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

Or build the container locally

```cnosle
$ git clone git@git.ligo.org:ml4gw/amplfi.git
$ cd amplfi
$ apptainer build $AMPLFI_CONTAINER_ROOT/amplfi.sif apptainer.def
```

This container can be also be used for launching the training
and tuning pipelines, but is not strictly required. See the [container
documentation](./containers.md) for more information.
