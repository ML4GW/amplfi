Environment Setup
=================


## Installation
`AMPLFI` can be installed with `pip`

```console
pip install amplfi
```

or directly from source

```console
git clone git@git.ligo.org:ml4gw/amplfi.git
cd amplfi
pip install .
```

## Containers
`AMPLFI` also provides a container 

To build the containers, first set the `$AMPLFI_CONTAINER_ROOT` environment variable
to a location where you want the images to be stored, e.g. `~/amplfi/images`. It is recommended to also add this 
environment variable to your `~/.bash_profile` (or your shells equivalent).

To build the `train` and `data` containers, make sure you are in the respective projects home directory. For example, starting from the 
`amplfi` repositories home directory the `data` image can be built by running the following"

```console
cd projects/data
apptainer build $AMPLFI_CONTAINER_ROOT/data.sif apptainer.def
```

Make sure you do the equivalent for the `train` image!
