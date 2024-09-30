Environment Setup
=================

```{eval-rst}
.. note::
    Running AMPLFI out-of-the-box requires access to an enterprise-grade GPU(s) (e.g. P100, V100, T4, A[30,40,100], etc.). There are several nodes on the LIGO Data Grid which meet these requirements_**.
```


Please see the [ml4gw quickstart](https://github.com/ml4gw/quickstart/) for help on setting up your environment 
on the [LIGO Data Grid](https://computing.docs.ligo.org/guide/computing-centres/ldg/) (LDG). This quickstart includes a comprehensive Makefile and instructions for setting up all of the necessary software, environment variables, and credentials required to access gravitational wave strain data and run `AMPLFI`.

Once setup, create a personal [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of this repository, and clone it.


```{eval-rst}
.. note::
    Ensure that you have added a [github ssh key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to your account_**
```

`AMPLFI` currently utilizes `git` [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Make sure to initialize and update those:

```bash
git submodule update --init
```


When pulling changes from this repository, it's recommended to use the `--recurse-submodules` flag to pull any updates from the submodules as well.

Next, install the base `amplfi` package, which is used for initializing `amplfi` runs. 

```{eval-rst}
.. tabs::

   .. tab:: Poetry

      .. code-block:: console

          $ poetry install

   .. tab:: Pip

      .. code-block:: console

          $ pip install . -e
```


## Building Containers
`AMPLFI` workflows can be run via `apptainer` images for ease of reproducibility.

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
