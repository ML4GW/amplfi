Running in Containers
=====================
`AMPLFI` also provides a container build available at ghcr.io/ml4gw/amplfi.

You can pull the container locally with either docker or apptainer

```{eval-rst}
.. tabs::

  .. tab:: apptainer

    .. code-block:: console

        $ apptainer pull docker://ghcr.io/ml4gw/amplfi/data:main

    Supported python versions: 3.9-3.12.

  .. tab:: docker

    .. code-block:: console

      $ docker pull ghcr.io/ml4gw/amplfi:main

    Supported python versions: 3.9-3.12.
```

All `AMPLFI` commands can now be run inside the container. 
