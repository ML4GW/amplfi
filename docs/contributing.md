Contributing
============

## Environment Setup

First, create a personal [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of this repository, and clone it.

Then, add the main (`ML4GW`) repository as a remote reference. A common practice is to rename this remote `upstream`.

```console
git remote add upstream git@github.com:ML4GW/amplfi.git
```

Now, install `AMPLFI` editably. It is recommended to do so in a virtual environment.

```{eval-rst}
.. tabs::
   .. tab:: Pip

      .. code-block:: console

          $ pip install -e .

   .. tab:: uv
      
      .. code-block:: console

          $ uv sync

   .. tab:: Poetry

      .. code-block:: console

          $ poetry install

      Supported python versions: 3.9-3.12.
```

## Contribution guidelines

To begin your contribution, checkout to a new branch for a _specific_ issue you're trying to solve

```console
git checkout -b new-feature
```

you can now edit files, and make commits!

### Pre-commit hooks
To keep the code style consistent and neat throughout the repo, we implement [pre-commit hooks](https://pre-commit.com/) to statically lint and style-check any code that wants to get added to the upstream `main` branch. `pre-commit` is already installed in the `Aframe` environment, so you can run 

```console
pre-commit install
```

to install the hooks.

Now any attempts to commit new code will require these tests to past first (and even do some reformatting for you if possible). To run the hooks on existing code, you can run 

```console
pre-commit run --all
```

### Docstring guidelines
- Annotate function arguments and returns as specifically as possible
- Adopt [Google docstring](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) formatting (this will eventually be used by Sphinx autodoc, so consistency is important)

## Adding your code
Once you've added all the code required to solve the issue you set out for, and have tested it and run it through the pre-commit hooks, you're ready to add it to the upstream repo! To do this, push the branch you've been working on back up to _your_ fork

```console
git push -u origin new-feature
```

Now submit a new [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) from your branch to upstream `main`, describing what your new code does and potentially linking to any issues it addresses. This will kick off CI workflows for unit testing and style checks, as well as a review from other amplfi contributors.
