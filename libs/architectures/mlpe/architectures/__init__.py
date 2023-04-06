import inspect

from mlpe.architectures.embeddings import (
    CoherentDenseEmbedding,
    Flattener,
    NChannelDenseEmbedding,
    ResNet,
)
from mlpe.architectures.flows import CouplingFlow, MaskedAutoRegressiveFlow


# see the comment in the `embeddings` dict below for an explanation of this
def _wrap_flow(arch):
    def func(*args, **kwargs):
        def f(shape, embedding):
            return arch(shape, embedding, *args, **kwargs)

        return f

    params = inspect.signature(arch).parameters
    params = list(params.values())[2:]
    func.__signature__ = inspect.Signature(params)
    return func


# this is a decorator that takes an embedding
# and returns a function `func` whose
# arguments are the same arguments as the embedding,
# but with the first argument
# (the "shape", i.e. number of ifos and the parameter dimension) removed.

# This allows us to specify the embedding
# as a callable in a function signature,
# and expose the embedding's arguments at the command line.

# When typeo parses the config file, it will call the function `func` with the
# arguments specified in the config file. `func` returns another function `f`,
# which has the embeddings arguments passed to it.

# This function `f` will then be called with the shape as the first argument
# and will instantiate the embedding.
# All of embeddings arguments are automatically passed to `f`.


def _wrap_embedding(embedding):
    def func(*args, **kwargs):
        def f(shape):
            return embedding(shape, *args, **kwargs)

        return f

    params = inspect.signature(embedding).parameters
    params = list(params.values())[1:]
    func.__signature__ = inspect.Signature(params)
    return func


flows = {
    "coupling": _wrap_flow(CouplingFlow),
    "maf": _wrap_flow(MaskedAutoRegressiveFlow),
}
embeddings = {
    "flattener": _wrap_embedding(Flattener),
    "dense": _wrap_embedding(NChannelDenseEmbedding),
    "coherent": _wrap_embedding(CoherentDenseEmbedding),
    "resnet": _wrap_embedding(ResNet),
}
