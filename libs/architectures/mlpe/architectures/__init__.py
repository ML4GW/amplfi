import inspect

from mlpe.architectures.embeddings import (
    CoherentDenseEmbedding,
    Flattener,
    NChannelDenseEmbedding,
    ResNet,
)
from mlpe.architectures.flows import CouplingFlow, MaskedAutoRegressiveFlow


# this is a decorator that takes a flow and returns a function that
# takes the same arguments as the flow, but with the first
# two arguments (the shape and embedding) removed.
# This is used to wrap the architectures in
# this file so that they can be used as a callable in the config file.
# This callable willthen be called with the shape and embedding
# as the first two arguments to instantiate the flow
def _wrap_flow(arch):
    def func(*args, **kwargs):
        def f(shape, embedding):
            return arch(shape, embedding, *args, **kwargs)

        return f

    params = inspect.signature(arch).parameters
    params = list(params.values())[2:]
    func.__signature__ = inspect.Signature(params)
    return func


# this is a decorator that takes an embedding and returns a function that
# takes the same arguments as the embedding, but with the first two
# arguments (the number of ifos and the parameter dimension) removed.
# This is used to wrap the architectures in
# this file so that they can be used as a callable in the config file.
# This callable will then be called with the number of ifos and the
# parameter dimension as the first two arguments to instantiate the
# embedding
def _wrap_embedding(arch):
    def func(*args, **kwargs):
        def f(shape):
            return arch(shape, *args, **kwargs)

        return f

    params = inspect.signature(arch).parameters
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
