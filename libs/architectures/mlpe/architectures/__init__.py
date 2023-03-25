import inspect

from mlpe.architectures.embeddings import Flattener, NChannelDenseEmbedding
from mlpe.architectures.flows import CouplingFlow, MaskedAutoRegressiveFlow


# This is a decorator that takes a flow and returns a function that
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


def _wrap_embedding(arch):
    def func(*args, **kwargs):
        def f(n_ifos):
            return arch(n_ifos, *args, **kwargs)

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
}
