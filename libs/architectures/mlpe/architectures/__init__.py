import inspect

from mlpe.architectures.embeddings import (
    CoherentDenseEmbedding,
    Flattener,
    NChannelDenseEmbedding,
    ResNet,
)
from mlpe.architectures.flows import CouplingFlow, MaskedAutoRegressiveFlow


def _wrap_flow(arch):
    def func(*args, **kwargs):
        def f(
            shape,
            embedding,
            preprocessor,
            opt,
            sched,
            inference_params,
            priors,
        ):
            return arch(
                shape,
                embedding,
                preprocessor,
                opt,
                sched,
                inference_params,
                priors,
                *args,
                **kwargs
            )

        return f

    params = inspect.signature(arch).parameters
    params = list(params.values())[7:]
    func.__signature__ = inspect.Signature(params)
    return func


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
