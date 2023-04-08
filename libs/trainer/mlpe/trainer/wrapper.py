import inspect


def _wrap_callable(callable):
    """
    Wrap callable so that all but its first arguments
    can be parsed from the command line via typeo.
    """

    def func(*args, **kwargs):
        def f(variable):
            return callable(variable, *args, **kwargs)

        return f

    params = inspect.signature(callable).parameters
    params = list(params.values())[1:]
    func.__signature__ = inspect.Signature(params)
    return func
