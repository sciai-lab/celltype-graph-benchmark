import inspect


def _constructor(self, **kwargs):
    self._kwargs = kwargs


def _caller(self, x):
    return self._fun(x)


def _parse_signature(fun):
    # names of all args and kwargs
    run_parameters = dict(inspect.signature(fun).parameters)
    default_args = {key: value.default for key, value in list(run_parameters.items())
                    if value.default is not inspect.Parameter.empty}

    required_args = [key for key, value in list(run_parameters.items())
                     if value.default is inspect.Parameter.empty]
    return required_args, default_args


def class_from_func(func):
    required_args, default_args = _parse_signature(func)
    new_doc = f'Dynamically created class from {func.__name__}\n' \
              f'init name signature: {default_args},\n' \
              f'call signature: {required_args},\n' \
              f'Original docstring:\n' \
              f'{func.__doc__}'

    produced_class = type(func.__doc__, (object,), {
        "_fun": staticmethod(func),
        "__init__": _constructor,
        "__call__": _caller,
        "__signature__": inspect.signature(func),
        "__doc__": new_doc,
    })
    return produced_class
