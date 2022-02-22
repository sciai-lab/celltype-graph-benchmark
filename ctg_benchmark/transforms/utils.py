import inspect


def _constructor(self, **kwargs):
    """wrap the default parameters"""
    self._kwargs = kwargs


def _caller(self, x):
    """wrap the function call"""
    return self._fun(x, **self._kwargs)


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

    produced_class = type(func.__name__, (object,), {
        "_fun": staticmethod(func),
        "__init__": _constructor,
        "__call__": _caller,
        "__signature__": inspect.signature(func),
        "__doc__": new_doc,
    })
    return produced_class


def _to_camel_case(name):
    new_name = []
    i = 0
    while i < len(name):
        _x = name[i]
        if i == 0:
            new_name.append(_x.capitalize())
        elif _x == '_':
            new_name.append(name[i+1].capitalize())
            i += 1
        else:
            new_name.append(_x)
        i += 1
    return ''.join(new_name)


def to_camel_case(name):
    name = name[0].capitalize() + name[1:]
    while name.find('_') != -1:
        idx = name.find('_')
        name = name[:idx] + name[idx + 1].capitalize() + name[idx + 2:]
    return name
