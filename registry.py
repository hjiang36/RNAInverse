"""
This file defines a registry structure so that we can reference certain model or loss
function using strings in the config files.
"""


def _register(module_dict, module_name, module):
    """
    Register module to the target dictionary

    :param module_dict: module dictionary
    :param module_name: the name of the module to be registered
    :param module: the module class to be registered
    :return: None
    """
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    """
    This class provides some helper functions for module registration.
    It extends the standard dictionary class and provides a register function.

    To create a registry:
        example_registry = Registry({"default": default_module})

    To register functions / class to the registry, use decorator:
        @example_registry.register("foo_module", foo)
        def foo():
            ...

    Access the module is same as a dictionary:
        f = example_registry["foo_module"]
    """

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)
        if "registry_name" in kwargs:
            self.registry_name = kwargs["registry_name"]

    def register(self, module_name, module=None):
        print("Registering: " + module_name)
        if module is not None:
            _register(self, module_name, module)
            return

        def register_fn(fn):
            _register(self, module_name, fn)
            return fn

        return register_fn


# Declare a few global registry
DATA_SETS = Registry()
LOSSES = Registry()
MODELS = Registry({"registry_name": "Models"})
