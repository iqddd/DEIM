"""
Tiny compatibility shim for vendored modules that still use the original @register decorator.
"""


def register(*args, **kwargs):
    def decorator(obj):
        return obj

    return decorator

