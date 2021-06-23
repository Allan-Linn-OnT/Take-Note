import numpy as np
import torch

def deepsubclasses(klass):
  """Iterate over direct and indirect subclasses of `klass`."""
  for subklass in klass.__subclasses__():
    yield subklass
    for subsubklass in deepsubclasses(subklass):
      yield subsubklass

class Factory(object):
  """Factory mixin.
  Provides a `make` method that searches for an appropriate subclass to
  instantiate given a key. Subclasses inheriting from a class that has Factory
  mixed in can expose themselves for instantiation through this method by
  setting the class attribute named `key` to an appropriate value.
  """

  @classmethod
  def make(cls, key, *args, **kwargs):
    """Instantiate a subclass of `cls`.
    Args:
      key: the key identifying the subclass.
      *args: passed on to the subclass constructor.
      **kwargs: passed on to the subclass constructor.
    Returns:
      An instantiation of the subclass that has the given key.
    Raises:
      KeyError: if key is not a child subclass of cls.
    """
    for subklass in deepsubclasses(cls):
      if subklass.key == key:
        return subklass(*args, **kwargs)

    raise KeyError("unknown %s subclass key %s" % (cls, key))

def identity(x):
  return x