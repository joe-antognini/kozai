#! /usr/bin/env python

from ..triplesec import Triple
from ..ts_vector import Triple_vector

###
### Object creation tests
###

def test_make_Triple():
  '''Try to create a Triple class.'''
  t = Triple()

def test_nooct():
  '''Try to turn the octupole term off.'''
  t = Triple(octupole=False)

# todo test cpu timeout

# todo integrate triple

# todo test conservation of constants
