#! /usr/bin/env python

import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
  user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

  def initialize_options(self):
    TestCommand.initialize_options(self)
    self.pytest_args = []

  def finalize_options(self):
    TestCommand.finalize_options(self)
    self.test_args = []
    self.test_suite = True

  def run_tests(self):
    import pytest
    errno = pytest.main(self.pytest_args)
    sys.exit(errno)

def readme():
  with open('README.md') as f:
    return f.read()

setup(name='kozai',
      version='0.2.1',
      description='Evolve hierarchical triples',
      long_description=readme(),
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics'],
      keywords='kozai lidov triple dynamics orbit star',
      url='https://github.com/joe-antognini/kozai',
      author='Joseph O\'Brien Antognini',
      author_email='joe.antognini@gmail.com',
      license='MIT',
      packages=['kozai'],
      scripts=[
        'scripts/kozai', 
        'scripts/kozai-test-particle', 
        'scripts/kozai-ekm'],
      install_requires=['numpy', 'scipy'],
      tests_require=['pytest'],
      cmdclass = {'test': PyTest},
      zip_safe=False)
