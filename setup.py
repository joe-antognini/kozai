"""Setup script for `kozai`."""

from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    author='Joseph O\'Brien Antognini',
    author_email='joe.antognini@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    description='Evolve hierarchical triples.',
    install_requires=['numpy>=1.18.4', 'scipy>=1.4.1'],
    keywords='kozai lidov triple dynamics orbit star',
    license='MIT',
    long_description=readme(),
    name='kozai',
    packages=['kozai'],
    python_requires='>=3.6',
    scripts=[
        'scripts/kozai',
        'scripts/kozai-test-particle',
        'scripts/kozai-ekm',
    ],
    tests_require=['pytest>=5.4.2'],
    url='https://github.com/joe-antognini/kozai',
    version='0.3.0',
    zip_safe=False,
)
