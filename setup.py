import os
import re
from setuptools import setup, find_packages

DESCRIPTION = """Manifold aware pytorch layers"""
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(PROJECT_ROOT, "README.rst"), encoding="utf-8") as buff:
    LONG_DESCRIPTION = buff.read()


def get_version(*path):
    version_file = os.path.join(*path)
    lines = open(version_file, "rt").readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError("Unable to find version in %s." % (version_file,))


classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "License :: OSI Approved :: Apache Software License",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Operating System :: OS Independent",
]

if __name__ == "__main__":
    setup(
        name="geoopt-layers",
        author="Maxim Kochurov",
        description=DESCRIPTION,
        maintainer_email="maxim.v.kochurov@gmail.com",
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=["geoopt"],
        version=get_version(PROJECT_ROOT, "geoopt_layers", "__init__.py"),
        url="https://github.com/ferrine/geoopt-layers",
        python_requires=">=3.6.0",
        license="Apache License, Version 2.0",
        classifiers=classifiers,
    )
