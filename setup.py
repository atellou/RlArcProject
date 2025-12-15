"""Package metadata for rlarcworld."""

from setuptools import find_packages, setup

setup(
    name="rlarcworld",
    version="0.1.0",
    packages=find_packages(
        include=["rlarcworld", "rlarcworld.*", "experiments", "experiments.*"]
    ),
    install_requires=[
        "torch",
        "gymnasium",
        "numpy",
        "torchrl",
        "tensorboard",
        "standard-imghdr",
        "torchvision",
        "pyyaml",
        "python-json-logger",
    ],
    python_requires="==3.10.*",
)
