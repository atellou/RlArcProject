"""Package metadata for rlarcworld."""

from setuptools import find_packages, setup

setup(
    name="rlarcworld",
    version="0.1.0",
    packages=find_packages(
        include=[
            "rlarcworld",
            "rlarcworld.*",
            "experiments",
            "experiments.*",
            "aws_sagemaker",
            "aws_sagemaker.*",
        ]
    ),
    install_requires=[
        "transformers>=4.57.3",
        "opencv-python>=4.12.0.88",
        "tensorboard>=2.20.0",
        "tensordict>=0.9.1",
        "torch>=2.8.0",
        "gymnasium>=1.2.0",
        "numpy>=2.2.6",
        "torchrl>=0.9.2",
        "torchvision>=0.23.0",
    ],
    extras_require={
        "aws-sagemaker": [
            "sagemaker>=3.0.0",
            "botocore[crt]>=1.32.23",
            "boto3>=1.42.16",
        ],
    },
    python_requires="==3.12.*",
)
