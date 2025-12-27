# Custom SageMaker training image for RlArcProject
# Base on the official PyTorch 2.1 GPU container that SageMaker uses.

# syntax=docker/dockerfile:1.4
ARG BASE_IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-sagemaker
FROM ${BASE_IMAGE}

# Install uv (fast dependency resolver) and upgrade pip just in case.
RUN python -m pip install --no-cache-dir --upgrade pip uv

# ---------------------------------------------------------------------------
# Install project dependencies declared in pyproject.toml using **uv**.
# Copy only metadata first to leverage Docker layer caching.
# ---------------------------------------------------------------------------
COPY pyproject.toml /opt/build/
RUN uv pip install --system --no-cache-dir /opt/build

# ---------------------------------------------------------------------------
# Copy the project source tree into the container image.
# We place it under /opt/ml/code so SageMaker picks it up automatically.
# ---------------------------------------------------------------------------
# Copy full repository (only what is needed) and then install editable project
COPY . /opt/ml/code

# Second-layer install to catch any extras / local-path deps
RUN uv pip install --system --no-cache-dir /opt/ml/code

# Make sure the project root is discoverable for Python imports
ARG PYTHONPATH
ENV PYTHONPATH=/opt/ml/code:$PYTHONPATH

# Workdir where SageMaker will invoke the entry-point
WORKDIR /opt/ml/code