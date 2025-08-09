# syntax=docker/dockerfile:1-labs
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest

# Distroless uv
COPY --from=ghcr.io/astral-sh/uv:0.8.6 /uv /uvx /bin/

RUN apt-get update && apt-get install -y curl apt-utils
# Set up working directory
WORKDIR /app

# Copy project files
COPY --exclude=*.ipynb --exclude=**/dataset \ 
./pyproject.toml \ 
./uv.lock \ 
./rlarcworld /app/

# Install dependencies
RUN uv sync --locked
ENV PATH="/app/.venv/bin:$PATH"

# Expose port 6006 for TensorBoard
EXPOSE 6006

# Set entrypoint
ENTRYPOINT ["uv", "run", "python", "rlarcworld/jobs/train.py"]

