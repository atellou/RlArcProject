# syntax=docker/dockerfile:1-labs
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest

RUN apt-get update && apt-get install -y curl apt-utils
# Set up working directory
WORKDIR /app

# Install Poetry
ENV POETRY_VIRTUALENVS_CREATE=false
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY --exclude=*.ipynb --exclude=**/dataset . /app/

# Install dependencies
RUN poetry install
RUN poetry build

# Expose port 6006 for TensorBoard
EXPOSE 6006

# Set entrypoint (change train.py to your script)
ENTRYPOINT ["poetry", "run", "python", "rlarcworld/jobs/train.py"]

