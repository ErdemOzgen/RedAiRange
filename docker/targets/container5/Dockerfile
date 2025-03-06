
FROM jupyter/minimal-notebook:python-3.10

# Set environment variables to reduce Python buffering and prevent bytecode generation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies required for Pillow and other packages, including pkg-config and libsentencepiece-dev
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    pkg-config \
    libsentencepiece-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to notebook user for remaining operations
USER ${NB_UID}

# Copy the aggregated requirements file into the container
COPY --chown=${NB_UID}:${NB_USER} requirements.txt /tmp/requirements.txt

# Install Python packages from the aggregated requirements file
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application files
COPY --chown=${NB_UID}:${NB_USER} . /home/jovyan/work

# Set healthcheck to verify Jupyter is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1
