FROM jupyter/minimal-notebook:python-3.9

# Set environment variables to reduce Python buffering and prevent bytecode generation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies required for Pillow and other packages
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to notebook user for remaining operations
USER ${NB_UID}

# Install Python packages with version pinning
# Split into multiple layers to leverage Docker caching

# Install data science packages
RUN pip install --no-cache-dir \
    numpy==1.24.* \
    pandas==2.0.* \
    matplotlib==3.7.*

# Install Pillow and TensorFlow
RUN pip install --no-cache-dir \
    Pillow==9.2.0 \
    tensorflow==2.15.*

# Install Flask and scikit-learn
RUN pip install --no-cache-dir \
    flask==2.3.* \
    scikit-learn

# Copy application files
COPY --chown=${NB_UID}:${NB_USER} . /home/jovyan/work

# Set healthcheck to verify Jupyter is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1

# The jupyter/minimal-notebook image already exposes port 8888 and has a CMD
# to start Jupyter, so we don't need to specify those again