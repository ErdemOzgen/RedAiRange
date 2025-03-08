#FROM jupyter/minimal-notebook:python-3.9

# TODO: we need tochange this to the correct image  
FROM rar  

# Install system dependencies required for Pillow and other packages
USER root


# Switch back to notebook user for remaining operations
USER ${NB_UID}


# Copy application files
COPY --chown=${NB_UID}:${NB_USER} . /home/jovyan/work

# Set healthcheck to verify Jupyter is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1

# The jupyter/minimal-notebook image already exposes port 8888 and has a CMD
# to start Jupyter, so we don't need to specify those again