# Use the latest pypiserver/pypiserver as the base image
FROM pypiserver/pypiserver:latest

# Create the packages directory and set the necessary permissions 
# for the already created pypiserver user and group
RUN mkdir -p /data/packages && \
    chown -R pypiserver:pypiserver /data/packages

# Switch to pypiserveruser
USER pypiserver

# Set the working directory
WORKDIR /data
