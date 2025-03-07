# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye
# Create a new user "mlflow"
RUN useradd -m mlflow
# Set the working directory in the container
# Add /home/mlflow/.local/bin to PATH
ENV PATH="/home/mlflow/.local/bin:${PATH}"
WORKDIR /home/mlflow
# Change to the new user
USER mlflow
# Upgrade pip
RUN pip install --upgrade pip --user
# Install MLflow
RUN pip install --no-cache mlflow --user
# Expose the port the app runs on
EXPOSE 5000
# Define environment variable
ENV MLFLOW_HOME /home/mlflow
# Run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0"]