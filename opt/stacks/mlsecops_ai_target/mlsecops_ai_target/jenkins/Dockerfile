FROM jenkins/jenkins:lts

USER root

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    python3-venv \
    wget \
    apt-transport-https \
    gnupg \
    lsb-release \
    clamav \
    clamav-daemon


# Freshen ClamAV's database
RUN freshclam

# Create a virtual environment
RUN python3 -m venv /app/venv
# Activate the virtual environment and install your Python packages
RUN . /app/venv/bin/activate && pip install modelscan mlflow adversarial-robustness-toolbox bandit

# Install Trivy
RUN apt-get install -y wget apt-transport-https gnupg lsb-release
RUN wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add -
RUN echo deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main | tee -a /etc/apt/sources.list.d/trivy.list
RUN apt-get update && apt-get install -y trivy

# Install Jenkins plugins
COPY plugins.txt /usr/share/jenkins/plugins.txt
#RUN /usr/local/bin/install-plugins.sh < /usr/share/jenkins/plugins.txt


# Setting up the environment for Python and security tools
RUN python3 -m venv /app/venv \
    && . /app/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install modelscan h5py tensorflow torch dill joblib mlflow adversarial-robustness-toolbox bandit

# Copy the job creation Groovy script and Jenkinsfile
#COPY create_pipeline_job.groovy /usr/share/jenkins/ref/init.groovy.d/create_pipeline_job.groovy
COPY pipelines/validate-model-pipeline.groovy /usr/share/jenkins/ref/jobs/ModelPipelineJob/validate-model-pipeline.groovy

# Ensure Jenkins owns all copied files
RUN chown -R jenkins:jenkins /usr/share/jenkins/ref

# Copy your Python scripts into the image
COPY scripts /app/scripts

# Ensure the script directory and its contents are owned by the correct user
RUN chown -R jenkins:jenkins /app/scripts

# Ensure the script directory and its contents are owned by the correct user
RUN chown -R jenkins:jenkins /app/scripts

USER jenkins
