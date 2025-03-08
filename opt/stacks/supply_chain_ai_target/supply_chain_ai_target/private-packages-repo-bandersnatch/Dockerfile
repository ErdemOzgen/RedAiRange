# Dockerfile

# Start from the original Bandersnatch image
FROM pypa/bandersnatch

# Install dependencies for Trivy
RUN apt-get update && \
    apt-get install -y wget apt-transport-https gnupg lsb-release

# Add Trivy repository
RUN wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add - && \
    echo deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -cs) main | tee -a /etc/apt/sources.list.d/trivy.list && \
    apt-get update

# Install Trivy
RUN apt-get install -y trivy

# Copy the custom blocklist plugin
COPY trivy_blocklist.py /usr/src/app/src/bandersnatch_filter_plugins/

# Copy the configuration file
COPY config.cfg /usr/src/app/
