#!/bin/bash

# Define the directory
SSL_DIR="./ssl"

# Check if directory exists
if [ ! -d "$SSL_DIR" ]; then
  # If not, create the directory
  mkdir "$SSL_DIR"
fi

# Generate the SSL certificate and key
openssl req -x509 -newkey rsa:4096 -keyout "$SSL_DIR/service_key.pem" -out "$SSL_DIR/service_cert.pem" -days 365 -nodes -subj "/C=GB/ST=Greater London/L=London/O=AISolutions/OU=ML/CN=localhost"
