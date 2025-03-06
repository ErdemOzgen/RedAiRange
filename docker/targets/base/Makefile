# Makefile for building and pushing multi-platform Docker images

# Variables (override these with command-line parameters if desired)
CONTAINER_NAME ?= rar-target-base
USERNAME       ?= erdemozgen
TAG            ?= latest
# Target platforms for buildx; adjust as needed
PLATFORMS      ?= linux/amd64,linux/arm64

# Full image name for Docker Hub
IMAGE          := $(USERNAME)/$(CONTAINER_NAME):$(TAG)

.PHONY: all build buildx-create multiarch

# Default target: runs all steps in logical order
all: build multiarch

# Build the image locally (single platform)
build:
	docker build -t $(CONTAINER_NAME) .

# Create and bootstrap a new buildx builder if not already set up
buildx-create:
	@echo "Creating and bootstrapping buildx builder..."
	-docker buildx create --name multiarch-builder --use
	docker buildx inspect --bootstrap

# Build and push multi-platform image to Docker Hub
multiarch: buildx-create
	docker buildx build --platform $(PLATFORMS) -t $(IMAGE) --push .
