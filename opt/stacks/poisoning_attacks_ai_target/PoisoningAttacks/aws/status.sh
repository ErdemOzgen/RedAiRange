#!/bin/bash

# Get the status of a SageMaker notebook instance
notebook_instance_status=$(aws sagemaker describe-notebook-instance --notebook-instance-name secure-ai --query 'NotebookInstanceStatus' --output text)

echo "SageMaker notebook instance 'secure-ai' is currently $notebook_instance_status."
