#!/bin/bash

notebook_instance_status=$(aws sagemaker describe-notebook-instance --notebook-instance-name secure-ai --query 'NotebookInstanceStatus' --output text)
if [[ $notebook_instance_status == 'InService' ]]; then
        echo "SageMaker notebook instance 'secure-ai' is already in service and ready to be used."
        exit
fi

# Start a SageMaker notebook instance
aws sagemaker start-notebook-instance --notebook-instance-name secure-ai

echo "SageMaker notebook instance 'secure-ai' is being started."

# Poll the status of the notebook instance until it's in service
while true; do
    notebook_instance_status=$(aws sagemaker describe-notebook-instance --notebook-instance-name secure-ai --query 'NotebookInstanceStatus' --output text)    
    if [[ $notebook_instance_status == 'InService' ]]; then
        echo "SageMaker notebook instance 'secure-ai' has been started and is ready."
        break
    else
        echo "SageMaker notebook instance 'secure-ai' is still being started. Waiting..."
        sleep 60
    fi
done
