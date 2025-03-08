#!/bin/bash

    notebook_instance_status=$(aws sagemaker describe-notebook-instance --notebook-instance-name secure-ai --query 'NotebookInstanceStatus' --output text)
if [[ $notebook_instance_status != 'InService' ]]; then
    echo "SageMaker notebook instance 'secure-ai' is  already stopped."
    exit
fi


# Stop a SageMaker notebook instance
aws sagemaker stop-notebook-instance --notebook-instance-name secure-ai

echo "SageMaker notebook instance 'secure-ai' is being stopped."

# Poll the status of the notebook instance until it's stopped
while true; do
    notebook_instance_status=$(aws sagemaker describe-notebook-instance --notebook-instance-name secure-ai --query 'NotebookInstanceStatus' --output text)
    
    if [[ $notebook_instance_status == 'Stopped' ]]; then
        echo "SageMaker notebook instance 'secure-ai' has been stopped."
        break
    else
        echo "SageMaker notebook instance 'secure-ai' is still being stopped. Waiting..."
        sleep 60
    fi
done
