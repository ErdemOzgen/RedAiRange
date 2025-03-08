#!/bin/bash

# Delete a stopped SageMaker notebook instance
aws sagemaker delete-notebook-instance --notebook-instance-name secure-ai
if [[ $? -ne 0 ]]; then
    echo "Failed to delete the SageMaker notebook instance 'secure-ai'. please check status and make sure it's either stopped or failed to be able to delete it"
    exit 1
fi
echo "SageMaker notebook instance 'secure-ai' is being deleted."
# Poll the status of the notebook instance until it's deleted
while true; do
    notebook_instance_status=$(aws sagemaker describe-notebook-instance --notebook-instance-name secure-ai --query 'NotebookInstanceStatus' --output text 2>/dev/null)
    
    if [[ $? -ne 0 ]]; then
        echo "SageMaker notebook instance 'secure-ai' has been deleted."
        break
    else
        echo "SageMaker notebook instance 'secure-ai' is still being deleted. Waiting..."
        sleep 60
    fi
done
