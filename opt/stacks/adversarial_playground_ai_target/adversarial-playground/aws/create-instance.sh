#create-notebook-instance
#--notebook-instance-name "secure-ai" --instance-type "ml.g4dn.xlarge" --role-arn  --kms-key-id --lifecycle-config-name #-direct-internet-access enabled 
# Get the default VPC
default_vpc_id=$(aws ec2 describe-vpcs --filter "Name=is-default,Values=true" --query "Vpcs[0].VpcId" --output text)
echo "Default VPC ID: $default_vpc_id"
# Get the first Subnet ID in the default VPC
first_subnet_id=$(aws ec2 describe-subnets --filter "Name=vpc-id,Values=$default_vpc_id" --query "Subnets[0].SubnetId" --output text)
echo "First Subnet ID: $first_subnet_id"
# Get the first Security Group associated with the first Subnet
first_security_group_id=$(aws ec2 describe-security-groups --filter "Name=vpc-id,Values=$default_vpc_id" --query "SecurityGroups[0].GroupId" --output text)
echo "First Security Group ID: $first_security_group_id"
# Get the role ARN for the existing role named SageMakerForAdversarialAI
role_arn=$(aws iam get-role --role-name SageMakerForAdversarialAI --query 'Role.Arn' --output text)
echo "using role ARN $role_arn"
# Get the KMS Key ID for the key named SageMakerKey
kms_key_id=$(aws kms describe-key --key-id alias/SageMakerKey --query 'KeyMetadata.KeyId' --output text)
echo "using KMS Key ARN $kms_key_id"
# Create a SageMaker notebook instance with the specified settings
aws sagemaker create-notebook-instance --notebook-instance-name secure-ai \--instance-type ml.g4dn.xlarge \
										--role-arn $role_arn \
										--kms-key-id $kms_key_id \
										--lifecycle-config-name setup-efs-workspace-config \
										--security-group-ids $first_security_group_id \
										--subnet-id $first_subnet_id \
										--direct-internet-access Enabled \
										--volume-size-in-gb 5

# Poll the status of the notebook instance until it's ready
while true; do
    notebook_instance_status=$(aws sagemaker describe-notebook-instance --notebook-instance-name secure-ai --query 'NotebookInstanceStatus' --output text)
    
    if [[ $notebook_instance_status == 'InService' ]]; then
        echo "SageMaker notebook instance 'secure-ai' is ready."
        break
    elif [[ $notebook_instance_status == 'Failed' ]]; then
        echo "SageMaker notebook instance 'secure-ai' failed to be created."
        exit 1
    else
        echo "SageMaker notebook instance 'secure-ai' is still being created. Waiting..."
        sleep 60
    fi
done