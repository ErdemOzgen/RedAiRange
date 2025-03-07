#!/bin/bash
# Setup the AWS Environment so that we can use SageMaker Notebooks for our adversarial AI playground
# This includes setting up 
# A limited customer-managed S3 access IAM Policy 
# An IAM Role for Sagemaker, configured with the right trust policy, and assigned with the right policies
# Create or use an existing Customer Managed Key (CMK) KMS Key with SageMakerKey.
# Create a custom lifecycle configuration which sets up 

RoleName="SageMakerForAdversarialAI"
S3PolicyARN=$(aws iam create-policy --policy-name "LimitedS3PermissionsForSageMaker" --policy-document file://assets/limited_s3_permissions_policy.json --query Policy.Arn --output text)
key_alias=SageMakerKey
aws iam create-role --role-name $RoleName --assume-role-policy-document file://assets/sagemaker-trust-policy.json
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess --role-name $RoleName
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess --role-name $RoleName
aws iam attach-role-policy --policy-arn $S3PolicyARN --role-name $RoleName
aws iam attach-role-policy --policy-arn $S3PolicyARN --role-name $RoleName

# Get AWS account ID
# Check if a key with the alias SageMakerKey exists
key_id=$(aws kms list-aliases --query "Aliases[?AliasName=='alias/$key_alias'].TargetKeyId" --output text)

# If the key does not exist, create it
if [[ -z "$key_id" ]]; then
	AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)

    echo "Key with alias '$key_alias' does not exist. Creating a new KMS Key now..."
    
    # Create a KMS key
    key_id=$(aws kms create-key --description "$key_alias" --query 'KeyMetadata.KeyId' --output text)

    # Create an alias for the KMS key
    aws kms create-alias --alias-name alias/$key_alias --target-key-id $key_id

    # Get the ARN of the role SageMakerForAdversarialAI
    role_arn=$(aws iam get-role --role-name SageMakerForAdversarialAI --query 'Role.Arn' --output text)

    # Create a key policy that enables IAM access and grants encryption and decryption permissions to the role
    # Read policy from key_policy_template.json file
   policy=$(sed -e "s#\$AWS_ACCOUNT_ID#$AWS_ACCOUNT_ID#g" -e "s#\$role_arn#$role_arn#g" assets/key_policy_template.json)
   echo $policy

    # Apply the policy to the KMS key
    aws kms put-key-policy --key-id $key_id --policy-name default --policy "$policy"

else
    echo "Key with alias '$key_alias' already exists."
fi


# EFS file system name
efs_name="adversarialAI"

# Check if an EFS file system with the given name exists
file_system_id=$(aws efs describe-file-systems --query "FileSystems[?Name=='$efs_name'].FileSystemId" --output text)

# If the file system does not exist, create it
if [[ -z "$file_system_id" ]]; then
    echo "EFS file system with name '$efs_name' does not exist. Creating one now..."
    
    # Create an EFS file system
        file_system_id=$(aws efs create-file-system --creation-token "$efs_name" --performance-mode generalPurpose --throughput-mode bursting --encrypted --kms-key-id $key_id --tags Key=Name,Value="$efs_name" --query 'FileSystemId' --output text)

    echo "Created new EFS file system with id '$file_system_id'."

else
    echo "EFS file system with name '$efs_name' already exists. Its id is '$file_system_id'."
fi

# Return the file system id
echo "EFS file system id is $file_system_id"

# lifecycle config script
script_file="assets/start-notebook-config.sh"
# Read the script file into a variable
script=$(cat $script_file)

# Perform the substitution in memory
modified_script=$(echo "$script" | sed "s|\$file_system_id|$file_system_id|g")


CREATE_NOTEBOOK_LCC_CONTENT=`openssl base64 -A -in assets/create-notebook-config.sh`
START_NOTEBOOK_LCC_CONTENT=$(echo -n "$modified_script" | openssl base64 -A)
## create lifecycle configurations
aws sagemaker create-notebook-instance-lifecycle-config --notebook-instance-lifecycle-config-name setup-efs-workspace-config  --on-create Content=$CREATE_NOTEBOOK_LCC_CONTENT --on-start Content=$START_NOTEBOOK_LCC_CONTENT 


