#!/bin/bash
key_alias=$1
# Your AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)

# Check if a key with the alias SageMakerKey exists
key_id=$(aws kms list-aliases --query "Aliases[?AliasName=='alias/$key_alias'].TargetKeyId" --output text)

# If the key does not exist, create it
if [[ -z "$key_id" ]]; then
    echo "Key with alias '$key_alias' does not exist. Creating one now..."
    
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
