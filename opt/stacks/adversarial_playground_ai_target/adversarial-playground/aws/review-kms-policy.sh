#!/bin/bash

# Your AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)


# Get the ARN of the role SageMakerForAdversarialAI
role_arn=$(aws iam get-role --role-name SageMakerForAdversarialAI --query 'Role.Arn' --output text)

# Create a key policy that enables IAM access and grants encryption and decryption permissions to the role
cat key_policy_template.json
# Read policy from key_policy_template.json file
policy=$(sed -e "s#\$AWS_ACCOUNT_ID#$AWS_ACCOUNT_ID#g" -e "s#\$role_arn#$role_arn#g" key_policy_template.json)


echo $policy