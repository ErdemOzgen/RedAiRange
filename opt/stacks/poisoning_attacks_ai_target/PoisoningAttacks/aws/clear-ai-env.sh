#!/bin/bash
RoleName=SageMakerForAdversarialAI
Account=$(aws sts get-caller-identity --query Account --output text)
S3PolicyARN=arn:aws:iam::$Account:policy/LimitedS3PermissionsForSageMaker
aws iam detach-role-policy --role-name $RoleName --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam detach-role-policy --role-name $RoleName --policy-arn arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess
aws iam detach-role-policy --role-name $RoleName --policy-arn $S3PolicyARN
aws iam delete-role --role-name $RoleName 
aws iam delete-policy --policy-arn $S3PolicyARN
aws sagemaker delete-notebook-instance-lifecycle-config --notebook-instance-lifecycle-config-name  setup-efs-workspace-config