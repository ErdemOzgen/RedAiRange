#!/bin/bash

set -e

sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport $file_system_id.efs.eu-west-2.amazonaws.com:/ /home/ec2-user/SageMaker/efs
sudo chmod go+rw /home/ec2-user/SageMaker/efs
sudo chown ec2-user:ec2-user /home/ec2-user/SageMaker/efs
SSH_KEY=/home/ec2-user/SageMaker/efs/.ssh/github 
if [[ ! -f $SSH_KEY  ]] 
then
	mkdir -p /home/ec2-user/SageMaker/efs/.ssh
	sudo chown ec2-user:ec2-user /home/ec2-user/SageMaker/efs/.ssh
	aws ssm get-parameter --name "/secure-ai/github/ssh-key" --with-decryption --query "Parameter.Value" --output text > /home/ec2-user/SageMaker/efs/.ssh/github
	chmod 600 /home/ec2-user/SageMaker/efs/.ssh/github
fi
VENV_DIR=/home/ec2-user/SageMaker/efs/secure-ai/.venv
if [[ -d $VENV_DIR  ]]

then
    source $VENV_DIR/bin/activate
	CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
	export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH
    python -m ipykernel install --name=secure-ai --display-name="Secure AI"
    deactivate
fi