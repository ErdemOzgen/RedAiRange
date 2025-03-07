pipeline {
    agent any
    parameters {
        string(name: 'MODEL_FILE_PATH', defaultValue: '', description: 'Path to the model file')
        string(name: 'MODEL_REF_CARD', defaultValue: '')
        string(name: 'MODEL_SOURCE', defaultValue: '')
        string(name: 'COMMENTS', defaultValue: '')
    }
    environment {
        // Use the environment variable from the Jenkins agent
        MLFLOW_TRACKING_URI = "${env.MLFLOW_TRACKING_URI}" 
    }

    stages {
        stage('AV Scan') {
            steps {
                echo "AV scanning for file at ${params.MODEL_FILE_PATH}"
                sh 'clamscan --infected --remove --recursive /downloads/models/${MODEL_FILE_PATH}'
            }
        }
        stage('Model Scan') {
            steps {
                script {
                    echo "Scanning model ${params.MODEL_FILE_PATH} with ModelScan"
                    sh '. /app/venv/bin/activate && modelscan -p /downloads/models/${MODEL_FILE_PATH}'
                }
            }
        }
        stage('Register Model') {
            steps {
                script {
                    def jobIdentifier = "${env.JOB_NAME}-${env.BUILD_ID}
                    sh '. /app/venv/bin/activate && python /scripts/register_external_model.py --tracking-uri ${MLFLOW_TRACKING_URI} --model-file ${MODEL_FILE_PATH} --model-name ${MODEL_NAME} --run=${jobIdentifier}
                }
            }
        }
    }
}
