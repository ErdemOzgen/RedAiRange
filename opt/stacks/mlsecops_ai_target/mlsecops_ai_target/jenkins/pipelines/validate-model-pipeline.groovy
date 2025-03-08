pipeline {
    agent any
    parameters {
        string(name: 'MODEL_FILE_PATH', defaultValue: '', description: 'Path to the model file')
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
                    echo "Scanning model  ${params.MODEL_FILE_PATH} with ModelScan"
                    sh '. /app/venv/bin/activate && /app/scripts/modelscan.sh -p /downloads/models/${MODEL_FILE_PATH}'

                }
            }
        }
        stage('Adversarial Tests') {
            steps {     
                script {           
                    sh '. /app/venv/bin/activate && python /app/scripts/adversarial_tests.py --model-file ${params.MODEL_FILE_PATH}'
            }
         }
        }
       stage('Register Model') {
            steps {
                script {
                sh '. /app/venv/bin/activate && python /app/scripts/register_model.py --tracking-uri ${MLFLOW_TRACKING_URI} --model-file ${params.MODEL_FILE_PATH}'
        }
    }
   }
    }
    post {
        failure {
            script {
                // Use the environment variable for the MLflow server URI in case of failure
                sh """. /app/venv/bin/activate && python -c "import mlflow; mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}'); mlflow.create_experiment('YourExperiment'); with mlflow.start_run() as run: mlflow.set_tag('mlflow.runName', 'Quarantined'); mlflow.log_param('model_file', '${params.MODEL_FILE_PATH}')" """
            }
        }
    }
}
