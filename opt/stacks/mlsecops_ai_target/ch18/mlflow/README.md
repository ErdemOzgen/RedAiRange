

docker build -t mlflow-server:latest .

docker images

docker run -p 5000:5000 mlflow-server:latest


docker run -p 8080:8080 -p 50000:50000 -p 5000:5000 -v jenkins_home:/var/jenkins_home jenkins-mlsecops v $(pwd)/models:/downloads/models