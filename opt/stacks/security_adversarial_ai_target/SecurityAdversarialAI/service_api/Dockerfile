# service_api/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY . /app
# Set the FLASK_APP environment variable to inference_service.py
ENV FLASK_APP=inference_service.py
CMD ["flask", "run", "--host=0.0.0.0"]
