# app.py for Flask web application
from flask import Flask, render_template, request, jsonify, redirect, url_for,  send_from_directory
import logging
import requests
import os

from flask import Flask, redirect, url_for, session, request, jsonify
from flask_oauthlib.client import OAuth
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


# Load environment variables from .env file
load_dotenv()

##setup OAuth Login
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
app.config['GITHUB_CLIENT_ID'] = os.getenv('GITHUB_CLIENT_ID')
app.config['GITHUB_CLIENT_SECRET'] = os.getenv('GITHUB_CLIENT_SECRET')

oauth = OAuth(app)
github = oauth.remote_app(
    'github',
    consumer_key=app.config['GITHUB_CLIENT_ID'],
    consumer_secret=app.config['GITHUB_CLIENT_SECRET'],
    request_token_params={'scope': 'user:email'},
    base_url='https://api.github.com/',
    request_token_url=None,
    access_token_method='POST',
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize'

FLASK_API_URL = os.environ.get('FLASK_API_URL', 'http://localhost:5000')

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/heartbeat', methods=['GET'])
def heartbeat():
    return jsonify({'system':'web','status': 'OK'}), 200

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    image_url =  None
    if request.method == 'POST':
        # Get the image from the form and send it to the Flask API
        image = request.files.get('image')
        if image:
            logger.info(f"Filename: {image.filename}")
            logger.info(f"Content-Type: {image.content_type}")
            logger.info(f"File size: {len(image.read())} bytes")    
            image.seek(0)
            image.save(os.path.join(UPLOAD_FOLDER, image.filename))
            image.seek(0)
            image_url = url_for('uploaded_file', filename=image.filename)
            logger.info(f"file uploaded and accessible as {image_url}")
            logger.info(f"Sending prediction request to {FLASK_API_URL}")
            response = requests.post(f"{FLASK_API_URL}/predict", files={'file': image})
            logger.info(f'Received prediction response: { response.json()}')
            if response.status_code == 200:
                prediction_data = response.json().get('prediction')
                if prediction_data:
                    prediction = prediction_data.get('class_name')
    return render_template('index.html', prediction=prediction,image_url=image_url )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    work_dir = os.getcwd();
    print(work_dir)
    app.run(host='0.0.0.0', port=8000)
