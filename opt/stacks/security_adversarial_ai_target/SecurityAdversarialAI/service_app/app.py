# app.py for Flask web application
from flask import Flask, render_template, request, jsonify, redirect, url_for,  send_from_directory
import logging
import requests
import os
from flask import Flask, redirect, url_for, session, request, jsonify
from auth import   create_github_oauth, login_required
import auth

FLASK_API_URL = os.environ.get('FLASK_API_URL', 'http://localhost:5000')

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create GitHub OAuth object
github = create_github_oauth(app)


@app.route('/heartbeat', methods=['GET'])
def heartbeat():
    return jsonify({'system':'web','status': 'OK'}), 200

@app.route('/', methods=['GET', 'POST'])
@login_required
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
            headers = {
                'x-api-key': auth.api_key
            }


            response = requests.post(f"{FLASK_API_URL}/predict", files={'file': image}, headers=headers )
            logger.info(f'Received prediction response: { response.json()}')
            if response.status_code == 200:
                prediction_data = response.json().get('prediction')
                if prediction_data:
                    prediction = prediction_data.get('class_name')
    return render_template('index.html', prediction=prediction,image_url=image_url )

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/login')
def login():
    return github.authorize(callback=url_for('authorized', _external=True))

@app.route('/logout')
def logout():
    session.pop('github_token', None)
    return redirect(url_for('home'))

@app.route('/login/authorized')
def authorized():
    response = github.authorized_response()
    if response is None or response.get('access_token') is None:
        return 'Access denied: reason={} error={}'.format(
            request.args['error'],
            request.args['error_description']
        )

    session['github_token'] = (response['access_token'], '')
    user_info = github.get('user')
    session['user_info'] = user_info.data  # Store user info in session
    
    next_url = request.args.get('next')
    return redirect(next_url or url_for('home'))




if __name__ == '__main__':
    work_dir = os.getcwd();
    print(work_dir)
    app.run(host='0.0.0.0', port=8000)
