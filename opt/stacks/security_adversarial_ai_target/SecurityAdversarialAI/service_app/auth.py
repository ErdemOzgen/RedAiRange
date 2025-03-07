from flask import redirect, url_for, session, request
from functools import wraps
from flask_oauthlib.client import OAuth
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv('API_KEY')

# Function to create OAuth object and GitHub remote app
def create_github_oauth(app):
    oauth = OAuth(app)
    app.secret_key = os.getenv('SECRET_KEY')
    github = oauth.remote_app(
        'github',
        consumer_key=os.getenv('GITHUB_CLIENT_ID'),
        consumer_secret=os.getenv('GITHUB_CLIENT_SECRET'),
        request_token_params={'scope': 'user:email'},
        base_url='https://api.github.com/',
        request_token_url=None,
        access_token_method='POST',
        access_token_url='https://github.com/login/oauth/access_token',
        authorize_url='https://github.com/login/oauth/authorize'
    )

    # Function to get OAuth token from session
    @github.tokengetter
    def get_github_oauth_token():
        return session.get('github_token')

    return github

# Authorization decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'github_token' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

