# Tooth
Tooth is not an Odontological Object Tagging Hub.

##Legal and Compliance Considerations

This project is designed with a strong emphasis on data minimization and privacy, avoiding persistent storage of sensitive information whenever possible, in alignment with the principles of the Brazilian General Data Protection Law (LGPD). Image processing is performed in a transient manner, reducing risks associated with unintended data retention. Additionally, the system does not implement user-generated content moderation or platform-like behavior, keeping it outside the scope of regulations such as the so-called “Felca Law” (ECA Digital).

##To Install

git clone https://github.com/UFSM-OdontoAI/Tooth.git

cd Tooth

git lfs install

git lfs pull

python -m venv .venv

source .venv/bin/activate

python -m pip install -r requirements.txt

python manage.py makemigrations imagemproc

python manage.py migrate

python manage.py collectstatic

python -m gunicorn myproject.wsgi:application --bind 0.0.0.0:9000

###Google OAuth Configuration

To enable Google authentication:

Go to the Google Cloud Console
https://console.cloud.google.com/
Create a project (or select an existing one)
Navigate to:
APIs & Services → Credentials
Click:
Create Credentials → OAuth Client ID
Application type:
Web application
Under Authorized redirect URIs, add exactly:
http://yourhost:9000/accounts/google/login/callback/

The URL must match exactly, including the trailing /

Save and copy:
client_id
client_secret
In Django Admin (http://yourhost:9000/admin):
Go to Social Applications
Create a new entry:
Provider: Google
Client ID: (client_id created on google cloud console)
Secret: (secret created on google cloud console)
Sites: select your active site
In settings.py, ensure:
SITE_ID = 1
(Recommended for development)

Add your account under:

OAuth Consent Screen → Test Users
