# SpamFilter

Spam filtering module for Web Application

## Installing

Install all required packages with:
```
sudo pip3 install -r requirements.txt
```

## Usage

Go to src/API directory, run ```python3 manage.py runserver``` on terminal to launch your API server.
Next, run ```python3 manage.py migrate``` to set up your database.
To use the API, send a GET request with your message in the field 'content' to `your/server/url/api`
