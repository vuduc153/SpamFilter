# SpamFilter

Python Django restaurant review spam classifier API, using Support Vector Machine model with Gaussian kernel.

Dataset consists of actual reviews from Yelp public dataset, which is available at 

https://www.yelp.com/dataset

and positive examples from email spams dataset at

https://www.kaggle.com/ozlerhakan/spam-or-not-spam-dataset


## Installing

Install all required packages with:
```
sudo pip3 install -r requirements.txt
```

## Usage

Go to src/API directory, run ```python3 manage.py runserver``` on terminal to launch your API server.

Next, run ```python3 manage.py migrate``` to set up your database.

To use the API, send a GET request with your message in the field 'content' to `your/server/url/api`
