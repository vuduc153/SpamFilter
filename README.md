# SpamFilter

Python Django restaurant review spam classifier API, using Support Vector Machine model with Gaussian kernel.

Dataset consists of actual reviews from Yelp public dataset, which is available at 

https://www.yelp.com/dataset

and positive examples from email spams dataset at

https://www.kaggle.com/ozlerhakan/spam-or-not-spam-dataset

Text data are vectorized using Term Frequency, Inverse Document Frequency embedding.

## Installing

To setup the system on your local machine.

1. Install all required packages with:
```
sudo pip3 install -r requirements.txt
```
2. Go to src/API directory, run ```python3 manage.py runserver``` on terminal to launch your API server.

3. Run ```python3 manage.py migrate``` to set up your database.

## Usage

To use the API, send a GET request with your review content in the field 'content' to `http://spamfilterap1.herokuapp.com/api/`

The API server returns a JSON response with `{"ouput": 1}` for positive/spam inputs and correspondingly `{"output": 0}` for negative/non-spam inputs.
