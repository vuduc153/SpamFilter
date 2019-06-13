import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

data = pd.read_csv('../data/spam_or_not_spam.csv')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# remove empty line
data.dropna(inplace=True, how='any')

# split data into training set and testing set
text_train, text_test, label_train, label_test = train_test_split(data['email'], data['label'])


def preprocess(content):

    # transform to lower case
    content.lower()
    # tokenize
    content = word_tokenize(content)
    # filter out stopwords
    content = [w for w in content if w not in stop_words]
    # create a tag map
    tag_map = dict()
    tag_map['N'] = 'n'  # noun
    tag_map['J'] = 'a'  # adj
    tag_map['V'] = 'v'  # verb
    tag_map['R'] = 'r'  # adv

    for i in range(len(content)):
        word = content[i]
        # transform all symbols to 'SYMBOL'
        if re.search('[^A-Za-z]+', word) is not None:
            word = 'SYMBOL'
        # transform all word containing 'NUMBER' to 'NUMBER'
        if re.search('NUMBER', word) is not None:
            word = 'NUMBER'
        # lemmatize
        tag = pos_tag([word])[0][1]
        try:
            content[i] = lemmatizer.lemmatize(word, tag_map[tag[0]])
        except KeyError:
            content[i] = lemmatizer.lemmatize(word)

    # join all tokens back into text
    content = ' '.join(content)
    return content


def vectorize():

    vectorizer = TfidfVectorizer(lowercase=False, min_df=0.01, preprocessor=preprocess)
    global text_train, text_test
    text_train = vectorizer.fit_transform(text_train)
    text_test = vectorizer.transform(text_test)


def train_naive_bayes():

    model = naive_bayes.MultinomialNB()
    model.fit(text_train, label_train)

    # predict the labels on validation dataset
    prediction = model.predict(text_test)

    # use accuracy_score function to get the accuracy
    print("Naive Bayes Accuracy Score -> ", accuracy_score(prediction, label_test) * 100)


if __name__ == '__main__':
    vectorize()
    train_naive_bayes()


