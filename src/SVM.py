import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

text = list()


def pre_process():
    data = pd.read_csv('../data/spam_or_not_spam.csv')

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # remove empty line
    data['email'].dropna(inplace=True)

    for email in data['email']:
        # transform to lower case
        email.lower()
        # tokenize
        email = word_tokenize(email)
        # filter out stopwords
        email = [w for w in email if w not in stop_words]
        # create a tag map
        tag_map = dict()
        tag_map['N'] = 'n'  # noun
        tag_map['J'] = 'a'  # adj
        tag_map['V'] = 'v'  # verb
        tag_map['R'] = 'r'  # adv

        for i in range(len(email)):
            word = email[i]
            # transform all symbols to 'SYMBOL'
            if re.search('[^A-Za-z]+', word) is not None:
                word = 'SYMBOL'
            # transform all word containing 'NUMBER' to 'NUMBER'
            if re.search('NUMBER', word) is not None:
                word = 'NUMBER'
            # lemmatize
            tag = pos_tag([word])[0][1]
            try:
                email[i] = lemmatizer.lemmatize(word, tag_map[tag[0]])
            except KeyError:
                email[i] = lemmatizer.lemmatize(word)

        email = ' '.join(email)
        text.append(email)


def vectorize():
    vectorizer = TfidfVectorizer(lowercase=False, min_df=0.01)
    X = vectorizer.fit_transform(text)
    print(X.shape)


if __name__ == '__main__':
    pre_process()
    vectorize()

