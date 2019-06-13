import pandas as pd
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from TfidfProcess import TfidfProcess


def train_naive_bayes(train_data, train_label, test_data, test_label):

    model = naive_bayes.MultinomialNB()
    model.fit(train_data, train_label)

    # predict the labels on validation dataset
    prediction = model.predict(test_data)

    # use accuracy_score function to get the accuracy
    print("Naive Bayes Accuracy Score -> ", accuracy_score(prediction, test_label) * 100)


if __name__ == '__main__':
    data = pd.read_csv('../data/spam_or_not_spam.csv')
    # remove empty line
    data.dropna(inplace=True, how='any')
    # split data into training set and testing set
    text_train, text_test, label_train, label_test = train_test_split(data['email'], data['label'])

    text_processor = TfidfProcess()
    text_train = text_processor.fit_vectorize(text_train)
    text_test = text_processor.transform_vectorize(text_test)
    train_naive_bayes(text_train, label_train, text_test, label_test)
