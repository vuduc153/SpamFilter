import pickle
from ModelController import ModelController


if __name__ == '__main__':
    model = ModelController.load_model('SVM')
    processor = ModelController.load_preprocessor()
    text = input("Enter: ")
    text = processor.transform_vectorize([text])
    print(model.predict(text))
