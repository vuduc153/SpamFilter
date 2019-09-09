import pickle
import sys
sys.path.append('filter/model/')


class ModelController:

    @staticmethod
    def save_model(model):
        filename = 'SVM.pkl'
        pickle.dump(model, open(filename, 'wb+'))

    @staticmethod
    def load_model():
        filename = 'filter/model/SVM.pkl'
        model = pickle.load(open(filename, 'rb'))
        return model

    @staticmethod
    def save_preprocessor(processor):
        filename = 'TfIdf.pkl'
        pickle.dump(processor, open(filename, 'wb+'))

    @staticmethod
    def load_preprocessor():
        filename = 'filter/model/TfIdf.pkl'
        preprocessor = pickle.load(open(filename, 'rb'))
        return preprocessor


