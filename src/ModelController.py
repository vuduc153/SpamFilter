import pickle


class ModelController:

    @staticmethod
    def save_model(model, name):
        filename = name + '.model'
        pickle.dump(model, open(filename, 'wb+'))

    @staticmethod
    def load_model(name):
        filename = name + '.model'
        model = pickle.load(open(filename, 'rb'))
        return model

    @staticmethod
    def save_preprocessor(processor):
        filename = 'TfIdf.proc'
        pickle.dump(processor, open(filename, 'wb+'))

    @staticmethod
    def load_preprocessor():
        filename = 'TfIdf.proc'
        preprocessor = pickle.load(open(filename, 'rb'))
        return preprocessor
