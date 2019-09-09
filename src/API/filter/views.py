from django.http import JsonResponse
from .model.ModelController import ModelController

# Create your views here.


def predict(request):

    if request.method == 'GET' and request.GET.get('content'):

        text = request.GET.get('content')
        # load model from pickle files
        preprocessor = ModelController().load_preprocessor()
        model = ModelController().load_model()
        # process the text
        processed_text = preprocessor.transform_vectorize([text])
        # predict
        output = model.predict(processed_text)[0]

        return JsonResponse({'output': int(output)})

    return JsonResponse({'error': True})
