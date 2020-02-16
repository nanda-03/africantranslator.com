from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
# from .fromColab import Trainer, main_list, get_paths_to_file, get_num_examples, get_test_size
# from .fromColab import getOtherAutorizedSymbols, sentencesSeparator
from .utils import langues
from django.shortcuts import redirect
main_path = "/content/drive/My Drive/datasets/YourVersion/"
checkpoint_dir ="/content/drive/My Drive/datasets/checkpoint_dir"

aphabetNormal = ['a', 'b', 'c'] # ...

#Application scope
# from . import App
# actualModel = None

def admin(request):
    # app = App()
    # app.startapp()
    return redirect('home')

def home(request):
    # Charger les variables pour la session
    # Voir Tamghuo pour charger une seule fois pour toute l'application
    # App.startapp()
    # from . import allModels
    #print(allModels)
    return render(request,'tf/index.html', {'langues': langues})


def about(request):
    return render(request, 'tf/about.html')

def menu(request):
    return render(request, 'tf/menu.html', {'langues': langues})

@csrf_exempt
def actualizeModel(request):
    """Lorsque les langues changent"""
    # global actualModel
    # from . import allModels
    post = request.POST
    # app = App()
    print(post['input_language'])
    print(post['target_language'])
    # actualModel = allModels[app.index(post['input_language'], post['target_language'])]
    # return  HttpResponse(json.dumps({"HTTPRESPONSE": "ok", 'otherAutorizedSymbols':clean(actualModel.otherAutorizedSymbols)}))
    return  HttpResponse(json.dumps({"HTTPRESPONSE": "ok"}))

@csrf_exempt
def translate(request):
    """Lorsque les textes changent"""
    text = request.POST['text'] # actualModel.translate(request.POST['text'])
    return  HttpResponse(
                json.dumps({"text": text})
            )

def clean(otherAutorizedSymbols):
    global aphabetNormal
    result = []
    for letter in otherAutorizedSymbols.split(''):
        if letter not in aphabetNormal :
            result.append(letter)
    return result