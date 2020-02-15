from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
# from .fromColab import Trainer, main_list, get_paths_to_file, get_num_examples, get_test_size
# from .fromColab import getOtherAutorizedSymbols, sentencesSeparator
from .utils import langues
main_path = "/content/drive/My Drive/datasets/YourVersion/"
checkpoint_dir ="/content/drive/My Drive/datasets/checkpoint_dir"

allModels = {}
actualModel = None


def home(request):
    # Charger les variables pour la session
    # Voir Tamghuo pour charger une seule fois pour toute l'application
    startapp()
    return render(request,'ts/index.html', {'langues': langues})


def about(request):
    return render(request, 'ts/about.html')


def menu(request):
    return render(request, 'ts/menu.html', {'langues': langues})


def actualizeModel(request):
    """Lorsque les langues changent"""
    global actualModel
    post = request.POST
    actualModel = allModels[index(post['input_language'], post['target_language'])]
    return  HttpResponse(json.dumps({"HTTPRESPONSE": "ok"}))

@csrf_exempt
def translate(request):
    """Lorsque les textes changent"""
    return  HttpResponse(
                json.dumps({"text": actualModel.translate(request.POST['text'])})
            )

def startapp():
    # Charger les variables pour la session
    l = len(langues)
    for i in range(l-1) :
        for j in range(i+1, l):
            pass
#            initialize(langues[i], langues[j])
#            initialize(langues[j], langues[i])

# def initialize(lang1, lang2):
#    global allModels
#    input_language = lang1["langue"]
#    target_language = lang2["langue"]
#    i = index(input_language, target_language)
#    allModels[i] = Trainer(
#                input_language=input_language,
#                target_language=target_language,
#                paths_to_dataset=get_paths_to_file(lang1, lang2),
#                sentencesSeparator=sentencesSeparator,
#                num_examples= get_num_examples(lang1["num_examples"] , lang2["num_examples"]),
#                test_size= get_test_size(lang1["test_size"] , lang2["test_size"]),
#                otherAutorizedSymbols=getOtherAutorizedSymbols([input_language, target_language])
#            )
#    allModels[i].compileAll(checkpoint_dir)

def index(input_language, target_language):
    return input_language+'_'+target_language


            