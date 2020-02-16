# import json
# from .fromColab import Trainer, main_list, get_paths_to_file, get_num_examples, get_test_size
# from .fromColab import getOtherAutorizedSymbols, sentencesSeparator
# from .utils import langues
# main_path = "/content/drive/My Drive/datasets/YourVersion/"
# checkpoint_dir ="/content/drive/My Drive/datasets/YourVersion/checkpoint_dir"

# allModels = {}
# class App():  
#     def __init__(self):
#         pass
#     def index(self,input_language, target_language):
#          return input_language+'_'+target_language

#     def startapp(self):
#         # Charger les variables pour la session
#         l = len(main_list)
#         for i in range(l-1) :
#             for j in range(i+1, l):
#                 self.initialize(main_list[i], main_list[j])
#                 self.initialize(main_list[j], main_list[i])

#     def initialize(self,lang1, lang2):
#         global allModels
#         global checkpoint_dir
#         global main_path
#         input_language = lang1["langue"]
#         target_language = lang2["langue"]
#         i = self.index(input_language, target_language)
#         allModels[i] = Trainer(
#                     input_language=input_language,
#                     target_language=target_language,
#                     paths_to_dataset=get_paths_to_file(lang1, lang2),
#                     sentencesSeparator=sentencesSeparator,
#                     num_examples= get_num_examples(lang1["num_examples"] , lang2["num_examples"]),
#                     test_size= get_test_size(lang1["test_size"] , lang2["test_size"]),
#                     otherAutorizedSymbols= getOtherAutorizedSymbols([input_language, target_language])
#                 )
#         allModels[i].compileAll(checkpoint_dir)
#         allModels[i].restoreCheckpoint()