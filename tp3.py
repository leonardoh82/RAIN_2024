#Se tomo como referencia la documentacion contenida en 
# https://notebook.community/vitojph/2016progpln/notebooks/5-nltk-corpus
'''El Corpus de Brown fue el primer gran corpus orientado a tareas de PLN. 
Desarrollado en la Universidad de Brown, contiene más de un millón de palabras provenientes de 500 fuentes.
La principal catacterística de este corpus es que sus textos están categorizados por género.'''

import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Brown está formado por 500 documentos
#Brown está categorizado: los textos están agrupados según su género o temática
nltk.download('brown')
from nltk.corpus import brown

print('Destokenizar el documento')
brown_corpus =  brown.raw('cg73')
destokenizador = TreebankWordDetokenizer()
documento_destokenizado = destokenizador.detokenize(brown_corpus)
print('Documento Destoquenizados\n' + documento_destokenizado)

# Tokenizar en oraciones
sentences = sent_tokenize(brown_corpus)

# Mostrar las primeras 10 oraciones
print("Las primeras 10 oraciones del archivo cg73 son:")
for i, sentence in enumerate(sentences[:10], 1):
    print(f"{i}. {sentence}")

