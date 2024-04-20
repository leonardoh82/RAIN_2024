import os
from typing import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

#Descargar los recursos necesarios de nltk
nltk.download('stopwords')
nltk.download('punkt')


caracteres = ",.:'"
# obtener las stopwords en español
archivo = open('entrada/archivo.txt', 'r',encoding='utf-8') 
texto1 = archivo.read()
texto1=texto1.lower()

#Tokenizar el texto
tokens_texto1 = word_tokenize(texto1, language='spanish')

#quitar caracteres, puntuacion  
for word in tokens_texto1:
    if word in caracteres:
        tokens_texto1.remove(word)
# eliminar palabras vacias
stop_words = set(stopwords.words('spanish'))
tokens = [token for token in tokens_texto1 if token not in stop_words and token.isalnum()]

#obtener la frecuenia de cada termino
frecuencia_tokens = Counter(tokens)

#Ordenar de forma descendente 
frecuencia_tokens = dict(sorted(frecuencia_tokens.items(), key=lambda item: item[1], reverse=True))

#tabular la salida
#print("{:10}Término\t\t{:10d}Frecuencia")
print('{0:10}                  {1:10}'.format('Termino','Frecuencia'))
print("______________________________________")
for token, frecuencia in frecuencia_tokens.items():
    print(f"{token:10}\t\t{frecuencia:8d}")
