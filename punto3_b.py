#Para realizar este punto y de acuerdo a los articulos consultados, tanto Porter como Lancaster
#fueron creador para aplicarse en el ideoma Ingles por lo que se puede aplicar al español, pero no de manera
#precisa, de tal manera que los resultados podrian ser no deseados. Es por esto que se aconseja aplicar otros
#algoritmos disponibles en la libreria NLTK como es el caso de SnowballStemer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer # Libreria espesifica para Español

#Descargar los recursos necesarios de nltk
nltk.download('stopwords')
nltk.download('punkt')


caracteres = ",.:'"
# obtener las stopwords en Ingles
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

# Stemming
snoball = SnowballStemmer('spanish')


#Utilizamos funciones disponibles en NLTK de SnowballStemmer
tokens_snowball = [snoball.stem(token) for token in tokens]

#Resultados
print("Palabra original\t\tSnowballStemmer")
print("--------------------------------------------")
for i in range(len(tokens)):
    print(f"{tokens[i]:10}\t\t\t{tokens_snowball[i]:10}")