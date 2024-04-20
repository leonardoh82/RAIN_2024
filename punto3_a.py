import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer

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
porter = PorterStemmer()
lancaster = LancasterStemmer()

#Utilizamos funciones disponibles en NLTK de Porter y Lancaster
tokens_porter = [porter.stem(token) for token in tokens]
tokens_lancaster = [lancaster.stem(token) for token in tokens]

#Resultados
print("Palabra original        Porter Stemming         Lancaster Stemming")
print("__________________________________________________________________________")
for i in range(len(tokens)):
    print(f"{tokens[i]:10}\t\t{tokens_porter[i]:10}\t\t{tokens_lancaster[i]:10}")
    