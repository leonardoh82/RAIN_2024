#La Libreria utilizada para los n-gramas es ngrams
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#extrae el primer parrafo del archivo 
def extraer_primer_parrafo(archivo):
    with open(archivo, 'r', encoding='utf-8') as file:
        texto = file.read()

    # Dividir el texto en párrafos
    parrafos = texto.split('\n\n')

    # Tomar el primer párrafo
    primer_parrafo = parrafos[0]

    return primer_parrafo

# Ruta de tu archivo de texto
ruta = 'entrada/archivo.txt'

# Llamar a la función para extraer el primer párrafo
texto1 = extraer_primer_parrafo(ruta)
texto1=texto1.lower()

#Tokenizar el texto
tokens_texto1 = word_tokenize(texto1, language='spanish')

#quitar caracteres, puntuacion  
caracteres = ",.:'"
for word in tokens_texto1:
    if word in caracteres:
        tokens_texto1.remove(word)
# eliminar palabras vacias
stop_words = set(stopwords.words('spanish'))
tokens = [token for token in tokens_texto1 if token not in stop_words and token.isalnum()]


gramas_2 = list(ngrams(tokens, 2))

# Obtener los 3-gramas
gramas_3 = list(ngrams(tokens, 3))

# Mostrar los resultados
print("2-gramas:")
for grama in gramas_2:
    print(grama)

print("\n3-gramas:")
for grama in gramas_3:
    print(grama)
