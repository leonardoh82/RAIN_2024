from functools import reduce
import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import re
import pandas as pd
import matplotlib.pyplot as plt

# Descargar recursos adicionales
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Función para eliminar ruido
def remove_noise(text):
    cleaned_text = re.sub(r'[\W\d_]+', ' ', text)
    return cleaned_text

# Función para normalizar el texto
def normalize(text):
    normalized_text = re.sub(r'[^\w\s]', '', text.lower())
    return normalized_text

# Función para eliminar palabras vacías
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    filtered_text = ' '.join(word for word in text.split() if word not in stop_words)
    return filtered_text

# Función para obtener las palabras más frecuentes
def get_top_words(text, n=20):
    tokens = text.split()
    word_freq = Counter(tokens)
    top_words = word_freq.most_common(n)
    return top_words

# Función para realizar stemming
def perform_stemming(text):
    stemmer = PorterStemmer()
    stemmed_text = ' '.join(stemmer.stem(word) for word in text.split())
    return stemmed_text

# Función para lematizar
def perform_lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return lemmatized_text

# Función para lematizar con PoS
def perform_lemmatization_with_pos(text):
    lemmatizer = WordNetLemmatizer()
    tagged_text = nltk.pos_tag(text.split())
    lemmatized_text = ' '.join(lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) if get_wordnet_pos(tag)=='v' else word for word, tag in tagged_text)
    return lemmatized_text

# Función para obtener el Part of Speech (PoS)
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'  # Adjetivo
    elif tag.startswith('V'):
        return 'v'  # Verbo
    elif tag.startswith('N'):
        return 'n'  # Sustantivo
    elif tag.startswith('R'):
        return 'r'  # Adverbio
    else:
        return None

# Obtener las primeras 10 oraciones del archivo cg73 del corpus Brown
cg73_sentences = brown.sents('cg73')[:10]

# Unir las oraciones en un solo texto
cg73_text = ' '.join(' '.join(sentence) for sentence in cg73_sentences)

# Paso 1: Eliminación de ruido
cleaned_text = remove_noise(cg73_text)

# Paso 2: Tokenización
tokens = word_tokenize(cleaned_text)

# Paso 3: Normalización
normalized_text = normalize(cleaned_text)

# Paso 4: Eliminación de palabras vacías
filtered_text = remove_stopwords(normalized_text)

# Paso 5: Obtener las 20 palabras más frecuentes
top_words_original = dict(get_top_words(filtered_text))

# Paso 6: Stemming
stemmed_text = perform_stemming(filtered_text)
top_words_stemmed = dict(get_top_words(stemmed_text))

# Paso 7: Lematización
lemmatized_text = perform_lemmatization(filtered_text)
top_words_lemmatized = dict(get_top_words(lemmatized_text))

# Paso 8: Lematización con PoS (solo para los verbos)
lemmatized_text_with_pos = perform_lemmatization_with_pos(filtered_text)
top_words_lemmatized_pos = dict(get_top_words(lemmatized_text_with_pos))

# Crear DataFrames para cada resultado
df_original = pd.DataFrame({'Palabra': list(top_words_original.keys()), 'Frecuencia Original': list(top_words_original.values())})
df_stemmed = pd.DataFrame({'Palabra': list(top_words_stemmed.keys()), 'Frecuencia Stemmed': list(top_words_stemmed.values())})
df_lemmatized = pd.DataFrame({'Palabra': list(top_words_lemmatized.keys()), 'Frecuencia Lematizada': list(top_words_lemmatized.values())})
df_lemmatized_pos = pd.DataFrame({'Palabra': list(top_words_lemmatized_pos.keys()), 'Frecuencia Lematizada con PoS': list(top_words_lemmatized_pos.values())})

# Combinar DataFrames
dfs_combined = [df_original, df_stemmed, df_lemmatized, df_lemmatized_pos]
df_combined = reduce(lambda left, right: pd.merge(left, right, on='Palabra', how='outer'), dfs_combined)

# Rellenar valores NaN con 0
df_combined.fillna(0, inplace=True)

# Graficar las 20 palabras más frecuentes de cada subítem
plt.figure(figsize=(12, 8))

for i, column in enumerate(df_combined.columns[1:], 1):
    plt.plot(df_combined['Palabra'], df_combined[column], marker='o', label=column)

plt.title('Comparación de las 20 palabras más frecuentes en diferentes procesamientos')
plt.xlabel('Palabra')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()