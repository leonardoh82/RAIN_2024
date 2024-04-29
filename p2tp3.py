import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import re
import pandas as pd

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
def get_top_words(text, n=50):
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

# Obtener las10 oraciones del archivo cg73 del corpus Brown
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

# Paso 5: Obtener las 50 palabras más frecuentes
top_words = get_top_words(filtered_text)

# Paso 6: Stemming
stemmed_text = perform_stemming(filtered_text)

# Paso 7: Lematización
lemmatized_text = perform_lemmatization(filtered_text)

# Paso 8: Lematización con PoS (solo para los verbos)
lemmatized_text_with_pos = perform_lemmatization_with_pos(filtered_text)

# Paso 9: Representación tabular de los primeros 30 tokens
import pandas as pd

# Crear un DataFrame para mostrar los resultados
df = pd.DataFrame(columns=['Palabra Normal', 'Stemming', 'Lematización', 'Lematización con PoS (verbos)'])

for token in tokens[:30]:
    df = df._append({'Palabra Normal': token,
                    'Stemming': perform_stemming(token),
                    'Lematización': perform_lemmatization(token),
                    'Lematización con PoS (verbos)': perform_lemmatization_with_pos(token)}, ignore_index=True)

print(df)