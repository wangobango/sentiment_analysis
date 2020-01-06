import nltk

try:
    nltk.data.find('stopwords')
except LookupError as err:
    nltk.download('stopwords')

try:
    nltk.data.find('punkt')
except LookupError as err:
    nltk.download('punkt')

try:
    nltk.data.find('wordnet')
except LookupError as err:
    nltk.download('wordnet')

# subprocess.run(['sudo python3 -m spacy download en_core_web_md'])