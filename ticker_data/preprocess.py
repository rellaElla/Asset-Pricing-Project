import pandas as pd
from time import time
import nltk
nltk.download('stopwords')
import re
from emoji import demojize

def preprocess(texts, quiet=False):
    """
    Takes in tweets as a pd.Series. quiet=True will return time to clean data. Returns pd.Series. 
    Operations:
    1. All text lowercase
    2. Replace special characterize
    3. Remove emojis
    4. Remove repetitions
    5. Change contractions to full words
    6. Remove English stopwords
    7. Stem tweets 
    """
    start: time = time()

    # Lowercasing
    texts: pd.Series = texts.str.lower()

    # Remove special chars
    texts = texts.str.replace(r"(http|@)\S+", "")
    texts = texts.apply(demojize)
    texts = texts.str.replace(r"::", ": :")
    texts = texts.str.replace(r"’", "'")
    texts = texts.str.replace(r"[^a-z\':_]", " ")

    # Remove repetitions
    pattern: re.SRE_Pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    texts = texts.str.replace(pattern, r"\1")

    # Transform short negation form
    texts = texts.str.replace(r"(can't|cannot)", 'can not')
    texts = texts.str.replace(r"n't", ' not')

    # Remove stop words
    stopwords: nltk.stem = nltk.corpus.stopwords.words('english')

    stopwords.remove('not')
    stopwords.remove('nor')
    stopwords.remove('no')
    texts = texts.apply(
        lambda x: ' '.join([word for word in x.split() if word not in stopwords])
        )

    # Stem Tweets
    stemmer = nltk.stem.SnowballStemmer('english')
    texts = texts.apply(
        lambda x: ''.join([stemmer.stem(word) for word in x])
        )

    if not quiet:
        print("Time to clean up: {:.2f} sec".format(time() - start))

    return texts