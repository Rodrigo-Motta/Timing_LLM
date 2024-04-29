from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import cophenet # cophenet is used to calculate the cophenetic correlation coefficient, which measures how well a hierarchical clustering preserves the original pairwise distances between data points.
from scipy.spatial.distance import squareform # squareform is used to convert a condensed distance matrix into a square distance matrix, which is a common format for hierarchical clustering algorithms.
from scipy.cluster.hierarchy import dendrogram, linkage
import nltk, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

# Function to remove stop words and lemmatize the words
def remove_stopwords_lemmatize(string_list):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return [' '.join(lemmatizer.lemmatize(word.lower()) for word in string.split() if word.lower() not in stop_words) for string in string_list]