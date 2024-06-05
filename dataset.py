from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import cophenet # cophenet is used to calculate the cophenetic correlation coefficient, which measures how well a hierarchical clustering preserves the original pairwise distances between data points.
from scipy.spatial.distance import squareform # squareform is used to convert a condensed distance matrix into a square distance matrix, which is a common format for hierarchical clustering algorithms.
from scipy.cluster.hierarchy import dendrogram, linkage
import nltk, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from utils import remove_stopwords_lemmatize
# Here we import the scales we will use
import scales_time
import eiko_depression_scales
from scales_time import *
import random

dler = nltk.downloader.Downloader()
dler._update_index()

dler.download('wordnet')
dler.download('stopwords')

class DatasetLoader:
    """
    A class to load and preprocess datasets.

    ...

    Attributes
    ----------
    all_names : list
        List of all attribute names from the module scales_time.
    filtered_names : filter
        Filtered list of attribute names excluding special names.
    list_names : list
        List of filtered attribute names.
    scales_raw : dict
        Dictionary containing raw scales and their sentences.
    scales_preprocessed : dict
        Dictionary containing preprocessed scales with stopwords removed and lemmatized sentences.
    scales_joint_raw : dict
        Dictionary containing joint raw scales where sentences are joined and periods are properly spaced.
    scales_joint_scrambled : dict
        Dictionary containing scrambled raw scales.
    scales_joint_raw_scrambled : dict
        Dictionary containing joint scrambled raw scales where sentences are joined and periods are properly spaced.

    Methods
    -------
    __init__():
        Initializes the DatasetLoader by loading and preprocessing the scales.
    scales_joint():
        Creates a joint raw scale where sentences are joined and periods are properly spaced.
    scramble_joint():
        Scrambles the joint raw scales and creates a joint scrambled raw scale where sentences are joined and periods are properly spaced.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the DatasetLoader object.

        Parameters
        ----------
        None
        """
        # Get the list of all names defined in the module
        self.all_names = dir(scales_time)
        # Filter out the special names that start and end with '__'
        self.filtered_names = filter(lambda name: not (name.startswith('__') and name.endswith('__')), self.all_names)
        self.list_names = list(self.filtered_names)

        # Create a dictionary containing scales and sentences
        self.scales_raw = {name: getattr(scales_time, name) for name in self.list_names}

        # Process all variables in the dictionary
        self.scales_preprocessed = {variable: remove_stopwords_lemmatize(string_list) for variable, string_list in self.scales_raw.items()}

    def scales_joint(self):
        """
        Creates a joint raw scale where sentences are joined and periods are properly spaced.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.scales_joint_raw = {variable: "".join(string_list).replace(".", ". ") for variable, string_list in self.scales_raw.items()}

    def scramble_joint(self):
        """
        Scrambles the joint raw scales and creates a joint scrambled raw scale where sentences are joined and periods are properly spaced.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.scales_joint_scrambled = {name: random.sample(values, len(values)) for name, values in self.scales_raw.items()}
        self.scales_joint_raw_scrambled = {variable: "".join(string_list).replace(".", ". ") for variable, string_list in self.scales_joint_scrambled.items()}