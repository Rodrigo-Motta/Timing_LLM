import nltk
import random
from utils import remove_stopwords_lemmatize
from data.time_scales import *

# Download necessary NLTK data
nltk.downloader.Downloader()._update_index()
nltk.download('wordnet')
nltk.download('stopwords')

class DatasetLoader:
    """
    A class to load, preprocess, and manipulate textual data from a given module, specifically focusing on scales and 
    sentences. This class provides methods to remove stopwords, lemmatize text, concatenate scales, and scramble 
    sentences within the scales.

    Attributes:
    -----------
    all_names : list
        A list of all names defined in the provided module.
    
    filtered_names : filter
        A filtered list of names, excluding special names that start and end with '__'.
    
    list_names : list
        A list of variable names containing scales and sentences from the module.
    
    scales_raw : dict
        A dictionary containing the raw scales data from the module, with variable names as keys and sentence lists as values.
    
    scales_preprocessed : dict
        A dictionary containing preprocessed scales where stopwords have been removed and words have been lemmatized.

    Methods:
    --------
    __init__(module):
        Initializes the DatasetLoader with a given module, extracts the relevant data, and preprocesses it.

    scales_joint():
        Concatenates text representations of scales, maintaining sentence separation with added spaces after periods.
        Sets the attribute `scales_joint_raw` as a dictionary with keys as variable names and values as concatenated strings.

    scramble_joint():
        Randomly scrambles the order of sentences within each scale and concatenates them.
        Sets the attributes `scales_joint_scrambled` and `scales_joint_raw_scrambled` with scrambled sentences and their concatenated forms.
    """
    def __init__(self, module):
        # Get the list of all names defined in the module
        self.all_names = dir(module)
        # Filter out the special names that start and end with '__'
        self.filtered_names = filter(lambda name: not (name.startswith('__') and name.endswith('__')), self.all_names)
        self.list_names = list(self.filtered_names)

        # Create a dictionary containing scales and sentences
        self.scales_raw = {name: getattr(module, name) for name in self.list_names}

        # Process all variables in the dictionary
        self.scales_preprocessed = {variable: remove_stopwords_lemmatize(string_list) for variable, string_list in self.scales_raw.items()}

    def scales_joint(self):
        self.scales_joint_raw = {variable: "".join(string_list).replace(".", ". ") for variable, string_list in
                                 self.scales_raw.items()}

    def scramble_joint(self):
        self.scales_joint_scrambled = randomized_dict = {name: random.sample(values, len(values)) for name, values in self.scales_raw.items()}
        self.scales_joint_raw_scrambled = {variable: "".join(string_list).replace(".", ". ") for variable, string_list in
                                           self.scales_joint_scrambled.items()}