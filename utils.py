"""
Utilities for Semantic Analysis in Psychological Measurement
=============================================================

This module provides functions for analyzing semantic similarities between
psychological scales using large language models and classical NLP methods.

Project: Self-report measures of subjective time: 
         An overview of existing measures and their semantic similarities


Authors: Rodrigo da Motta-Cabral, Thiago Augusto de Souza Bonifacio
Date: January 2026
Reviewed by: Andre Mascioli Cravo
"""
# =============================================================================
# STEP 1: IMPORTS
# =============================================================================


from scipy.stats import spearmanr
import pandas as pd # Needed for data preparation snippet

# Standard library
import os
import random
from typing import Union, Tuple, List, Dict, Optional

# Third-party: Core scientific computing
import numpy as np
import pandas as pd

# Third-party: NLP and text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Third-party: Machine learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Third-party: Statistical analysis
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster

# Third-party: Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Dict
from adjustText import adjust_text
from matplotlib.colors import ListedColormap

# Third-party: Deep learning embeddings
from sentence_transformers import SentenceTransformer, util

# Third-party: Classical embeddings (optional)
try:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# Third-party: OpenAI (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Third-party: Network analysis (optional)
try:
    import networkx as nx
    import community as community_louvain
    NETWORK_AVAILABLE = True
except ImportError:
    NETWORK_AVAILABLE = False

# Third-party: Interactive visualization (optional)
try:
    import holoviews as hv
    from bokeh.io import output_file, show
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    INTERACTIVE_VIZ_AVAILABLE = True
except ImportError:
    INTERACTIVE_VIZ_AVAILABLE = False

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Standardize Plots
def configure_matplotlib():
    """Configure matplotlib with Roboto font and high DPI settings."""
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Updated line: Try Roboto first, but fall back to Arial or generic sans-serif if missing
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Arial', 'DejaVu Sans', 'sans-serif']

    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.bbox'] = 'tight'


# =============================================================================
# SECTION 1: TEXT PREPROCESSING
# =============================================================================

def remove_stopwords_lemmatize(string_list: List[str]) -> List[str]:
    """
    Remove stopwords and lemmatize words in a list of strings.

    Parameters
    ----------
    string_list : list of str
        List of strings to be processed.

    Returns
    -------
    list of str
        List of processed strings with stopwords removed and words lemmatized.
        
    Examples
    --------
    >>> texts = ["The cats are running", "Dogs were playing"]
    >>> processed = remove_stopwords_lemmatize(texts)
    >>> print(processed)
    ['cat running', 'dog playing']
    """
    if not string_list or not isinstance(string_list, list):
        raise ValueError("Input must be a non-empty list of strings")
    
    # Initialize the set of stop words and the lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Process each string in the list
    return [
        ' '.join(
            lemmatizer.lemmatize(word.lower())
            for word in string.split()
            if word.lower() not in stop_words
        )
        for string in string_list if string  # Skip empty strings
    ]


# =============================================================================
# SECTION 2: EMBEDDING GENERATION
# =============================================================================

def get_embedding(data, model_list: List[str], num_refs: int, 
                  num_scrambles: int) -> np.ndarray:
    """
    Generates embeddings for scrambled joint raw scales using multiple models.

    Parameters
    ----------
    data : DatasetLoader
        DatasetLoader object containing the data.
    model_list : list of str
        List of model names to be used for embeddings.
    num_refs : int
        Number of references.
    num_scrambles : int
        Number of scrambles.

    Returns
    -------
    np.ndarray
        Array of embeddings for the dataset with joint sentences.
    """
    similarities_array = np.zeros((len(model_list), num_refs, num_scrambles, num_scrambles))
    similarities_array[:] = np.nan

    # Loop over each model
    for iModel, Model in enumerate(model_list):
        print(Model)
        embed_list = []
        model = SentenceTransformer(Model)

        # Loop over each reference and scramble data multiple times
        for iRef, Ref in enumerate(data.list_names):
            aux = []
            for scrambles in range(num_scrambles):
                data.scramble_joint()
                embeddings = [model.encode(data.scales_joint_raw_scrambled[Ref], convert_to_tensor=False)]
                aux.append(embeddings)
            embed_list.append(np.array(aux).mean(axis=0))

    return np.array(embed_list)


def get_embedding_gpt(data, model_list: List[str], num_refs: int, 
                      num_scrambles: int, api_key: Optional[str] = None) -> np.ndarray:
    """
    Generate embeddings using OpenAI's embedding models.

    Parameters
    ----------
    data : DatasetLoader
        DatasetLoader object containing the data.
    model_list : list of str
        List of model names (currently only supports one at a time).
    num_refs : int
        Number of references.
    num_scrambles : int
        Number of scrambles to average over.
    api_key : str, optional
        OpenAI API key. If None, uses OPENAI_API_KEY environment variable.

    Returns
    -------
    np.ndarray
        Array of averaged embeddings, shape (num_refs, embedding_dim).
        
    Raises
    ------
    ImportError
        If OpenAI package is not installed.
    ValueError
        If API key is not provided and not found in environment.
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    # Get API key from parameter or environment
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    client = OpenAI(api_key=api_key)
    embed_list = []

    for iModel, Model in enumerate(model_list):
        print(f"Processing model: {Model}")

        # Loop over each reference and scramble data multiple times
        for iRef, Ref in enumerate(data.list_names):
            scramble_embeddings = []
            
            for _ in range(num_scrambles):
                data.scramble_joint()
                response = client.embeddings.create(
                    input=data.scales_joint_raw_scrambled[Ref],
                    model="text-embedding-3-small"
                )
                embeddings = response.data[0].embedding
                scramble_embeddings.append(embeddings)
            
            # Average across scrambles
            avg_embedding = np.array(scramble_embeddings).mean(axis=0)
            embed_list.append(avg_embedding)

    return np.array(embed_list)


def get_embedding_classical(data, model_list: List[str], num_refs: int, 
                           num_scrambles: int) -> np.ndarray:
    """
    Generate embeddings using classical NLP methods (TF-IDF or Doc2Vec).

    Parameters
    ----------
    data : DatasetLoader
        DatasetLoader object containing the data.
    model_list : list of str
        List of model names: 'tfidf' or 'doc2vec'.
    num_refs : int
        Number of references.
    num_scrambles : int
        Number of scrambles to average over.

    Returns
    -------
    np.ndarray
        Array of averaged embeddings, shape (num_refs, embedding_dim).
        
    Raises
    ------
    ImportError
        If gensim is not installed (for Doc2Vec).
    ValueError
        If unsupported model type is specified.
    """
    # Loop over each model (TF-IDF or Doc2Vec)
    for iModel, Model in enumerate(model_list):
        print(f"Processing model: {Model}")
        embed_list = []

        if Model == "tfidf":
            # Combine all texts before scrambling to fit a global vocabulary
            all_texts = list(data.scales_joint_raw.values())
            vectorizer = TfidfVectorizer()
            vectorizer.fit(all_texts)

        elif Model == "doc2vec":
            if not GENSIM_AVAILABLE:
                raise ImportError("Gensim not installed. Install with: pip install gensim")
            
            # Create and train Doc2Vec model
            documents = list(data.scales_joint_raw.values())
            random.shuffle(documents)
            training_documents = documents[:len(documents) // 2]
            tokenized_data = [word_tokenize(doc.lower()) for doc in training_documents]
            
            tagged_data = [
                TaggedDocument(words=words, tags=[str(idx)])
                for idx, words in enumerate(tokenized_data)
            ]
            
            doc2vec_model = Doc2Vec(
                vector_size=100, 
                window=5, 
                min_count=1, 
                workers=4,
                epochs=100
            )
            doc2vec_model.build_vocab(tagged_data)
            doc2vec_model.train(
                tagged_data,
                total_examples=doc2vec_model.corpus_count,
                epochs=doc2vec_model.epochs
            )
        else:
            raise ValueError(f"Unsupported model type: {Model}. Use 'tfidf' or 'doc2vec'")

        # Loop over each reference and scramble data multiple times
        for iRef, Ref in enumerate(data.list_names):
            scramble_embeddings = []

            for _ in range(num_scrambles):
                data.scramble_joint()
                scrambled_sentence = data.scales_joint_raw_scrambled[Ref]

                if Model == "tfidf":
                    embeddings = vectorizer.transform([scrambled_sentence]).toarray()[0]
                elif Model == "doc2vec":
                    embeddings = doc2vec_model.infer_vector(scrambled_sentence.split())

                scramble_embeddings.append(embeddings)

            # Average across scrambles
            avg_embedding = np.vstack(scramble_embeddings).mean(axis=0)
            embed_list.append(avg_embedding)

    return np.array(embed_list)


def analyze_embeddings(embed_arr):
    """Analyze embedding characteristics."""
    X = np.array(embed_arr)
    
    print(f"Shape: {X.shape}")
    print(f"Mean: {X.mean():.4f}")
    print(f"Std: {X.std():.4f}")
    print(f"Min: {X.min():.4f}")
    print(f"Max: {X.max():.4f}")
    
    # Check for zero variance features
    variances = X.var(axis=0)
    zero_var = (variances < 1e-10).sum()
    print(f"\nFeatures with near-zero variance: {zero_var}/{X.shape[1]}")
    
    # Check feature variance distribution
    print(f"Variance range: {variances.min():.6f} to {variances.max():.6f}")
    
    return X

# =============================================================================
# SECTION 3: SIMILARITY CALCULATIONS
# =============================================================================

def similarities(data, model_list: List[str], num_refs: int, 
                scrambled: bool = False) -> np.ndarray:
    """
    Calculate pairwise similarities for joint scales using multiple models.

    Parameters
    ----------
    data : DatasetLoader
        DatasetLoader object containing the data.
    model_list : list of str
        List of Sentence Transformer model names.
    num_refs : int
        Number of references.
    scrambled : bool, optional
        Whether to use scrambled joint raw scales (default is False).

    Returns
    -------
    np.ndarray
        Array of similarities, shape (len(model_list), num_refs, num_refs).
        Upper triangle filled, lower triangle contains NaN.
    """
    similarities_array = np.zeros((len(model_list), num_refs, num_refs))
    similarities_array[:] = np.nan

    for iModel, Model in enumerate(model_list):
        print(f"Processing model: {Model}")
        model = SentenceTransformer(Model)

        # Encode data with or without scrambling
        if scrambled:
            ref_embeddings = [
                model.encode(data.scales_joint_raw_scrambled[Ref], convert_to_tensor=True)
                for Ref in data.list_names
            ]
        else:
            ref_embeddings = [
                model.encode(data.scales_joint_raw[Ref], convert_to_tensor=True)
                for Ref in data.list_names
            ]

        # Calculate pairwise similarities (upper triangle only)
        for iRef in range(num_refs):
            for iComp in range(iRef + 1, num_refs):
                similarity = util.pytorch_cos_sim(
                    ref_embeddings[iRef],
                    ref_embeddings[iComp]
                )
                similarities_array[iModel, iRef, iComp] = similarity

    return similarities_array


def similarities_average(data, model_list: List[str], num_refs: int, 
                        num_scrambles: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate average similarities with scrambling for robustness.

    Parameters
    ----------
    data : DatasetLoader
        DatasetLoader object containing the data.
    model_list : list of str
        List of Sentence Transformer model names.
    num_refs : int
        Number of references.
    num_scrambles : int
        Number of scrambling iterations.

    Returns
    -------
    average_similarities : np.ndarray
        Mean similarities, shape (len(model_list), num_refs, num_refs).
    std_similarities : np.ndarray
        Standard deviation of similarities, same shape.
    """
    similarities_array = np.zeros((len(model_list), num_refs, num_refs))
    similarities_array[:] = np.nan

    for iModel, Model in enumerate(model_list):
        print(f"Processing model: {Model}")
        model = SentenceTransformer(Model)
        all_similarities = np.zeros((num_scrambles, num_refs, num_refs))

        # Perform multiple scrambles
        for n in range(num_scrambles):
            data.scramble_joint()
            ref_embeddings = [
                model.encode(data.scales_joint_raw_scrambled[Ref], convert_to_tensor=True)
                for Ref in data.list_names
            ]

            # Calculate pairwise similarities
            for iRef in range(num_refs):
                for iComp in range(iRef + 1, num_refs):
                    similarity = util.pytorch_cos_sim(
                        ref_embeddings[iRef],
                        ref_embeddings[iComp]
                    )
                    similarities_array[iModel, iRef, iComp] = similarity

            all_similarities[n, :, :] = similarities_array[iModel, :, :]

    # Compute statistics across scrambles
    average_similarities = all_similarities.mean(axis=0)[np.newaxis, ...]
    std_similarities = all_similarities.std(axis=0)[np.newaxis, ...]

    return average_similarities, std_similarities


def similarities_average_gpt(data, model_list: List[str], num_refs: int, 
                            num_scrambles: int, 
                            api_key: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate average similarities using OpenAI embeddings.

    Parameters
    ----------
    data : DatasetLoader
        DatasetLoader object containing the data.
    model_list : list of str
        List of model names (supports one at a time).
    num_refs : int
        Number of references.
    num_scrambles : int
        Number of scrambling iterations.
    api_key : str, optional
        OpenAI API key.

    Returns
    -------
    average_similarities : np.ndarray
        Mean similarities, shape (len(model_list), num_refs, num_refs).
    std_similarities : np.ndarray
        Standard deviation of similarities, same shape.
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package not installed")
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required")
    
    client = OpenAI(api_key=api_key)
    similarities_array = np.zeros((len(model_list), num_refs, num_refs))
    similarities_array[:] = np.nan

    for iModel, Model in enumerate(model_list):
        print(f"Processing model: {Model}")
        all_similarities = np.zeros((num_scrambles, num_refs, num_refs))

        # Perform multiple scrambles
        for n in range(num_scrambles):
            data.scramble_joint()
            ref_embeddings = [
                client.embeddings.create(
                    input=data.scales_joint_raw_scrambled[Ref],
                    model="text-embedding-3-small"
                ).data[0].embedding
                for Ref in data.list_names
            ]

            # Calculate pairwise similarities
            for iRef in range(num_refs):
                for iComp in range(iRef + 1, num_refs):
                    similarity = util.pytorch_cos_sim(
                        ref_embeddings[iRef],
                        ref_embeddings[iComp]
                    )
                    similarities_array[iModel, iRef, iComp] = similarity

            all_similarities[n, :, :] = similarities_array[iModel, :, :]

    # Compute statistics across scrambles
    average_similarities = all_similarities.mean(axis=0)[np.newaxis, ...]
    std_similarities = all_similarities.std(axis=0)[np.newaxis, ...]

    return average_similarities, std_similarities


def similarities_average_classical(data, model_list: List[str], num_refs: int,
                                  num_scrambles: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate average similarities using classical NLP methods.

    Parameters
    ----------
    data : DatasetLoader
        DatasetLoader object containing the data.
    model_list : list of str
        List of model names: 'tfidf' or 'doc2vec'.
    num_refs : int
        Number of references.
    num_scrambles : int
        Number of scrambling iterations.

    Returns
    -------
    average_similarities : np.ndarray
        Mean similarities, shape (len(model_list), num_refs, num_refs).
    std_similarities : np.ndarray
        Standard deviation of similarities, same shape.
    """
    similarities_array = np.zeros((len(model_list), num_refs, num_refs))
    similarities_array[:] = np.nan

    for iModel, Model in enumerate(model_list):
        print(f"Processing model: {Model}")
        all_similarities = np.zeros((num_scrambles, num_refs, num_refs))

        if Model == "tfidf":
            all_texts = list(data.scales_joint_raw.values())
            vectorizer = TfidfVectorizer()
            vectorizer.fit(all_texts)

        elif Model == "doc2vec":
            if not GENSIM_AVAILABLE:
                raise ImportError("Gensim not installed")
            
            documents = list(data.scales_joint_raw.values())
            random.shuffle(documents)
            training_documents = documents[:len(documents) // 2]
            tokenized_data = [word_tokenize(doc.lower()) for doc in training_documents]
            
            tagged_data = [
                TaggedDocument(words=words, tags=[str(idx)])
                for idx, words in enumerate(tokenized_data)
            ]
            
            doc2vec_model = Doc2Vec(vector_size=100, window=3, min_count=1, 
                                   workers=4, epochs=1000)
            doc2vec_model.build_vocab(tagged_data)
            doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count,
                              epochs=doc2vec_model.epochs)

        # Perform multiple scrambles
        for n in range(num_scrambles):
            data.scramble_joint()

            if Model == "tfidf":
                ref_embeddings = [
                    vectorizer.transform([data.scales_joint_raw_scrambled[Ref]]).toarray().mean(axis=0)
                    for Ref in data.list_names
                ]
            elif Model == "doc2vec":
                ref_embeddings = [
                    doc2vec_model.infer_vector(
                        word_tokenize(data.scales_joint_raw_scrambled[Ref].lower())
                    )
                    for Ref in data.list_names
                ]

            # Calculate pairwise similarities
            for iRef in range(num_refs):
                for iComp in range(iRef + 1, num_refs):
                    similarity = util.pytorch_cos_sim(
                        ref_embeddings[iRef],
                        ref_embeddings[iComp]
                    )
                    similarities_array[iModel, iRef, iComp] = similarity

            all_similarities[n, :, :] = similarities_array[iModel, :, :]

    # Compute statistics across scrambles
    average_similarities = all_similarities.mean(axis=1)
    std_similarities = all_similarities.std(axis=1)

    return average_similarities, std_similarities


def intra_similarities(data, model_list: List[str], num_refs: int, 
                      num_scrambles: int) -> np.ndarray:
    """
    Calculate intra-similarities for scrambled scales.
    
    This computes similarities between different scrambled versions of the
    same scale to measure internal consistency.

    Parameters
    ----------
    data : DatasetLoader
        DatasetLoader object containing the data.
    model_list : list of str
        List of model names to be used for embeddings.
    num_refs : int
        Number of references.
    num_scrambles : int
        Number of scrambles.

    Returns
    -------
    np.ndarray
        Array of cosine similarities, shape (len(model_list), num_refs, 
        num_scrambles, num_scrambles).
    """
    similarities_array = np.zeros((len(model_list), num_refs, num_scrambles, num_scrambles))
    similarities_array[:] = np.nan

    for iModel, Model in enumerate(model_list):
        print(f"Processing model: {Model}")
        model = SentenceTransformer(Model)

        for iRef, Ref in enumerate(data.list_names):
            embed_list = []

            # Generate embeddings for multiple scrambles
            for _ in range(num_scrambles):
                data.scramble_joint()
                embeddings = model.encode(
                    data.scales_joint_raw_scrambled[Ref],
                    convert_to_tensor=True
                )
                embed_list.append(embeddings)

            # Calculate similarities between all pairs of scrambles
            for i in range(num_scrambles):
                for j in range(i, num_scrambles):
                    similarity = util.pytorch_cos_sim(embed_list[i], embed_list[j])
                    similarities_array[iModel, iRef, i, j] = similarity

    return similarities_array


def calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate pairwise cosine similarity matrix from embeddings.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Embedding matrix of shape (n_samples, embedding_dim)
        
    Returns:
    --------
    similarity_matrix : np.ndarray
        Symmetric similarity matrix of shape (n_samples, n_samples)
    """

    similarity_matrix = cosine_similarity(embeddings)
    
    # Set diagonal to NaN or 1.0 depending on your needs
    np.fill_diagonal(similarity_matrix, 1.0)
    
    return similarity_matrix

# =============================================================================
# SECTION 4: CLUSTERING METHODS
# =============================================================================

def hierarchical_clustering(embed_arr: np.ndarray, data=None, 
                           method: str = 'average', 
                           metric: str = 'cosine',
                           labels: Optional[List[str]] = None,
                           plot: bool = True) -> Dict:
    """
    Perform hierarchical clustering on embeddings.

    Parameters
    ----------
    embed_arr : np.ndarray
        Array of embeddings.
    data : DatasetLoader, optional
        DatasetLoader object (used for labels if provided).
    method : str, default='average'
        Linkage method: 'single', 'complete', 'average', 'ward', etc.
    metric : str, default='cosine'
        Distance metric.
    labels : list of str, optional
        Labels for dendrogram. If None and data is provided, uses data.list_names.
    plot : bool, default=True
        Whether to display dendrogram plot.

    Returns
    -------
    dict
        Dictionary containing:
        - 'linkage_matrix': Linkage matrix
        - 'cophenetic_correlation': Cophenetic correlation coefficient
        - 'method': Linkage method used
        - 'metric': Distance metric used
    """
    if labels is None and data is not None:
        labels = data.list_names
    
    # Perform hierarchical clustering
    linked = linkage(embed_arr, method=method, metric=metric)

    # Compute the Cophenetic Correlation Coefficient
    cophenetic_corr = cophenet(linked, pdist(embed_arr, metric=metric))[1]
    print(f"Cophenetic Correlation Coefficient: {cophenetic_corr:.4f}")

    # Visualize the dendrogram
    if plot:
        plt.figure(figsize=(12, 6))
        dendrogram(
            linked,
            orientation='top',
            labels=labels,
            distance_sort='descending',
            show_leaf_counts=True
        )
        plt.xlabel('Concepts')
        plt.ylabel(f'{metric.capitalize()} Distance')
        plt.title(f'Hierarchical Clustering Dendrogram ({method} linkage)')
        plt.tight_layout()
        plt.show()

    return {
        'linkage_matrix': linked,
        'cophenetic_correlation': cophenetic_corr,
        'method': method,
        'metric': metric
    }


def perform_kmeans_clustering(embed_arr: np.ndarray, data, 
                              n_clusters: int) -> Tuple[pd.DataFrame, float]:
    """
    Perform K-means clustering on embeddings.

    Parameters
    ----------
    embed_arr : np.ndarray
        Array of embeddings.
    data : DatasetLoader
        DatasetLoader object containing the data.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing embeddings, names, and cluster labels.
    silhouette_avg : float
        Silhouette score for the clustering solution.
    """
    df = pd.DataFrame(embed_arr)
    df['Names'] = data.list_names

    # Perform K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    ).fit(df.iloc[:, :-1])
    
    df['cluster'] = pd.Categorical(kmeans.labels_)
    
    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(df.iloc[:, :-2], kmeans.labels_)
    print(f"Silhouette Score: {silhouette_avg:.4f}")

    return df, silhouette_avg


def cluster_high_dim_embeddings(embed_arr: np.ndarray, data, 
                               n_clusters: int = 3, 
                               n_components: int = 15) -> Tuple:
    """
    Cluster high-dimensional embeddings with proper preprocessing.
    """
    X = np.array(embed_arr)
    
    # Step 1: Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("After scaling:")
    print(f"  Mean: {X_scaled.mean():.4f}")
    print(f"  Std: {X_scaled.std():.4f}")
    
    # Step 2: PCA - reduce dimensions
    # Note: This limits components if you have very few data points
    n_components = min(n_components, X.shape[0] // 2)
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nPCA reduction: {X.shape[1]} -> {X_pca.shape[1]} dimensions")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    print("Per-component variance:")
    for i, var in enumerate(pca.explained_variance_ratio_[:5], 1):
        print(f"  PC{i}: {var:.2%}")
    
    # Step 3: Cluster on PCA space
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    ).fit(X_pca)
    
    silhouette_pca = silhouette_score(X_pca, kmeans.labels_)
    
    # --- FIXED SECTION START ---
    # Create result DataFrame
    df = pd.DataFrame(X)
    df['Names'] = data.list_names
    df['cluster'] = pd.Categorical(kmeans.labels_)
    
    # Dynamically save ALL PCA components found (PC1 to PCn)
    for i in range(X_pca.shape[1]):
        df[f'PC{i+1}'] = X_pca[:, i]
    # --- FIXED SECTION END ---

    print(f"\nSilhouette Score (PCA space): {silhouette_pca:.3f}")
    
    # Compare with original space
    kmeans_orig = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    silhouette_orig = silhouette_score(X_scaled, kmeans_orig.labels_)
    print(f"Silhouette Score (original space): {silhouette_orig:.3f}")
    print(f"Improvement: {silhouette_pca - silhouette_orig:+.3f}")
    
    return df, silhouette_pca, X_pca, pca, kmeans


def find_optimal_clusters(embed_arr: np.ndarray, data, 
                         max_clusters: int = 10) -> Dict[int, float]:
    """
    Find optimal number of clusters using Silhouette Score.

    Parameters
    ----------
    embed_arr : np.ndarray
        Array of embeddings.
    data : DatasetLoader
        DatasetLoader object.
    max_clusters : int, default=10
        Maximum number of clusters to test.

    Returns
    -------
    dict
        Dictionary with number of clusters as keys and silhouette scores as values.
    """
    X = np.array(embed_arr)
    scores = {}
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            max_iter=300,
            n_init=10,
            random_state=42
        ).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        scores[k] = score
        print(f"K={k}: Silhouette Score = {score:.3f}")
    
    return scores


def optimize_pca_clusters(embed_arr: np.ndarray, 
                         max_components: int = 20, 
                         max_clusters: int = 8) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Find optimal number of PCA components and clusters.

    Parameters
    ----------
    embed_arr : np.ndarray
        Array of embeddings.
    max_components : int, default=20
        Maximum PCA components to test.
    max_clusters : int, default=8
        Maximum number of clusters to test.

    Returns
    -------
    df_results : pd.DataFrame
        DataFrame with all tested combinations and scores.
    best : pd.Series
        Best configuration found.
    """
    X = np.array(embed_arr)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    
    for n_comp in range(2, min(max_components + 1, X.shape[0] // 2)):
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca)
            score = silhouette_score(X_pca, kmeans.labels_)
            
            results.append({
                'n_components': n_comp,
                'n_clusters': k,
                'silhouette': score,
                'explained_var': pca.explained_variance_ratio_.sum()
            })
    
    df_results = pd.DataFrame(results)
    best = df_results.loc[df_results['silhouette'].idxmax()]
    
    print("Best configuration:")
    print(f"  PCA components: {int(best['n_components'])}")
    print(f"  Clusters: {int(best['n_clusters'])}")
    print(f"  Silhouette: {best['silhouette']:.3f}")
    print(f"  Explained variance: {best['explained_var']:.2%}")
    
    return df_results, best


def analyze_cluster_composition(df, label="Items"):
    """
    Display detailed cluster composition with statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'cluster', 'Names', 'PC1', 'PC2' columns
    label : str
        Label for output (e.g., 'Items' or 'Constructs')
    """
    print(f"\n{'='*70}")
    print(f"Cluster Composition Analysis: {label}")
    print("="*70)
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        cluster_members = cluster_data['Names'].tolist()
        
        print(f"\nCluster {cluster_id} ({len(cluster_members)} members):")
        print("-" * 70)
        for i, member in enumerate(cluster_members, 1):
            print(f"  {i:2d}. {member}")
        
        # PCA space statistics
        pc1_mean = cluster_data['PC1'].mean()
        pc2_mean = cluster_data['PC2'].mean()
        pc1_std = cluster_data['PC1'].std()
        pc2_std = cluster_data['PC2'].std()
        
        print(f"\nPCA Space Position:")
        print(f"  PC1: {pc1_mean:+.2f} ± {pc1_std:.2f}")
        print(f"  PC2: {pc2_mean:+.2f} ± {pc2_std:.2f}")

def find_cluster_representatives(embeddings, labels, scale_names):
    """
    For each cluster, finds the scale with the lowest average cosine distance
    to all other scales in the same cluster.
    
    Args:
        embeddings:   np.array shape (n_scales, n_dims) — PCA-reduced embeddings (e.g. 15 dims)
        labels:       np.array shape (n_scales,) — k-means cluster assignments (.labels_)
        scale_names:  list of strings with scale names
    
    Returns:
        dict with representative scale and distance info per cluster
    """
    results = {}
    
    for cluster_id in np.unique(labels):
        # Indices of scales in this cluster
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_names = [scale_names[i] for i in cluster_indices]
        
        # Cosine distance matrix (n x n) within the cluster
        dist_matrix = cosine_distances(cluster_embeddings)
        
        # Exclude self-distances (diagonal = 0) before averaging
        np.fill_diagonal(dist_matrix, np.nan)
        mean_distances = np.nanmean(dist_matrix, axis=1)
        
        # Scale with the lowest mean distance
        best_local_idx = np.argmin(mean_distances)
        
        results[cluster_id] = {
            "representative": cluster_names[best_local_idx],
            "mean_cosine_dist": mean_distances[best_local_idx],
            "all_scales": cluster_names,
            "all_mean_dists": dict(zip(cluster_names, mean_distances))
        }
    
    return results


def find_cluster_representatives_optimal(embeddings, labels, scale_names):
    """
    Finds the most central scale in each cluster by calculating the cluster's 
    geometric centroid and finding the scale with the highest Cosine Similarity to it.
    
    Args:
        embeddings:   np.array shape (n_scales, n_dims) — PCA-reduced embeddings
        labels:       np.array shape (n_scales,) — k-means cluster assignments
        scale_names:  list of strings with scale names
    
    Returns:
        dict with representative scale, raw centroid similarity, and normalized scores
    """
    results = {}
    
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_names = [scale_names[i] for i in cluster_indices]
        
        # --- 1. Calculate the Centroid ---
        # The geometric average of all embeddings in this cluster
        centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
        
        # --- 2. Calculate Similarity to Centroid ---
        # Returns raw similarities ranging from -1.0 to 1.0 (Higher is better)
        similarities = cosine_similarity(cluster_embeddings, centroid).flatten()
        
        # --- 3. Normalize for a clean 0 to 1 Centrality Score ---
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        
        if max_sim == min_sim:
            centrality_scores = np.ones_like(similarities)
        else:
            # Min-Max scaling: Lowest similarity = 0.0, Highest similarity = 1.0
            centrality_scores = (similarities - min_sim) / (max_sim - min_sim)
        
        # The most central item has the highest similarity to the centroid
        best_local_idx = np.argmax(similarities)
        
        results[cluster_id] = {
            "representative": cluster_names[best_local_idx],
            "centroid_similarity": similarities[best_local_idx], 
            "centrality_score": centrality_scores[best_local_idx], # Will be 1.0
            "all_scales": cluster_names,
            "all_centroid_similarities": dict(zip(cluster_names, similarities)),
            "all_centrality_scores": dict(zip(cluster_names, centrality_scores))
        }
    
    return results



# =============================================================================
# SECTION 5: DIMENSIONALITY REDUCTION
# =============================================================================

def apply_pca(arr: np.ndarray, n_components: int, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply PCA to reduce dimensionality of embeddings.

    Parameters
    ----------
    arr : np.ndarray
        Array of embeddings.
    n_components : int
        Number of principal components.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_pca : np.ndarray
        Transformed embeddings.
    explained_variance_ratio : np.ndarray
        Explained variance ratio for each component.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(arr)
    
    return X_pca, pca.explained_variance_ratio_


def apply_tsne(embeddings: np.ndarray, 
              n_components: int = 2, 
              perplexity: int = 30,
              random_state: int = 42,
              **kwargs) -> np.ndarray:
    """
    Apply t-SNE for dimensionality reduction.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of embeddings.
    n_components : int, default=2
        Number of dimensions for embedding.
    perplexity : int, default=30
        t-SNE perplexity parameter.
    random_state : int, default=42
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to TSNE.

    Returns
    -------
    np.ndarray
        t-SNE transformed embeddings.
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        **kwargs
    )
    
    return tsne.fit_transform(embeddings)

# =============================================================================
# SECTION 6: DATA TRANSFORMATION UTILITIES
# =============================================================================

def convert_arr_to_pandas(similarity_array: np.ndarray, 
                         list_names: List[str]) -> pd.DataFrame:
    """
    Convert a similarity array to a symmetric pandas DataFrame.

    Parameters
    ----------
    similarity_array : np.ndarray
        A 2D or 3D array of similarities. If 3D, averages along first axis.
    list_names : list of str
        List of names for columns and index.

    Returns
    -------
    pd.DataFrame
        Symmetric DataFrame of similarities with diagonal = 1.

    Raises
    ------
    ValueError
        If list_names length doesn't match array dimensions.
    """
    if len(list_names) != similarity_array.shape[-1]:
        raise ValueError(
            f"Length of list_names ({len(list_names)}) must match array "
            f"dimension ({similarity_array.shape[-1]})"
        )

    # Average across models if 3D array
    if similarity_array.ndim == 3:
        data = np.nanmean(similarity_array, axis=0)
    else:
        data = similarity_array
    
    # Create DataFrame and make symmetric
    df = pd.DataFrame(data=data, columns=list_names, index=list_names)
    df = df.replace(np.nan, 0)
    df = (df + df.T).replace(0.0, 1.0)

    return df


def min_max_norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply min-max normalization to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with values in [0, 1].
    """
    min_val = df.min().min()
    max_val = df.max().max()
    
    return (df - min_val) / (max_val - min_val)


def standard_scaler(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard scaling (z-score normalization) to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to standardize.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with mean=0 and std=1.
    """
    return (df - df.mean()) / df.std()


def remove_triangle(df: pd.DataFrame) -> np.ndarray:
    """
    Remove upper triangle and diagonal from a symmetric matrix.
    
    Useful for extracting unique pairwise values from similarity matrices.

    Parameters
    ----------
    df : pd.DataFrame
        Symmetric DataFrame (e.g., correlation or similarity matrix).

    Returns
    -------
    np.ndarray
        1D array of lower triangle values (excluding diagonal).
    """
    df = df.astype(float)
    
    # Set upper triangle and diagonal to NaN
    df.values[np.triu_indices_from(df, k=0)] = np.nan
    
    # Flatten and remove NaN values
    values = df.values.flatten()
    values = values[~np.isnan(values)]
    
    return values.reshape((1, len(values)))


# =============================================================================
# SECTION 7: VISUALIZATION FUNCTIONS
# =============================================================================

# Global configuration for all figures
def configure_matplotlib():
    """Configure matplotlib with Roboto font and high DPI settings."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.bbox'] = 'tight'
    
# Call this at module import
configure_matplotlib()

# -------------------------
# 7.1: Distribution Plots
# -------------------------

def plot_intra_barplot(arr: np.ndarray, data, save_path: str = None, 
                       figsize: tuple = (7, 4), format: str = "svg"):
    """
    Plot bar chart of intra-similarity scores with error bars.

    Parameters
    ----------
    arr : np.ndarray
        Array of similarities, shape (n_models, n_refs, n_scrambles, n_scrambles).
    data : DatasetLoader
        DatasetLoader object with list_names.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(7, 4)
        Figure size as (width, height) in inches.
    format : str, default="svg"
        Format to save the figure (e.g., 'svg', 'pdf', 'png', 'jpg').
    """
    mean = np.nanmean(np.nanmean(arr[0], axis=-1), axis=-1)
    error = np.nanstd(np.nanstd(arr[0], axis=-1), axis=-1)

    plt.figure(figsize=figsize)
    plt.bar(x=data.list_names, height=mean, yerr=error, capsize=5)
    plt.xticks(rotation=90)
    plt.ylabel('Cosine Similarity')
    plt.title('Intra-Similarity Scores by Scale')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format=format)
    plt.show()


# -------------------------
# 7.2: Hierarchical & Heatmap Visualizations
# -------------------------

from scipy.cluster.hierarchy import set_link_color_palette

def plot_dendrogram_clusters(df_similarities: pd.DataFrame, 
                             threshold: float = 0.35,
                             custom_colors: list = None,
                             save_path: str = "dendrogram_clusters.svg",
                             figsize: tuple = (10, 10),
                             format: str = "svg") -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Plot dendrogram with custom colors and matching colored labels.
    
    Parameters
    ----------
    df_similarities : pd.DataFrame
        Symmetric similarity matrix.
    threshold : float, default=0.35
        Distance threshold for cluster coloring.
    custom_colors : list of str, optional
        List of hex codes or color names (e.g., ['#FF0000', 'green', 'blue']).
        These will cycle for the clusters found below the threshold.
    save_path : str, default="dendrogram_clusters.svg"
        Path to save the figure.
    figsize : tuple, default=(10, 10)
        Figure size as (width, height) in inches.
    format : str, default="svg"
        Format to save the figure (e.g., 'svg', 'pdf', 'png', 'jpg').
    
    Returns
    -------
    Z : np.ndarray
        Linkage matrix from hierarchical clustering.
    df_clusters : pd.DataFrame
        DataFrame with cluster assignments.
    """

    # --- 1. Calculate Distances ---
    similarities = df_similarities.values
    distances = 1 - similarities
    distances = np.clip(distances, 0, None)
    np.fill_diagonal(distances, 0)
    distances_condensed = squareform(distances)

    # --- 2. Clustering ---
    Z = linkage(distances_condensed, method='average')
    c, _ = cophenet(Z, distances_condensed)
    print(f"Cophenetic correlation: {c:.4f}")

    # --- 3. Setup Custom Colors ---
    if custom_colors:
        # Set the palette for clusters below threshold
        set_link_color_palette(custom_colors)
    
    # --- 4. Plotting ---
    fig, ax = plt.subplots(figsize=figsize) 
    
    # Capture the return value (ddata) to get color info
    ddata = dendrogram(
        Z,
        orientation='right',
        labels=df_similarities.index,
        leaf_font_size=13, 
        color_threshold=threshold,
        above_threshold_color='#AAAAAA', # Color for links above threshold (grey)
        ax=ax
    )
    
    # --- 5. Color the Text Labels ---
    y_labels = ax.get_ymajorticklabels()
    for label, color in zip(y_labels, ddata['leaves_color_list']):
        label.set_color(color)

    # Reset palette to default so it doesn't affect other plots later
    set_link_color_palette(None)

    # Styling
    ax.set_xlabel('Distance (1 - Similarity)', fontsize=14, fontweight='bold')
    plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight', format=format)
    plt.show()

    # --- 6. Return Data ---
    cluster_labels = fcluster(Z, t=threshold, criterion='distance')
    df_clusters = pd.DataFrame({
        'Scale': df_similarities.index,
        'Cluster': cluster_labels
    })

    return Z, df_clusters


def plot_heatmap(df: pd.DataFrame, title: str = "Similarity Heatmap", 
                cmap: str = 'viridis', annot: bool = False,
                save_path: str = None, figsize: tuple = (8, 7), format: str = "svg"):
    """
    Plot a heatmap of the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Data to visualize.
    title : str, default="Similarity Heatmap"
        Plot title.
    cmap : str, default='viridis'
        Colormap name.
    annot : bool, default=False
        Whether to annotate cells with values.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(8, 7)
        Figure size as (width, height) in inches.
    format : str, default="svg"
        Format to save the figure (e.g., 'svg', 'pdf', 'png', 'jpg').
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=annot, cmap=cmap, square=True, cbar_kws={'label': 'Similarity'})
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format=format)
    plt.show()


def plot_dendrogram_and_heatmap(df_similarities: pd.DataFrame,
                                save_path: str = None, figsize: tuple = (8, 7), 
                                format: str = "svg"):
    """
    Plot aligned dendrogram and heatmap using seaborn clustermap.

    Parameters
    ----------
    df_similarities : pd.DataFrame
        Symmetric similarity matrix.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(8, 7)
        Figure size as (width, height) in inches.
    format : str, default="svg"
        Format to save the figure (e.g., 'svg', 'pdf', 'png', 'jpg').
    """
    df_similarities = df_similarities.fillna(0)

    # Convert to distances for clustering
    similarities = df_similarities.values
    distances = 1 - similarities
    np.fill_diagonal(distances, 0)
    distances = squareform(distances)

    Z = linkage(distances, method='average', metric='cosine')
    clusters = fcluster(Z, 3, criterion='maxclust')

    # Create row colors
    unique_clusters = np.unique(clusters)
    colors_palette = sns.color_palette("Set2", len(unique_clusters))
    lut = dict(zip(unique_clusters, colors_palette))
    row_colors = pd.Series(clusters).map(lut)

    # Create clustermap
    g = sns.clustermap(
        df_similarities,
        metric='euclidean',
        method='average',
        cmap='coolwarm',
        figsize=figsize,
        annot=False,
        cbar_kws={'label': 'Similarity Score'},
        xticklabels=True,
        yticklabels=True,
        col_cluster=False,
        row_colors=row_colors,
        dendrogram_ratio=(.1, .2)
    )

    g.fig.suptitle('Aligned Dendrogram and Heatmap', fontsize=16, y=1.01)
    g.ax_heatmap.set_xlabel('Questionnaires', fontsize=12)
    g.ax_heatmap.set_ylabel('Questionnaires', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format=format)
    plt.show()

# -------------------------
# 7.3: Dimensionality Reduction Plots
# -------------------------

def plot_2d_pca(df: pd.DataFrame, pca_model, 
                cluster_labels: Optional[Dict] = None,
                label_col: str = None,
                save_path: str = None, 
                figsize: tuple = (5, 5), 
                format: str = "svg",
                ax: plt.Axes = None,          # <-- NEW: Allows passing an existing subplot
                title: str = 'K-Means Clustering in PCA Space'): # <-- NEW: Customizable title
    """Plots cluster visualization in PCA Space.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'PC1', 'PC2' and 'cluster' columns.
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model.
    cluster_labels : dict, optional
        Dictionary mapping cluster IDs to names.
    label_col : str, optional
        Column name to use for data point text labels.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(5, 5)
        Figure size as (width, height) in inches.
    format : str, default="svg"
        Format to save the figure (e.g., 'svg', 'pdf', 'png', 'jpg').
    ax : plt.Axes, optional
        Allows passing an existing subplot
    title : str, optional
        Customizable title
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    
    if cluster_labels is None:
        cluster_labels = {i: f'Cluster {i+1}' for i in df['cluster'].unique()}
    
    if ax is None:
        fig, current_ax = plt.subplots(figsize=figsize)
        is_standalone = True
    else:
        current_ax = ax
        is_standalone = False
        fig = current_ax.figure
    
    colors_palette = ['#4600b0', '#b604b3', "#dd0c3a", "#f07762", "#f5cd91"]
    texts = []
    
    # 1. Plot the Scatter Points
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
        color = colors_palette[cluster_id % len(colors_palette)]
        
        current_ax.scatter(
            cluster_data['PC1'], cluster_data['PC2'],
            c=color, label=label, s=60, alpha=0.7,
            edgecolors='black', linewidth=1.3
        )
        
        # 2. Create Text Objects
        for idx, row in cluster_data.iterrows():
            if label_col:
                txt_label = str(row[label_col])
            else:
                try:
                    txt_label = str(int(idx) + 1)
                except (ValueError, TypeError):
                    txt_label = str(idx)
            
            t = current_ax.text(
                row['PC1'], row['PC2'], 
                txt_label, 
                color=color, 
                fontsize=11, 
                fontweight='bold',
                ha='left', va='bottom'
            )
            texts.append(t)
        
    if texts:
        adjust_text(
            texts, 
            x=df['PC1'].values, 
            y=df['PC2'].values, 
            ax=current_ax,
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
            force_points=5.0, force_text=0.1, expand_points=(2.2, 2.2)
        )

    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = pca_model.explained_variance_ratio_[1] * 100
    
    current_ax.set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=12, fontweight='bold')
    current_ax.set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=12, fontweight='bold')
    current_ax.set_title(title, fontsize=13, fontweight='bold')
    current_ax.legend(fontsize=10, loc='best')
    current_ax.grid(True, alpha=0.3, linestyle='--') 
    
    if is_standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format=format)
        plt.show()
    
    return current_ax

def plot_2d_pca_with_legend(df: pd.DataFrame, pca_model, 
                            cluster_labels: Optional[Dict] = None,
                            label_col: str = None,
                            description_col: str = None,
                            save_path: str = None, figsize: tuple = (9, 5), 
                            format: str = "svg"):
    """
    Plots PCA clusters (Left) and a detailed colored item list (Right).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'PC1', 'PC2' and 'cluster' columns.
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model.
    cluster_labels : dict, optional
        Dictionary mapping cluster IDs to names.
    label_col : str
        Column used for short text on the plot (e.g., 'ID' or Index).
    description_col : str
        Column used for the full name in the side legend.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(9, 5)
        Figure size as (width, height) in inches.
    format : str, default="svg"
        Format to save the figure (e.g., 'svg', 'pdf', 'png', 'jpg').
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    
    if cluster_labels is None:
        cluster_labels = {i: f'Cluster {i+1}' for i in df['cluster'].unique()}
    
    # Create figure with 2 panels: Left (Plot) is wider than Right (Legend)
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1.5, 1]})
    ax_plot = axes[0]
    ax_legend = axes[1]
    
    # Colors
    colors_palette = ['#4600b0', '#b604b3', "#dd0c3a", "#f07762", "#d4a35a"]
    
    texts = []
    
    # --- 1. LEFT PANEL: SCATTER PLOT ---
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
        color = colors_palette[cluster_id % len(colors_palette)]
        
        ax_plot.scatter(
            cluster_data['PC1'], cluster_data['PC2'],
            c=color, label=label, s=60, alpha=0.7,
            edgecolors='black', linewidth=1.0
        )
        
        # Add labels to points
        for idx, row in cluster_data.iterrows():
            # If label_col is provided, use it; otherwise use the Index
            short_label = str(row[label_col]) if label_col else str(idx)
            
            t = ax_plot.text(
                row['PC1'], row['PC2'], 
                short_label, 
                color=color, 
                fontsize=10, 
                fontweight='bold',
                ha='left', va='bottom'
            )
            texts.append(t)

    # Adjust text positions to avoid overlap
    if texts:
        adjust_text(
            texts, 
            x=df['PC1'].values, y=df['PC2'].values, ax=ax_plot,
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
            force_points=3.0, force_text=0.2, expand_points=(1.5, 1.5)
        )

    # Axis Labels
    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = pca_model.explained_variance_ratio_[1] * 100
    ax_plot.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=12, fontweight='bold')
    ax_plot.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=12, fontweight='bold')
    ax_plot.set_title('K-Means Clustering', fontsize=14, fontweight='bold')
    ax_plot.legend(fontsize=10, loc='upper left')
    ax_plot.grid(True, alpha=0.2, linestyle='--')

    # --- 2. RIGHT PANEL: DETAILED LEGEND ---
    ax_legend.axis('off')
    
    # We will print text line by line, starting from the top
    y_pos = 1.0
    line_height = 1.0 / (len(df) + 5) # Dynamically calculate spacing based on number of items
    
    # Iterate through clusters to keep the list grouped
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        color = colors_palette[cluster_id % len(colors_palette)]
        
        # Add a small header for the cluster
        ax_legend.text(0, y_pos, f"{cluster_labels.get(cluster_id, f'Cluster {cluster_id}')}", 
                       color=color, fontsize=10, fontweight='bold', va='top')
        y_pos -= line_height
        
        for idx, row in cluster_data.iterrows():
            short_label = str(row[label_col]) if label_col else str(idx)
            
            # Get description (if provided), otherwise use "Item {idx}"
            full_desc = str(row[description_col]) if description_col else f"Item {idx}"
            
            # Format: "1 : Apple"
            text_line = f"{short_label} : {full_desc}"
            
            ax_legend.text(0.05, y_pos, text_line, 
                           color=color, fontsize=9, va='top', fontfamily='Roboto')
            
            y_pos -= line_height
        
        # Add extra space between clusters
        y_pos -= (line_height * 0.5)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format=format)
    
    return fig

def plot_2d_pca_2panel(df: pd.DataFrame, pca_model, kmeans_model,
                       cluster_labels: Optional[Dict] = None,
                       label_col: str = None,
                       save_path: str = None, figsize: tuple = (9, 5), 
                       format: str = "svg"):
    """
    Create 2-panel figure with PCA clustering (left) and scree plot (right).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'PC1', 'PC2' and 'cluster' columns.
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model.
    kmeans_model : sklearn.cluster.KMeans
        Fitted KMeans model.
    cluster_labels : dict, optional
        Dictionary mapping cluster IDs to names.
    label_col : str, optional
        Column name to use for data point text labels.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(9, 5)
        Figure size as (width, height) in inches.
    format : str, default="svg"
        Format to save the figure (e.g., 'svg', 'pdf', 'png', 'jpg').
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    
    if cluster_labels is None:
        cluster_labels = {i: f'Cluster {i}' for i in df['cluster'].unique()}
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel A: Cluster visualization
    colors_palette = ['#4600b0', '#b604b3', "#dd0c3a", "#f07762", "#d4a35a"]
    
    texts = []
    
    # 1. Plot the Scatter Points
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
        color = colors_palette[cluster_id % len(colors_palette)]
        
        axes[0].scatter(
            cluster_data['PC1'], cluster_data['PC2'],
            c=color, label=label, s=50, alpha=0.7,
            edgecolors='black', linewidth=1.3
        )
        
        # 2. Create Text Objects
        for idx, row in cluster_data.iterrows():
            txt_label = str(row[label_col]) if label_col else str(idx)
            
            # TRICK: We intentionally don't center the text perfectly.
            # We assume it should start slightly above/right to give the
            # repulsion algorithm a "hint" of which direction to go.
            t = axes[0].text(
                row['PC1'], row['PC2'], 
                txt_label, 
                color=color, 
                fontsize=10, 
                fontweight='bold',
                ha='left', va='bottom' # Start alignment off-center
            )
            texts.append(t)
        
    # --- ADJUST_TEXT CONFIGURATION ---
    if texts:
        adjust_text(
            texts, 
            x=df['PC1'].values,  # Force repulsion from these X coordinates
            y=df['PC2'].values,  # Force repulsion from these Y coordinates
            ax=axes[0],
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
            
            # TUNED PARAMETERS FOR "STUCK" LABELS:
            force_points=5.0,        # Very high repulsion from points
            force_text=0.1,          # Lower repulsion between texts (prioritize points)
            expand_points=(2.2, 2.2) # Pretend points are 2.2x bigger than they really are
        )
    # ---------------------------------

    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = pca_model.explained_variance_ratio_[1] * 100
    
    axes[0].set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=12, fontweight='bold')
    axes[0].set_title('A. K-Means Clustering in PCA Space', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(False) # or axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Panel B: Scree plot
    var_explained = pca_model.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(var_explained)
    
    x_pos = np.arange(1, len(var_explained) + 1)
    axes[1].bar(x_pos, var_explained, alpha=0.6, color='steelblue', label='Individual')
    axes[1].plot(x_pos, cumulative_var, 'ro-', linewidth=2, markersize=6, label='Cumulative')
    axes[1].axhline(y=80, color='gray', linestyle='--', linewidth=1, label='80% threshold')
    
    axes[1].set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('B. Variance Explained by PCA Components', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[1].set_xticks(x_pos[::2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format=format)
    
    return fig


def plot_3d_pca(df: pd.DataFrame, pca_model, 
                cluster_labels: Optional[Dict] = None,
                label_col: str = None,
                save_path: str = None, figsize: tuple = (7, 5), format: str = "svg"):
    """
    Create a standalone 3D visualization of PCA clusters.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'PC1', 'PC2', 'PC3' and 'cluster' columns.
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model (used for variance explained percentages).
    cluster_labels : dict, optional
        Dictionary mapping cluster IDs to names (e.g., {0: 'Control', 1: 'Test'}).
    label_col : str, optional
        Column name to use for data point text labels.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(7, 5)
        Figure size as (width, height) in inches.
    format : str, default="svg"
        Format to save the figure (e.g., 'svg', 'pdf', 'png', 'jpg').
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    # 1. Setup
    if cluster_labels is None:
        cluster_labels = {i: f'Cluster {i}' for i in df['cluster'].unique()}
    
    # Check if we have PC3 (Crucial based on your previous fix)
    if 'PC3' not in df.columns:
        raise ValueError("DataFrame must contain 'PC3' column. Ensure your clustering function saves at least 3 components.")

    # Create single figure (adjusted size for standalone 3D plot)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    colors_palette = ['#4600b0', '#b604b3', "#dd0c3a", "#f07762", "#d4a35a"]
    
    # 2. Plotting Loop
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
        # Cyclical color picker
        color = colors_palette[cluster_id % len(colors_palette)]
        
        # Plot 3D Scatter
        ax.scatter(
            cluster_data['PC1'], cluster_data['PC2'], cluster_data['PC3'],
            c=color, label=label, s=60, alpha=0.7,
            edgecolors='black', linewidth=0.5
        )
        
        # Add labels to points
        for idx, row in cluster_data.iterrows():
            txt_label = str(row[label_col]) if label_col else str(idx)
            # Only label if the string is not empty/NaN
            if txt_label and txt_label.lower() != 'nan':
                ax.text(
                    row['PC1'], row['PC2'], row['PC3'], 
                    txt_label, 
                    color='black', fontsize=10, fontweight='bold',
                    zorder=20
                )

    # 3. Formatting
    # Calculate variance explained
    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = pca_model.explained_variance_ratio_[1] * 100
    var3 = pca_model.explained_variance_ratio_[2] * 100
    
    ax.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=10, fontweight='bold')
    ax.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=10, fontweight='bold')
    ax.set_zlabel("") 
    
    # Manually place the Z-label on the 2D Figure canvas (Nuclear Option)
    # x=0.04 places it on the far left, y=0.5 centers it vertically
    fig.text(0.33, 0.38, f'PC3 ({var3:.1f}%)', 
             va='center', rotation='vertical', 
             fontsize=12, fontweight='bold')

    ax.set_title('3D K-Means Clustering', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Set a default viewing angle for better initial perspective
    ax.view_init(elev=30, azim=25)
    
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.9)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format=format)

    return fig


def plot_3d_pca_2panel(df: pd.DataFrame, pca_model, kmeans_model,
                       cluster_labels: Optional[Dict] = None,
                       label_col: str = None,
                       save_path: str = None, figsize: tuple = (9.5, 5.5), 
                       format: str = "svg"):
    """
    Create publication-ready figure with 3D clustering results (Panel A) and Scree Plot (Panel B).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'PC1', 'PC2', 'PC3' and 'cluster' columns.
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model.
    kmeans_model : sklearn.cluster.KMeans
        Fitted KMeans model.
    cluster_labels : dict, optional
        Dictionary mapping cluster IDs to names.
    label_col : str, optional
        Column name to use for data point text labels.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(9.5, 5.5)
        Figure size as (width, height) in inches.
    format : str, default="svg"
        Format to save the figure (e.g., 'svg', 'pdf', 'png', 'jpg').
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if cluster_labels is None:
        cluster_labels = {i: f'Cluster {i}' for i in df['cluster'].unique()}
    
    # Create figure with 2 subplots (Panel A is 3D, Panel B is 2D)
    fig = plt.figure(figsize=figsize)
    
    # --- Panel A: 3D Cluster Visualization ---
    ax1 = fig.add_subplot(121, projection='3d')
    colors_palette = ['#4600b0', '#b604b3', "#dd0c3a", "#f07762", "#d4a35a"]
    
    # Check if we have PC3
    if 'PC3' not in df.columns:
        raise ValueError("DataFrame must contain 'PC3' column for 3D plotting.")

    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
        color = colors_palette[cluster_id % len(colors_palette)]
        
        # Plot 3D Scatter
        ax1.scatter(
            cluster_data['PC1'], cluster_data['PC2'], cluster_data['PC3'],
            c=color, label=label, s=50, alpha=0.7,
            edgecolors='black', linewidth=0.5
        )
        
        # Add simple labels
        for idx, row in cluster_data.iterrows():
            txt_label = str(row[label_col]) if label_col else str(idx)
            ax1.text(
                row['PC1'], row['PC2'], row['PC3'], 
                txt_label, 
                color='black', fontsize=9, fontweight='bold',
                zorder=20  # Try to bring text to front
            )

    # Calculate variance explained
    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = pca_model.explained_variance_ratio_[1] * 100
    var3 = pca_model.explained_variance_ratio_[2] * 100
    
    ax1.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=10, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=10, fontweight='bold')
    ax1.set_zlabel(f'PC3 ({var3:.1f}%)', fontsize=10, fontweight='bold')
    ax1.set_title('A. 3D K-Means Clustering', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    
    # --- Panel B: Scree Plot (Standard 2D) ---
    ax2 = fig.add_subplot(122)
    
    var_explained = pca_model.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(var_explained)
    x_pos = np.arange(1, len(var_explained) + 1)
    
    ax2.bar(x_pos, var_explained, alpha=0.6, color='steelblue', label='Individual')
    ax2.plot(x_pos, cumulative_var, 'ro-', linewidth=2, markersize=6, label='Cumulative')
    ax2.axhline(y=80, color='gray', linestyle='--', linewidth=1, label='80% threshold')
    
    ax2.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Variance Explained by PCA Components', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_xticks(x_pos[::2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format=format)
    
    return fig


# -------------------------
# 7.4: Publication-Ready Figures
# -------------------------

def create_publication_figure(df: pd.DataFrame, pca_model, kmeans_model,
                             cluster_labels: Optional[Dict] = None,
                             save_path: str = None, figsize: tuple = (9.5, 5.5), 
                             format: str = "svg"):
    """
    Create publication-ready figure showing PCA clustering results.

    Parameters
    ----------
    df : pd.DataFrame
        Clustered data with PC1, PC2, and cluster columns.
    pca_model : PCA
        Fitted PCA model.
    kmeans_model : KMeans
        Fitted KMeans model.
    cluster_labels : dict, optional
        Mapping of cluster IDs to semantic labels.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(9.5, 5.5)
        Figure size as (width, height) in inches.
    format : str, default="svg"
        Format to save the figure (e.g., 'svg', 'pdf', 'png', 'jpg').

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if cluster_labels is None:
        cluster_labels = {i: f'Cluster {i}' for i in df['cluster'].unique()}
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel A: Cluster visualization
    colors_palette = ['#4600b0', '#b604b3', "#dd0c3a", "#f07762", "#d4a35a"]
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
        color = colors_palette[cluster_id % len(colors_palette)]
        
        axes[0].scatter(
            cluster_data['PC1'], cluster_data['PC2'],
            c=color, label=label, s=120, alpha=0.7,
            edgecolors='black', linewidth=1.5
        )
    
    # Plot centroids
    centroids = kmeans_model.cluster_centers_
    axes[0].scatter(
        centroids[:, 0], centroids[:, 1],
        c='gold', marker='*', s=300, edgecolors='black',
        linewidth=2, label='Centroids', zorder=5
    )
    
    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = pca_model.explained_variance_ratio_[1] * 100
    
    axes[0].set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=12, fontweight='bold')
    axes[0].set_title('A. K-Means Clustering in PCA Space', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10, loc='best')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Panel B: Scree plot
    var_explained = pca_model.explained_variance_ratio_ * 100
    cumulative_var = np.cumsum(var_explained)
    
    x_pos = np.arange(1, len(var_explained) + 1)
    axes[1].bar(x_pos, var_explained, alpha=0.6, color='steelblue', label='Individual')
    axes[1].plot(x_pos, cumulative_var, 'ro-', linewidth=2, markersize=6, label='Cumulative')
    axes[1].axhline(y=80, color='gray', linestyle='--', linewidth=1, label='80% threshold')
    
    axes[1].set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('B. Variance Explained by PCA Components', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
    axes[1].set_xticks(x_pos[::2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format=format)
    
    return fig

def plot_dual_similarity_matrices(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    title1: str = "Similarity Matrix 1",
    title2: str = "Similarity Matrix 2",
    cmap: str = "viridis",
    annot: bool = False,
    figsize: tuple = (14, 6),
    save_path: str = None,
    format: str = "svg",
    dpi: int = 300,
    panel_labels: tuple = ("A", "B"),
    font_size: int = 11,
    title_font_size: int = 12,
    shared_scale: bool = False,
    scale_override: list = None,
):
    n = df1.shape[0]
    numeric_labels = list(range(1, n + 1))

    def get_scale(df, override):
        data_min, data_max = df.values.min(), df.values.max()
        if override is None:
            return data_min, data_max
        vmin, vmax = override
        return (data_min if vmin is None else vmin,
                data_max if vmax is None else vmax)

    if scale_override:
        scales = [get_scale(df1, scale_override[0]), get_scale(df2, scale_override[1])]
    elif shared_scale:
        vmin = min(df1.values.min(), df2.values.min())
        vmax = max(df1.values.max(), df2.values.max())
        scales = [(vmin, vmax), (vmin, vmax)]
    else:
        scales = [(df1.values.min(), df1.values.max()),
                  (df2.values.min(), df2.values.max())]

    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(
        1, 4,
        figure=fig,
        width_ratios=[1, 0.18, 1, 0.055],
        wspace=0.08,
    )

    ax1      = fig.add_subplot(gs[0, 0])
    ax2      = fig.add_subplot(gs[0, 2])
    cbar_ax  = fig.add_subplot(gs[0, 3]) 

    datasets = list(zip([ax1, ax2], [df1, df2], [title1, title2], scales))

    for idx, (ax, df, title, (vmin, vmax)) in enumerate(datasets):

        df_plot         = df.copy()
        df_plot.index   = numeric_labels
        df_plot.columns = numeric_labels

        sns.heatmap(
            df_plot,
            ax=ax,
            cmap=cmap,
            annot=annot,
            square=True,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            cbar=False,
            xticklabels=1,
            yticklabels=1,
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=font_size - 1)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=font_size - 1)
        ax.set_title(title, fontsize=title_font_size, fontweight="bold", pad=8)
        ax.text(
            -0.12, 1.04, panel_labels[idx],
            transform=ax.transAxes,
            fontsize=title_font_size + 2,
            fontweight="bold",
            va="bottom", ha="left",
        )

    vmin2, vmax2 = scales[1]
    norm = mcolors.Normalize(vmin=vmin2, vmax=vmax2)
    sm   = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Similarity", fontsize=font_size, labelpad=8)
    cbar_ax.tick_params(labelsize=font_size - 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", format=format, dpi=dpi)
        print(f"Saved → {save_path}")

    plt.show()

# -------------------------
# 7.5: Network Visualizations (Optional)
# -------------------------

def plot_chord_diagram(graph, save_path: str = "chord_diagram.html"):
    """
    Create chord diagram for network visualization.
    
    Note: Requires holoviews and bokeh packages.

    Parameters
    ----------
    graph : networkx.Graph
        Network graph.
    save_path : str, default="chord_diagram.html"
        Path to save the HTML file.
    """
    if not INTERACTIVE_VIZ_AVAILABLE:
        raise ImportError("holoviews and bokeh required. Install with: pip install holoviews bokeh")
    
    hv.extension("bokeh")

    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    names = list(graph.nodes())

    # Create edges data
    edges = []
    num_nodes = len(adjacency_matrix)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = adjacency_matrix[i, j]
            if weight != 0:
                edges.append((names[i], names[j], weight))

    edges_df = pd.DataFrame(edges, columns=["source", "target", "value"])

    # Node colors based on degree
    degrees = dict(graph.degree(names, weight='weight'))
    norm = colors.Normalize(vmin=min(degrees.values()), vmax=max(degrees.values()))
    colormap = cm.ScalarMappable(norm=norm, cmap='Blues')
    color_df = pd.DataFrame({
        'index': names,
        'color': [colors.to_hex(colormap.to_rgba(degrees[name])) for name in names]
    })

    edges_df = edges_df.merge(color_df, left_on='source', right_on='index', how='inner')
    edges_df = edges_df.rename(columns={'color': 'edge_color'})

    # Filter weak edges
    avg_weight = edges_df['value'].mean() + 1.5 * edges_df['value'].std()
    edges_df.loc[edges_df['value'] < avg_weight, 'value'] = 0.0001

    chord = hv.Chord(edges_df[['source', 'target', 'value', 'edge_color']])
    chord.opts(
        labels='index',
        node_color='index',
        edge_color='edge_color',
        width=1000,
        height=1000
    )

    output_file(save_path)
    show(hv.render(chord, backend='bokeh'))
    print(f"Chord diagram saved to {save_path}")