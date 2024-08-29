from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import cophenet # cophenet is used to calculate the cophenetic correlation coefficient, which measures how well a hierarchical clustering preserves the original pairwise distances between data points.
from scipy.spatial.distance import squareform # squareform is used to convert a condensed distance matrix into a square distance matrix, which is a common format for hierarchical clustering algorithms.
from scipy.cluster.hierarchy import dendrogram, linkage
import nltk, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sentence_transformers import SentenceTransformer, util
import networkx as nx
import holoviews as hv
import pandas as pd
from bokeh.io import output_file, show
import matplotlib.colors as colors
import matplotlib.cm as cm
import community as community_louvain
from sklearn.cluster import KMeans

def remove_stopwords_lemmatize(string_list):
    """
    Removes stopwords and lemmatizes words in a list of strings.

    Parameters
    ----------
    string_list : list of str
        List of strings to be processed.

    Returns
    -------
    list of str
        List of processed strings with stopwords removed and words lemmatized.
    """
    # Initialize the set of stop words and the lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Process each string in the list
    return [' '.join(
        lemmatizer.lemmatize(word.lower())
        for word in string.split()
        if word.lower() not in stop_words
    )
        for string in string_list]


def intra_similarities(data, model_list, num_refs, num_scrambles):
    """
    Calculates intra-similarities for scrambled and joint scales using multiple models.

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
        Array of cosine similarities for the dataset with joint sentences.
    """
    # Initialize array for storing similarities
    similarities_array = np.zeros((len(model_list), num_refs, num_scrambles, num_scrambles))
    similarities_array[:] = np.nan

    # Loop over each model
    for iModel, Model in enumerate(model_list):
        print(Model)
        model = SentenceTransformer(Model)

        # Loop over each reference
        for iRef, Ref in enumerate(data.list_names):
            embed_list = []

            # Scramble and encode data multiple times
            for scrambles in range(num_scrambles):
                data.scramble_joint()
                embeddings = [model.encode(data.scales_joint_raw_scrambled[Ref], convert_to_tensor=True)]
                embed_list.append(embeddings)

            # Calculate similarities between scrambled data
            for i in range(num_scrambles):
                list_1 = embed_list[i][0]

                for j in range(i, num_scrambles):
                    list_2 = embed_list[j][0]
                    dists = util.pytorch_cos_sim(list_1, list_2)
                    similarities_array[iModel, iRef, i, j] = dists

    return similarities_array


def similarities(data, model_list, num_refs, scrambled=False):
    """
    Calculates similarities for joint scales using multiple models.

    Parameters
    ----------
    data : DatasetLoader
        DatasetLoader object containing the data.
    model_list : list of str
        List of model names to be used for embeddings.
    num_refs : int
        Number of references.
    scrambled : bool, optional
        Whether to use scrambled joint raw scales (default is False).

    Returns
    -------
    np.ndarray
        Array of similarities for the dataset with joint sentences.
    """
    # Initialize array for storing distances
    similarities_array = np.zeros((len(model_list), num_refs, num_refs))
    similarities_array[:] = np.nan

    # Loop over each model
    for iModel, Model in enumerate(model_list):
        print(Model)
        model = SentenceTransformer(Model)

        # Encode data with or without scrambling
        if scrambled:
            ref_embeddings_joint_raw = [model.encode(data.scales_joint_raw_scrambled[Ref], convert_to_tensor=True) for
                                        Ref in data.list_names]
        else:
            ref_embeddings_joint_raw = [model.encode(data.scales_joint_raw[Ref], convert_to_tensor=True) for Ref in
                                        data.list_names]

        # Calculate similarities between encoded data
        for iRef in range(num_refs):
            list_1 = ref_embeddings_joint_raw[iRef]

            for iComp in range(iRef + 1, num_refs):
                list_2 = ref_embeddings_joint_raw[iComp]

                dists = util.pytorch_cos_sim(list_1, list_2)
                similarities_array[iModel, iRef, iComp] = dists

    return similarities_array


def similarities_average(data, model_list, num_refs, num_scrambles):
    """
    Calculates average similarities for scrambled joint scales using multiple models.

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
    tuple of np.ndarray
        Average and standard deviation of similarities for the dataset with joint sentences.
    """
    N = num_scrambles
    similarities_array = np.zeros((len(model_list), num_refs, num_refs))
    similarities_array[:] = np.nan

    # Loop over each model
    for iModel, Model in enumerate(model_list):
        print(Model)
        model = SentenceTransformer(Model)
        all_similarities = np.zeros((len(model_list), N, num_refs, num_refs))

        # Perform multiple scrambles and calculate distances
        for n in range(N):
            data.scramble_joint()
            ref_embeddings_joint_raw = [model.encode(data.scales_joint_raw_scrambled[Ref], convert_to_tensor=True) for
                                        Ref in data.list_names]

            for iRef in range(num_refs):
                list_1 = ref_embeddings_joint_raw[iRef]

                for iComp in range(iRef + 1, num_refs):
                    list_2 = ref_embeddings_joint_raw[iComp]

                    similarities = util.pytorch_cos_sim(list_1, list_2)
                    similarities_array[iModel, iRef, iComp] = similarities

            all_similarities[iModel, n, :, :] = similarities_array[0, :, :]

    average_similarities = all_similarities.mean(axis=1)
    std_similarities = all_similarities.std(axis=1)

    return average_similarities, std_similarities


def get_embedding(data, model_list, num_refs, num_scrambles):
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


def hierarquical_clustering(embed_arr, data):
    """
    Performs hierarchical clustering on embeddings and visualizes the dendrogram.

    Parameters
    ----------
    embed_arr : np.ndarray
        Array of embeddings.
    data : DatasetLoader
        DatasetLoader object containing the data.

    Returns
    -------
    np.ndarray
        Linkage matrix.
    """
    # Perform hierarchical clustering
    linked = linkage(embed_arr, method='average', metric='cosine')

    # Visualize the dendrogram
    plt.figure(figsize=(12, 6))
    dendrogram(linked,
               orientation='top',
               labels=data.list_names,
               distance_sort='descending',
               show_leaf_counts=True)
    plt.xlabel('Concepts')
    plt.ylabel('Cosine distances')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.tight_layout()
    plt.show()

    return linked


def clusters(embed_arr, data, clusters):
    """
    Performs K-means clustering on embeddings.

    Parameters
    ----------
    embed_arr : np.ndarray
        Array of embeddings.
    data : DatasetLoader
        DatasetLoader object containing the data.
    clusters : int
        Number of clusters.

    Returns
    -------
    pd.DataFrame
        DataFrame containing embeddings, names, and cluster labels.
    """
    df = pd.DataFrame(np.array(embed_arr))
    df['Names'] = data.list_names

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=clusters, init='k-means++', max_iter=300, n_init=10).fit(df.iloc[:, :-1])
    df['cluster'] = pd.Categorical(kmeans.labels_)

    return df


def convert_arr_to_pandas(similarity_array, list_names):
    """
    Converts a distance array to a symmetric pandas DataFrame.
    
    Parameters:
    similarity_array (np.ndarray): A 2D array of similarities.
    list_names (list): A list of names to use for columns and index.

    Returns:
    pd.DataFrame: Symmetric DataFrame of similarities.
    """
    # Ensure list_names is provided and is the right length
    if len(list_names) != similarity_array.shape[1]:
        raise ValueError("The length of list_names must match the dimensions of the similarity array.")

    # Create DataFrame and make it symmetric
    df = pd.DataFrame(data=np.nanmean(similarity_array, axis=0), columns=list_names,
                      index=list_names).replace(np.nan, 0)
    df = (df + df.T).replace(0.0, 1.0)

    return df


def min_max_norm(df):
    """
    Applies min-max normalization to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be normalized.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame.
    """
    # Apply min-max normalization
    df = ((df - df.min().min()) / (df.max().max() - df.min().min()))

    return df

def standard_scaler(df):
    """
    Applies standard scaling (z-score normalization) to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be standardized.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with mean 0 and standard deviation 1.
    """
    # Apply standard scaling
    df = (df - df.mean()) / df.std()

    return df


def PCA_embeddings(arr, n_comp):
    """
    Applies PCA to reduce the dimensionality of embeddings.

    Parameters
    ----------
    arr : np.ndarray
        Array of embeddings.
    n_comp : int
        Number of principal components.

    Returns
    -------
    tuple of np.ndarray
        Transformed embeddings and explained variance ratio.
    """
    from sklearn.decomposition import PCA

    # Perform PCA
    pca = PCA(n_components=n_comp)
    pca.fit(arr)
    X_pca = pca.fit_transform(arr)

    return X_pca, pca.explained_variance_ratio_


def TSNE_embeddings(data, model_list, n_comp):
    """
    Applies t-SNE to reduce the dimensionality of embeddings.

    Parameters
    ----------
    data : DatasetLoader
        DatasetLoader object containing the data.
    model_list : list of str
        List of model names to be used for embeddings.
    n_comp : int
        Number of components for t-SNE.

    Returns
    -------
    np.ndarray
        Transformed embeddings.
    """
    from sklearn.manifold import TSNE

    # Loop over each model
    for iModel, Model in enumerate(model_list):
        print(Model)
        model = SentenceTransformer(Model)

        # Encode data
        ref_embeddings_joint_raw = np.array(
            [model.encode(data.scales_joint_raw[Ref], convert_to_tensor=False) for Ref in data.list_names])

        # Perform t-SNE
        X_embedded = TSNE(n_components=n_comp, learning_rate='auto', init='random', perplexity=15,
                          n_iter=300).fit_transform(ref_embeddings_joint_raw)

    return X_embedded


def remove_triangle(df):
    """
    Removes the upper triangle and diagonal of a symmetric matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Symmetric DataFrame.

    Returns
    -------
    np.ndarray
        Array with upper triangle and diagonal removed.
    """
    # Remove upper triangle and diagonal
    df = df.astype(float)
    df.values[np.triu_indices_from(df, k=1)] = np.nan
    df = ((df.T).values.reshape((1, (df.shape[0]) ** 2)))
    df = df[~np.isnan(df)]
    df = df[df != 1]

    return df.reshape((1, len(df)))


def summarize(data, model_list, num_refs):
    """
    Summarizes the joint raw scales in one word using a pre-trained model.

    Parameters
    ----------
    data : DatasetLoader
        DatasetLoader object containing the data.
    model_list : list of str
        List of model names to be used for embeddings.
    num_refs : int
        Number of references.

    Returns
    -------
    None
    """
    from transformers import AutoTokenizer, AutoModelWithLMHead

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('T5-base')
    model = AutoModelWithLMHead.from_pretrained('T5-base', return_dict=True)

    # Loop over each reference
    for iRef, Ref in enumerate(data.list_names):
        print(Ref)
        text = (data.scales_joint_raw[Ref])
        text = text.replace("?", '"').replace("?", "'")
        text = text.strip().replace("\n", " ")

        if not text.endswith("."):
            text = text + "."

        # Prepare text for summarization
        t5_prepared_Text = "summarize in one word: " + text
        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt")
        summary_ids = model.generate(tokenized_text, num_beams=4, no_repeat_ngram_size=3, min_length=1, max_length=3,
                                     length_penalty=2.0, temperature=0.8)
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(output)


############### PLOTS #####################
def plot_intra_barplot(arr, data):
    """
    Plots a bar plot of the intra-similarity scores with error bars.

    Parameters
    ----------
    arr : np.ndarray
        Array of similarities.
    data : DatasetLoader
        DatasetLoader object containing the data.

    Returns
    -------
    None
    """
    # Calculate the mean and standard deviation of the distances
    mean = np.nanmean(np.nanmean(arr[0], axis=-1), axis=-1)
    error = np.nanstd(np.nanstd(arr[0], axis=-1), axis=-1)

    # Plot the bar plot with error bars
    plt.bar(x=data.list_names, height=mean, yerr=error)
    plt.xticks(ticks=range(len(data.list_names)), labels=data.list_names, rotation=90)
    plt.tight_layout()
    plt.ylabel('Cosine Similarity')
    plt.show()


def plot_dendrogram(df_Similarities, threshold=0.35, x_fontsize=10):
    """
    Plots a dendrogram based on the normalized similarity.

    Parameters
    ----------
    df_Similarities : pd.DataFrame
        DataFrame containing the normalized similarity matrix.
    threshold : float, optional
        The threshold to apply when forming flat clusters (default is 0.35).
    x_fontsize : int, optional
        Font size for x-axis labels (default is 10).
    """
    # Convert similarities to dissimilarities
    Similarities = df_Similarities.values
    Distances = 1 - Similarities
    np.fill_diagonal(Distances, 0)
    Distances = squareform(Distances)

    # Perform hierarchical clustering
    Z = linkage(Distances, method='average')

    # Calculate the cophenetic correlation coefficient
    c, coph_dists = cophenet(Z, Distances)
    # print(f"Cophenetic correlation coefficient: {c}")

    # Plot the dendrogram
    f, ax = plt.subplots(figsize=(15, 6))
    plt.ylabel('Distance', fontsize=12, loc='center')
    dendrogram(
        Z,
        leaf_rotation=45,
        leaf_font_size=10,
        labels=df_Similarities.columns,
        orientation='top',
        color_threshold=threshold,
        above_threshold_color='#55a1ab',
        ax=ax
    )
    
    # Set font size for x-axis labels
    plt.tick_params(axis='x', labelsize=x_fontsize)
    
    ax.set_yticks(np.arange(0.001, 1.1, 0.25))
    ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1], fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_heatmap(df):
    """
    Plots a heatmap of the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 8))
    sns.heatmap(df, annot=False)
    plt.tight_layout()
    plt.show()


def plot_node_degree(df):
    """
    Plots a bar plot of the average node degree.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the node degrees.

    Returns
    -------
    None
    """
    plt.figure(figsize=(15, 5))
    sns.barplot(df.mean(axis=0).sort_values())
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()


def plot_PCA_embeddings(arr, list_names, clusters):
    """
    Plots a 2D PCA embedding scatter plot with clusters.

    Parameters
    ----------
    arr : np.ndarray
        Array of PCA embeddings.
    list_names : list of str
        List of names corresponding to the embeddings.
    clusters : list of int
        List of cluster labels.

    Returns
    -------
    None
    """
    df = pd.DataFrame(arr)
    df['Names'] = list_names
    df['cluster'] = clusters

    plt.figure(figsize=(8, 8))
    p1 = sns.scatterplot(x=0, y=1, data=df, s=100, legend=False, hue="cluster")

    # Add text labels to the plot
    for line in range(0, df.shape[0]):
        p1.text(df.iloc[line, 0] + 0.01, df.iloc[line, 1], df.Names[line], horizontalalignment='left', size='small',
                color='black', weight='normal')

    plt.title('Embedding')
    plt.xlabel('PC_0')
    plt.ylabel('PC_1')
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.6, 0.6)
    plt.tight_layout()
    plt.show()


def plot_3D_PCA(arr, list_names, clusters):
    """
    Plots a 3D PCA embedding scatter plot with clusters.

    Parameters
    ----------
    arr : np.ndarray
        Array of PCA embeddings.
    list_names : list of str
        List of names corresponding to the embeddings.
    clusters : list of int
        List of cluster labels.

    Returns
    -------
    None
    """
    df = pd.DataFrame(arr)
    df['Names'] = list_names
    df['cluster'] = clusters

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], s=100, c=df['cluster'], cmap='viridis')

    # Add text labels to the plot
    for line in range(0, df.shape[0]):
        ax.text(df.iloc[line, 0], df.iloc[line, 1], df.iloc[line, 2], df['Names'][line], horizontalalignment='left',
                size='small', color='black', weight='normal')

    plt.title('3D Embedding')
    ax.set_xlabel('PC_0')
    ax.set_ylabel('PC_1')
    ax.set_zlabel('PC_2')
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)
    plt.tight_layout()
    plt.show()


def PCA_Kmeans_clusters(arr, list_names, clusters):
    """
    Performs K-means clustering on PCA embeddings.

    Parameters
    ----------
    arr : np.ndarray
        Array of PCA embeddings.
    list_names : list of str
        List of names corresponding to the embeddings.
    clusters : int
        Number of clusters.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the PCA embeddings, names, and cluster labels.
    """
    df = pd.DataFrame(arr)
    df['Names'] = list_names

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=clusters).fit(df.iloc[:, :-1])
    df['cluster'] = pd.Categorical(kmeans.labels_)

    return df


def plot_3D_PCA_controls(arr, list_names):
    """
    Plots a 3D PCA embedding scatter plot with highlighted controls.

    Parameters
    ----------
    arr : np.ndarray
        Array of PCA embeddings.
    list_names : list of str
        List of names corresponding to the embeddings.

    Returns
    -------
    None
    """
    df = pd.DataFrame(arr)
    df['Names'] = list_names

    # Define the points that you want to highlight
    highlight_points = ['DASS', 'RESS']
    df['color'] = df.index.where(df['Names'].isin(highlight_points), 'Other')

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all non-highlighted points using a single color
    other_points = df[df['color'] == 'Other']
    ax.scatter(other_points.iloc[:, 0], other_points.iloc[:, 1], other_points.iloc[:, 2], s=100, c='grey',
               label='Other', alpha=0.6)

    # Highlight the specific points (DASS and RESS)
    highlighted_points = df[df['color'] != 'Other']
    ax.scatter(highlighted_points.iloc[:, 0], highlighted_points.iloc[:, 1], highlighted_points.iloc[:, 2], s=150,
               c='red', label='Controls', alpha=0.8)

    # Add text labels to the plot
    for line in range(0, df.shape[0]):
        ax.text(df.iloc[line, 0] + 0.05, df.iloc[line, 1], df.iloc[line, 2], df['Names'][line],
                horizontalalignment='left', size='small', color='black', weight='bold')

    plt.title('3D PCA Embedding')
    ax.set_xlabel('PC_0')
    ax.set_ylabel('PC_1')
    ax.set_zlabel('PC_2')
    plt.tight_layout()
    plt.show()


def create_network(data, df_Similarities):
    """
    Creates a network graph from the similarity matrix.

    Parameters
    ----------
    data : DatasetLoader
        DatasetLoader object containing the data.
    df_Similarities : pd.DataFrame
        DataFrame containing the similarity matrix.

    Returns
    -------
    networkx.Graph
        Network graph.
    """
    names = data.list_names
    adjacency_matrix = df_Similarities.values

    # Initialize an empty graph
    graph = nx.Graph()

    # Add edges with weights from the adjacency matrix
    num_nodes = len(adjacency_matrix)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i][j] != 0:
                graph.add_edge(names[i], names[j], weight=adjacency_matrix[i][j])

    return graph


def plot_chord(graph):
    """
    Plots a chord diagram for the network graph.

    Parameters
    ----------
    graph : networkx.Graph
        Network graph.

    Returns
    -------
    None
    """
    hv.extension("bokeh")

    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    names = list(graph.nodes())

    # Create a data structure suitable for a chord diagram
    edges = []
    num_nodes = len(adjacency_matrix)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = adjacency_matrix[i][j]
            if weight != 0:
                edges.append((names[i], names[j], weight))

    edges_df = pd.DataFrame(edges, columns=["source", "target", "value"])

    # Calculate node degrees
    degrees = dict(graph.degree(names, weight='weight'))

    # Normalize degrees and create a colormap using 'Blues'
    norm = colors.Normalize(vmin=min(degrees.values()), vmax=max(degrees.values()))
    colormap = cm.ScalarMappable(norm=norm, cmap='Blues')
    color_df = pd.DataFrame({'index': names, 'color': [colormap.to_rgba(degrees[name]) for name in names]})
    color_df['color'] = color_df['color'].apply(lambda rgba: colors.to_hex(rgba))

    edges_df['color'] = edges_df.merge(color_df, how='inner', left_on='source', right_on='index')['color']
    cmap = colors.ListedColormap(color_df['color'].values)

    # Calculate the average weight
    average_weight = edges_df['value'].mean() + 1.5 * edges_df['value'].std()
    edges_df.loc[edges_df['value'] < average_weight, 'value'] = 0.0001

    min_weight = edges_df['value'].min()
    max_weight = edges_df['value'].max()
    edges_df['line_width'] = (edges_df['value'] - min_weight) / (max_weight - min_weight) * 10

    chord = hv.Chord(edges_df)
    chord.opts(
        labels='index',
        node_color='index',
        edge_color='color',
        node_cmap=cmap,
        edge_cmap=cmap,
        edge_line_width=hv.dim('value'),
        width=1000, height=1000
    )

    output_file("chord_diagram.html")
    show(hv.render(chord, backend='bokeh'))


def invert_weights(graph, max_weight=None):
    """
    Inverts the weights of the edges in the graph.

    Parameters
    ----------
    graph : networkx.Graph
        Network graph.
    max_weight : float, optional
        Maximum weight for inversion (default is None).

    Returns
    -------
    networkx.Graph
        Graph with inverted weights.
    """
    new_graph = nx.Graph()

    if max_weight is None:
        max_weight = max(data['weight'] for u, v, data in graph.edges(data=True))

    for u, v, data in graph.edges(data=True):
        new_weight = max_weight - data['weight']
        new_graph.add_edge(u, v, weight=new_weight)

    return new_graph


def shortest_path(graph):
    """
    Calculates shortest path properties of the graph.

    Parameters
    ----------
    graph : networkx.Graph
        Network graph.

    Returns
    -------
    dict
        Dictionary containing shortest path properties.
    """
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))

    if nx.is_connected(graph):
        avg_path_length = nx.average_shortest_path_length(graph, weight='weight')
    else:
        avg_path_length = "Graph is not connected, cannot calculate average path length."

    eccentricity = nx.eccentricity(graph, sp=shortest_paths)
    diameter = nx.diameter(graph, e=eccentricity)

    print("Shortest Paths (all pairs):", shortest_paths)
    print("Average Path Length (weighted):", avg_path_length)
    print("Eccentricity (weighted):", eccentricity)
    print("Diameter (weighted):", diameter)

    graph_prop = {
        "Shortest Paths (all pairs)": shortest_paths,
        "Average Path Length (weighted)": avg_path_length,
        "Eccentricity (weighted)": eccentricity,
        "Diameter (weighted)": diameter,
    }
    return graph_prop


def graph_properties(graph):
    """
    Calculates various properties of the network graph.

    Parameters
    ----------
    graph : networkx.Graph
        Network graph.

    Returns
    -------
    tuple
        DataFrame containing node-specific properties and dictionary of global metrics.
    """
    weighted_degrees = dict(graph.degree(weight='weight'))
    weighted_clustering_coefficients = nx.clustering(graph, weight='weight')
    weighted_betweenness = nx.betweenness_centrality(graph, weight='weight')
    weighted_closeness = nx.closeness_centrality(graph, distance='weight')
    partition = community_louvain.best_partition(graph, weight='weight')
    weighted_eigenvector = nx.eigenvector_centrality(graph, weight='weight')

    df = pd.DataFrame({
        'Weighted Degree': weighted_degrees,
        'Weighted Clustering Coefficient': weighted_clustering_coefficients,
        'Weighted Betweenness Centrality': weighted_betweenness,
        'Weighted Closeness Centrality': weighted_closeness,
        'Community (Partition)': partition,
        'Weighted Eigenvector Centrality': weighted_eigenvector
    })

    total_weight = sum(data['weight'] for u, v, data in graph.edges(data=True))
    num_possible_edges = len(graph) * (len(graph) - 1) / 2
    weighted_density = total_weight / num_possible_edges
    weight_correlations = nx.degree_pearson_correlation_coefficient(graph, weight='weight')

    global_metrics = {
        'Average Weighted Degree': sum(weighted_degrees.values()) / len(graph),
        'Weighted Density': weighted_density,
        'Assortativity (Weight Correlation)': weight_correlations
    }

    print("Node-Specific Properties DataFrame:")
    print(df)

    print("\nGlobal Metrics:")
    for name, value in global_metrics.items():
        print(f"{name}: {value}")

    return df, global_metrics


def plot_dendrogram_and_heatmap(df_Similarities):
    """
    Plots an aligned dendrogram and heatmap based on the distance matrix.

    Parameters
    ----------
    df_Similarities : pd.DataFrame
        DataFrame containing the similarity matrix.

    Returns
    -------
    None
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    df_Similarities = df_Similarities.fillna(0)

    Similarities = df_Similarities.values
    Distances = 1 - Similarities
    np.fill_diagonal(Distances, 0)
    Distances = squareform(Distances)

    Z = linkage(Distances, method='average')

    clusters = fcluster(Z, 3, criterion='maxclust')

    unique_clusters = np.unique(clusters)
    colors = sns.color_palette("Set2", len(unique_clusters))
    lut = dict(zip(unique_clusters, colors))
    row_colors = (clusters).map(lut)

    g = sns.clustermap(
        df_Similarities,
        metric='euclidean',
        method='average',
        cmap='coolwarm',
        figsize=(12, 8),
        annot=False,
        cbar_kws={'label': 'Similarity Score'},
        xticklabels=True,
        yticklabels=True,
        col_cluster=False,
        row_colors=row_colors,
        dendrogram_ratio=(.1, .2)
    )

    g.fig.suptitle('Aligned Dendrogram and Heatmap', fontsize=16)
    g.ax_heatmap.set_xlabel('Questionnaires', fontsize=12)
    g.ax_heatmap.set_ylabel('Questionnaires', fontsize=12)
    plt.show()