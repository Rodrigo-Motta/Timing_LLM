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

# Function to remove stop words and lemmatize the words
def remove_stopwords_lemmatize(string_list):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return [' '.join(lemmatizer.lemmatize(word.lower()) for word in string.split() if word.lower() not in stop_words) for string in string_list]

def similaraties(data,  model_list, num_refs,scrambled=False):
    # Initialize arrays
    distances_array_joint_raw = np.zeros((len(model_list), num_refs, num_refs)) #(num_model, num_refs)
    distances_array_joint_raw[:] = np.nan

    # Loop over models
    for iModel, Model in enumerate(model_list):
        print(Model)
        model = SentenceTransformer(Model)

        # Precompute embeddings for dataset with joint sentences
        # for iRef, Ref in enumerate(data.list_names):
        #     print(data.list_names[iRef])
        if scrambled == True:
            ref_embeddings_joint_raw = [model.encode(data.scales_joint_raw_scrambled[Ref], convert_to_tensor=True) for
                                        Ref in data.list_names]

        else:
            ref_embeddings_joint_raw = [model.encode(data.scales_joint_raw[Ref], convert_to_tensor=True) for Ref in
                                            data.list_names]

        # Calculate distances for dataset with joint sentences
        for iRef in range(num_refs):
            list_1 = ref_embeddings_joint_raw[iRef]

            for iComp in range(iRef + 1, num_refs):
                Comp = data.list_names[iComp]
                list_2 = ref_embeddings_joint_raw[iComp]

                dists = util.pytorch_cos_sim(list_1, list_2)
                dists = util.dot_score(list_1, list_2)

                distances_array_joint_raw[iModel, iRef, iComp] = dists
    return distances_array_joint_raw

def PCA_embeddings(data,  model_list, n_comp):
    import numpy as np
    from sklearn.decomposition import PCA

    # Initialize arrays

    # Loop over models
    for iModel, Model in enumerate(model_list):
        print(Model)
        model = SentenceTransformer(Model)

        # Precompute embeddings for dataset with joint sentences
        # for iRef, Ref in enumerate(data.list_names):
        #     print(data.list_names[iRef])
        ref_embeddings_joint_raw = np.array([model.encode(data.scales_joint_raw[Ref], convert_to_tensor=False) for Ref in
                                        data.list_names])
        print(ref_embeddings_joint_raw.shape)
        pca = PCA(n_components=n_comp)
        pca.fit(ref_embeddings_joint_raw)
        PCA(n_components=2)
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)
        X_pca = pca.fit_transform(ref_embeddings_joint_raw)

    return X_pca

def TSNE_embeddings(data,  model_list, n_comp):
    import numpy as np
    from sklearn.manifold import TSNE

    # Initialize arrays

    # Loop over models
    for iModel, Model in enumerate(model_list):
        print(Model)
        model = SentenceTransformer(Model)

        # Precompute embeddings for dataset with joint sentences
        # for iRef, Ref in enumerate(data.list_names):
        #     print(data.list_names[iRef])
        ref_embeddings_joint_raw = np.array([model.encode(data.scales_joint_raw[Ref], convert_to_tensor=False) for Ref in
                                        data.list_names])

        X_embedded = TSNE(n_components=n_comp, learning_rate='auto',
                          init = 'random', perplexity = 15, n_iter=300).fit_transform(ref_embeddings_joint_raw)

    return X_embedded

def remove_triangle(df):
    # Remove triangle of a symmetric matrix and the diagonal

    df = df.astype(float)
    df.values[np.triu_indices_from(df, k=1)] = np.nan
    df = ((df.T).values.reshape((1, (df.shape[0]) ** 2)))
    df = df[~np.isnan(df)]
    df = df[df != 1]
    return (df).reshape((1, len(df)))


def plot_dendogram(df_Distances_joint_raw):
    # Find the row and column labels for the maximum value with stopwords
    max_joint_raw = df_Distances_joint_raw.stack().idxmax()
    max_row_label_joint_raw, max_col_label_joint_raw = max_joint_raw[0], max_joint_raw[1]
    print(f"The maximum similarity value with stopwords is located between {max_row_label_joint_raw} and {max_col_label_joint_raw}")

    # Find the row and column labels for the minimum value with stopwords
    min_joint_raw = df_Distances_joint_raw.stack().idxmin()
    min_row_label_joint_raw, min_col_label_joint_raw = min_joint_raw[0], min_joint_raw[1]
    print(f"The minimum similarity value with stopwords is located between {min_row_label_joint_raw} and {min_col_label_joint_raw}")

    df_Distances_joint_raw = df_Distances_joint_raw.fillna(0)

    Similarities = df_Distances_joint_raw.values
    Distances = 1-Similarities # converting similarity values into dissimilarity values.
    np.fill_diagonal(Distances, 0)
    Distances = squareform(Distances)

    Z_joint_raw = linkage(Distances, method = 'average') # the method can be single, complete or average
    # print(Z)

    c_joint_raw, coph_dists = cophenet(Z_joint_raw, Distances) # the cophenetic correlation coefficient measures how well the hierarchical clustering represented by the linkage matrix Z preserves the original distances in the distance matrix Distances.
    print(c_joint_raw)

    f, ax = plt.subplots(figsize = (15, 6)) # f, ax = plt.subplots(figsize = (3.2, 10))
    plt.ylabel('Distance', fontsize = 11, loc = 'center')
    dendrogram(
        Z_joint_raw,
        leaf_rotation = 90,  # rotates the x axis labels
        leaf_font_size = 8,  # font size for the x axis labels
        labels = df_Distances_joint_raw.columns,
        # distance_sort = 'descending',
        orientation = 'top',
        color_threshold = .35,
        above_threshold_color = '#7591a1', #bcbddc
        ax = ax
    )
    ax.set_yticks(np.arange(.001, 1.1, .25))
    ax.set_yticklabels([0, .25, .5, .75, 1], fontsize = 6)
    plt.tight_layout()
    plt.show()

def plot_heatmap(df_Distances_joint_raw):
    plt.figure(figsize=(8,8))
    sns.heatmap(df_Distances_joint_raw)
    plt.tight_layout()
    plt.show()

def plot_node_degree(df_Distances_joint_raw):
    plt.figure(figsize=(15,5))
    sns.barplot(df_Distances_joint_raw.mean(axis=0).sort_values())
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()

def plot_PCA(df):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=8).fit(df.iloc[:,[0,1]])
    df['cluster'] = pd.Categorical(kmeans.labels_)

    plt.figure(figsize=(8,8))

    p1 = sns.scatterplot(x=0,  # Horizontal axis
                         y=1,  # Vertical axis
                         data=df,  # Data source
                         s=100,
                         legend=False,
                         hue="cluster")

    for line in range(0, df.shape[0]):
        p1.text(df.iloc[line, 0] + 0.01, df.iloc[line, 1],
                df.Q[line], horizontalalignment='left',
                size='small', color='black', weight='normal')

    plt.title('Embedding')
    # Set x-axis label
    plt.xlabel('PC_0')
    # Set y-axis label
    plt.ylabel('PC_1')
    plt.xlim(-.6,.6)
    plt.ylim(-.6,.6)
    plt.tight_layout()
    plt.show()

def plot_3D_PCA(df):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=5).fit(df.iloc[:, [0, 1, 2]])
    df['cluster'] = pd.Categorical(kmeans.labels_)

    # Creating a 3D scatter plot
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2],
                         s=100, c=df['cluster'], cmap='viridis')

    # Adding annotations to each point
    for line in range(0, df.shape[0]):
        ax.text(df.iloc[line, 0], df.iloc[line, 1], df.iloc[line, 2], df['Q'][line],
                horizontalalignment='left', size='small', color='black', weight='normal')

    # Titles and labels
    plt.title('3D Embedding')
    ax.set_xlabel('PC_0')
    ax.set_ylabel('PC_1')
    ax.set_zlabel('PC_2')
    ax.set_xlim(-.3,.3)
    ax.set_ylim(-.3,.3)
    ax.set_zlim(-.3,.3)

    plt.tight_layout()
    plt.show()

def plot_3D_PCA_controls(df):
    # Define the points that you want to highlight
    highlight_points = ['DASS', 'RESS']
    df['color'] = df.index.where(df['Q'].isin(highlight_points), 'Other')

    # Creating a 3D scatter plot
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all non-highlighted points using a single color
    other_points = df[df['color'] == 'Other']
    ax.scatter(other_points.iloc[:, 0], other_points.iloc[:, 1], other_points.iloc[:, 2],
               s=100, c='grey', label='Other', alpha=0.6)

    # Highlight the specific points (DASS and RESS)
    highlighted_points = df[df['color'] != 'Other']
    ax.scatter(highlighted_points.iloc[:, 0], highlighted_points.iloc[:, 1], highlighted_points.iloc[:, 2],
               s=150, c='red', label='Controls', alpha=0.8)

    df = df[(df.Q == 'RESS') | (df.Q == 'DASS')].reset_index().drop(columns='index')
    for line in range(0, df.shape[0]):
        ax.text(df.iloc[line, 0] + 0.05, df.iloc[line, 1], df.iloc[line, 2], df['Q'][line],
        horizontalalignment = 'left', size = 'small', color = 'black', weight = 'bold')

    # Titles and labels
    plt.title('3D PCA Embedding')
    ax.set_xlabel('PC_0')
    ax.set_ylabel('PC_1')
    ax.set_zlabel('PC_2')
    #ax.set_xlim(3*-0.3, 3*0.3)
    #ax.set_ylim(3*-0.3, 3*0.3)
    #ax.set_zlim(3*-0.3, 3*0.3)

    # Legend
    ax.legend()
    plt.tight_layout()
    plt.show()

def create_network(data, df_Distances_joint_raw):
    names = data.list_names
    adjacency_matrix = df_Distances_joint_raw.values
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

    # Convert to DataFrame for manipulation and normalization
    edges_df = pd.DataFrame(edges, columns=["source", "target", "value"])

    # Calculate node degrees (number of links per node)
    degrees = dict(graph.degree(names, weight='weight'))

    # Normalize degrees and create a colormap using 'Blues'
    norm = colors.Normalize(vmin=min(degrees.values()), vmax=max(degrees.values()))
    colormap = cm.ScalarMappable(norm=norm, cmap='Blues')

    # Apply color mapping to each node
    color_df = pd.DataFrame({'index': names, 'color': [colormap.to_rgba(degrees[name]) for name in names]})

    # Convert RGBA colors to hex values
    color_df['color'] = color_df['color'].apply(lambda rgba: colors.to_hex(rgba))

    edges_df['color'] = edges_df.merge(color_df, how='inner', left_on='source', right_on='index')['color']

    cmap = colors.ListedColormap(color_df['color'].values)

    # Calculate the average weight
    average_weight = edges_df['value'].mean() + 1.5 * edges_df['value'].std()

    # Filter edges above the average weight
    edges_df.loc[edges_df['value'] < average_weight, 'value'] = 0.0001

    min_weight = edges_df['value'].min()
    max_weight = edges_df['value'].max()
    edges_df['line_width'] = (edges_df['value'] - min_weight) / (max_weight - min_weight) * 10

    # Combine nodes and filtered edges
    chord = hv.Chord(edges_df)
    chord.opts(
        labels='index',
        node_color='index',
        edge_color='color',
        node_cmap=cmap,
        edge_cmap=cmap,
        edge_line_width=hv.dim('value'),  # line_width
        width=1000, height=1000
    )

    # chord = hv.Chord(edges_df)
    # chord.opts(
    #     labels='index',
    #     node_color='index',
    #     edge_color=hv.dim('source').str(),
    #     node_cmap='Category20',
    #     edge_cmap=cmap,
    #     edge_line_width=hv.dim('value'),  # line_width
    #     width=1000, height=1000
    # )

    # Display the chord diagram
    output_file("chord_diagram.html")
    show(hv.render(chord, backend='bokeh'))

def invert_weights(graph, max_weight=None):
    new_graph = nx.Graph()

    if max_weight is None:
        max_weight = max(data['weight'] for u, v, data in graph.edges(data=True))

    for u, v, data in graph.edges(data=True):
        # Inverting weights so higher weight becomes less costly
        new_weight = max_weight - data['weight']
        new_graph.add_edge(u, v, weight=new_weight)

    return new_graph

def shortest_path(graph):
    # Calculate all pairs' shortest path distances using weights
    shortest_paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))

    # Calculate the weighted average path length
    if nx.is_connected(graph):
        avg_path_length = nx.average_shortest_path_length(graph, weight='weight')
    else:
        avg_path_length = "Graph is not connected, cannot calculate average path length."

    # Calculate eccentricity
    eccentricity = nx.eccentricity(graph, sp=shortest_paths)

    # Calculate diameter
    diameter = nx.diameter(graph, e=eccentricity)

    print("Shortest Paths (all pairs):", shortest_paths)
    print("Average Path Length (weighted):", avg_path_length)
    print("Eccentricity (weighted):", eccentricity)
    print("Diameter (weighted):", diameter)
    graph_prop = {
    "Shortest Paths (all pairs)" : shortest_paths,
    "Average Path Length (weighted)" : avg_path_length,
    "Eccentricity (weighted)" : eccentricity,
    "Diameter (weighted)" : diameter,
    }
    return graph_prop

def graph_properties(graph):
    # Assume the graph is already loaded or created
    # `graph` is a NetworkX graph where the edge weights represent semantic similarity

    # Weighted Degree (Strength)
    weighted_degrees = dict(graph.degree(weight='weight'))

    # Weighted Clustering Coefficient
    weighted_clustering_coefficients = nx.clustering(graph, weight='weight')

    # Weighted Betweenness Centrality
    weighted_betweenness = nx.betweenness_centrality(graph, weight='weight')

    # Weighted Closeness Centrality
    weighted_closeness = nx.closeness_centrality(graph, distance='weight')

    # Weighted Modularity (Community Detection via Louvain method)
    partition = community_louvain.best_partition(graph, weight='weight')

    # Weighted Eigenvector Centrality
    weighted_eigenvector = nx.eigenvector_centrality(graph, weight='weight')

    # Combine individual properties into a DataFrame
    df = pd.DataFrame({
        'Weighted Degree': weighted_degrees,
        'Weighted Clustering Coefficient': weighted_clustering_coefficients,
        'Weighted Betweenness Centrality': weighted_betweenness,
        'Weighted Closeness Centrality': weighted_closeness,
        'Community (Partition)': partition,
        'Weighted Eigenvector Centrality': weighted_eigenvector
    })

    # Adding global metrics that are not node-specific
    total_weight = sum(data['weight'] for u, v, data in graph.edges(data=True))
    num_possible_edges = len(graph) * (len(graph) - 1) / 2
    weighted_density = total_weight / num_possible_edges
    weight_correlations = nx.degree_pearson_correlation_coefficient(graph, weight='weight')

    global_metrics = {
        'Average Weighted Degree': sum(weighted_degrees.values()) / len(graph),
        'Weighted Density': weighted_density,
        'Assortativity (Weight Correlation)': weight_correlations
    }

    # Display the DataFrame and global metrics
    print("Node-Specific Properties DataFrame:")
    print(df)

    print("\nGlobal Metrics:")
    for name, value in global_metrics.items():
        print(f"{name}: {value}")

    return df, global_metrics

def plot_dendrogram_and_heatmap(df_Distances_joint_raw):
    from scipy.cluster.hierarchy import linkage, fcluster

    df_Distances_joint_raw = df_Distances_joint_raw.fillna(0)

    Similarities = df_Distances_joint_raw.values
    Distances = 1-Similarities # converting similarity values into dissimilarity values.
    np.fill_diagonal(Distances, 0)
    Distances = squareform(Distances)

    Z = linkage(Distances, method = 'average') # the method can be single, complete or average

    # print(Z)
    # Assign clusters using the flat cluster method
    clusters = fcluster(Z, 3, criterion='maxclust')

    # Create a mapping from clusters to colors
    unique_clusters = np.unique(clusters)
    colors = sns.color_palette("Set2", len(unique_clusters))
    lut = dict(zip(unique_clusters, colors))
    row_colors = (clusters).map(lut)

    # Create a cluster map with seaborn
    g = sns.clustermap(
        df_Distances_joint_raw,
        metric='euclidean',  # You can also use 'correlation' or other distance metrics
        method='average',  # Clustering method: 'single', 'complete', 'average', etc.
        cmap='coolwarm',  # Color scheme for the heatmap
        figsize=(12, 8),
        annot=False,  # Set to True to display the values in the cells
        cbar_kws={'label': 'Similarity Score'},  # Customize the color bar
        xticklabels=True,
        yticklabels=True,
        col_cluster=False,
        row_colors=row_colors,
        dendrogram_ratio=(.1, .2)
    )

    # Customize the plot with title, etc.
    g.fig.suptitle('Aligned Dendrogram and Heatmap', fontsize=16)
    g.ax_heatmap.set_xlabel('Questionnaires', fontsize=12)
    g.ax_heatmap.set_ylabel('Questionnaires', fontsize=12)

    # Show the plot
    plt.show()
