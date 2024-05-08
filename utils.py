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

# Function to remove stop words and lemmatize the words
def remove_stopwords_lemmatize(string_list):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return [' '.join(lemmatizer.lemmatize(word.lower()) for word in string.split() if word.lower() not in stop_words) for string in string_list]


def similaraties(data,  model_list, num_refs):
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

        ref_embeddings_joint_raw = [model.encode(data.scales_joint_raw[Ref], convert_to_tensor=True) for Ref in
                                        data.list_names]

        # ref_embeddings_joint_raw = [model.encode(data.scales_joint_raw_scrambled[Ref], convert_to_tensor=True) for Ref in
        #                                 data.list_names]

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