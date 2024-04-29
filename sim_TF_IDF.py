from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import cophenet # cophenet is used to calculate the cophenetic correlation coefficient, which measures how well a hierarchical clustering preserves the original pairwise distances between data points.
from scipy.spatial.distance import squareform # squareform is used to convert a condensed distance matrix into a square distance matrix, which is a common format for hierarchical clustering algorithms.
from scipy.cluster.hierarchy import dendrogram, linkage
import nltk, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from dataset import DatasetLoader

# nltk.download('wordnet')
# nltk.download('stopwords')

data = DatasetLoader()

# Process all variables in the dictionary
scales_preprocessed_joint = {variable: " ".join(string_list).replace('.', '').replace('?', '') for variable, string_list in data.scales_preprocessed.items()}

##
# Count Vectorizer give number of frequency with respect to index of
# vocabulary whereas tf-idf consider overall documents of weight of
# words.
##

# Create Count and TF-IDF vectorizers
countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english')
vectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english') # TfidfVectorizer convert a collection of raw documents to a matrix of TF-IDF features.

# Compute the TF-IDF vectors for each statement
count_matrix = countvectorizer.fit_transform(scales_preprocessed_joint.values())
tfidf_matrix = vectorizer.fit_transform(scales_preprocessed_joint.values()) # fit_transform() Learn vocabulary and idf, return document-term matrix. This is equivalent to fit followed by transform, but more efficiently implemented.

# print(count_matrix.shape)
# print(np.sum(tfidf_matrix, axis=0))

# Compute the cosine similarity between all scales
count_similarity = cosine_similarity(count_matrix)
tfidf_similarity = cosine_similarity(tfidf_matrix) #tfidf_matrix)

print(count_similarity)
# print(tfidf_similarity)

# Creating a DataFrame to store the similarities
df = pd.DataFrame(count_similarity, index=data.list_names, columns=data.list_names)

# Filling the diagonal with NaN values
np.fill_diagonal(df.values, np.nan)

df.head().round(3)

# Creating a DataFrame to store the similarities
df_tfidf_similarity = pd.DataFrame(tfidf_similarity, index=list_names, columns=list_names)

# Filling the diagonal with NaN values
np.fill_diagonal(df_tfidf_similarity.values, np.nan)

df_tfidf_similarity.head().round(3)

# Find the row and column labels for the maximum value with stopwords
max = df.stack().idxmax()
max_row_label_joint_raw, max_col_label_joint_raw = max[0], max[1]
print(f"The maximum BoW similarity value is located between {max_row_label_joint_raw} and {max_col_label_joint_raw}.")

# Find the row and column labels for the minimum value with stopwords
min = df.stack().idxmin()
min_row_label_joint_raw, min_col_label_joint_raw = min[0], min[1]
print(f"The minimum BoW similarity value is located between {min_row_label_joint_raw} and {min_col_label_joint_raw}.")

# Find the pair with the maximum - TF-IDF
max_joint_raw = df_tfidf_similarity.stack().idxmax()
max_row_label_joint_raw, max_col_label_joint_raw = max_joint_raw[0], max_joint_raw[1]
print(f"The maximum TF-IDF similarity value is located between {max_row_label_joint_raw} and {max_col_label_joint_raw}.")

# Find the pair with the minimum - TF-IDF
min_joint_raw = df_tfidf_similarity.stack().idxmin()
min_row_label_joint_raw, min_col_label_joint_raw = min_joint_raw[0], min_joint_raw[1]
print(f"The minimum TF-IDF similarity value is located between {min_row_label_joint_raw} and {min_col_label_joint_raw}.")

df = df_tfidf_similarity.fillna(0)

# similarity values
Similarities = df.values

# converting similarity values into dissimilarity values.
Distances = 1-Similarities
np.fill_diagonal(Distances, 0)
Distances = squareform(Distances)

Z = linkage(Distances, method = 'average')
# print(Z)

c, coph_dists = cophenet(Z, Distances)
print(c.round(3))

f, ax = plt.subplots(figsize=(10, 12))

plt.xlabel('Distance', fontsize=11, loc='center')

dendrogram(
    Z,
    leaf_rotation=35,  # rotates the x axis labels
    leaf_font_size=7,  # font size for the x axis labels
    labels=df.columns,
    # distance_sort='descending',
    orientation='right',  # Change orientation to 'right' for vertical dendrogram
    color_threshold=.60,
    above_threshold_color='#7591a1',
    ax=ax
)

ax.set_xticks(np.arange(.001, 1.1, .25))
ax.set_xticklabels([0, .25, .5, .75, 1], fontsize=6)

plt.savefig('tf-idf_dendrogram.pdf', dpi = 600, orientation = 'landscape', bbox_inches='tight')

plt.show()