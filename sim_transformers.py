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
from sentence_transformers import SentenceTransformer, util

# nltk.download('wordnet')
# nltk.download('stopwords')

data = DatasetLoader()

model_list = ['all-mpnet-base-v2', # our favorite
              'all-MiniLM-L12-v2',
              'gtr-t5-large',
              'all-mpnet-base-v1',
              'multi-qa-mpnet-base-dot-v1',
              'multi-qa-mpnet-base-cos-v1',
              'all-distilroberta-v1',
              'all-MiniLM-L12-v1',
              'multi-qa-distilbert-dot-v1',
              'multi-qa-distilbert-cos-v1']

#model_list = ['all-mpnet-base-v2']


scales_joint_raw = {variable: "".join(string_list).replace(".", ". ") for variable, string_list in data.scales_raw.items()}

for iModel, Model in enumerate(model_list):
  print(Model)
  model = SentenceTransformer(Model)

  # Initialize models outside the loops
  models = [SentenceTransformer(model_name) for model_name in model_list]

  num_models = len(models)
  num_refs = len(data.list_names)


# Initialize arrays
distances_array_joint_raw = np.zeros((num_models, num_refs, num_refs))
distances_array_joint_raw[:] = np.nan

# Loop over models
for iModel, model in enumerate(models):
    print(model_list[iModel])

    # Precompute embeddings for dataset with joint sentences
    for iRef, Ref in enumerate(data.list_names):
        print(data.list_names[iRef])
        ref_embeddings_joint_raw = [model.encode(scales_joint_raw[Ref], convert_to_tensor=True) for Ref in data.list_names]

    # Calculate distances for dataset with joint sentences
    for iRef in range(num_refs):
        list_1 = ref_embeddings_joint_raw[iRef]

        for iComp in range(iRef + 1, num_refs):
            Comp = data.list_names[iComp]
            list_2 = ref_embeddings_joint_raw[iComp]

            dists = util.pytorch_cos_sim(list_1, list_2)
            #dists_1 = np.max(np.array(dists), axis=0)
            #dists_2 = np.max(np.array(dists), axis=1)
            #final_dist = np.round(np.mean(np.hstack([dists_1, dists_2])), 3)

            distances_array_joint_raw[iModel, iRef, iComp] = dists

# If you need distances for iComp < iRef, you can copy values from above to save computations
for iModel in range(num_models):
    for iRef in range(num_refs):
        for iComp in range(iRef):
            distances_array_joint_raw[iModel, iRef, iComp] = distances_array_joint_raw[iModel, iComp, iRef]

# Create Raw DataFrame
df_Distances_joint_raw = pd.DataFrame(data=np.mean(distances_array_joint_raw, axis=0), columns=data.list_names,
                                      index=data.list_names)
#
# # Find the row and column labels for the maximum value with stopwords
# max_joint_raw = df_Distances_joint_raw.stack().idxmax()
# max_row_label_joint_raw, max_col_label_joint_raw = max_joint_raw[0], max_joint_raw[1]
# print(f"The maximum similarity value with stopwords is located between {max_row_label_joint_raw} and {max_col_label_joint_raw}")
#
# # Find the row and column labels for the minimum value with stopwords
# min_joint_raw = df_Distances_joint_raw.stack().idxmin()
# min_row_label_joint_raw, min_col_label_joint_raw = min_joint_raw[0], min_joint_raw[1]
# print(f"The minimum similarity value with stopwords is located between {min_row_label_joint_raw} and {min_col_label_joint_raw}")
#
# df_Distances_joint_raw = df_Distances_joint_raw.fillna(0)
#
# Similarities = df_Distances_joint_raw.values
# Distances = 1-Similarities # converting similarity values into dissimilarity values.
# np.fill_diagonal(Distances, 0)
# Distances = squareform(Distances)
#
# Z_joint_raw = linkage(Distances, method = 'average') # the method can be single, complete or average
# # print(Z)
#
# c_joint_raw, coph_dists = cophenet(Z_joint_raw, Distances) # the cophenetic correlation coefficient measures how well the hierarchical clustering represented by the linkage matrix Z preserves the original distances in the distance matrix Distances.
# print(c_joint_raw)
#
# f, ax = plt.subplots(figsize = (15, 6)) # f, ax = plt.subplots(figsize = (3.2, 10))
# plt.ylabel('Distance', fontsize = 11, loc = 'center')
# dendrogram(
#     Z_joint_raw,
#     leaf_rotation = 90,  # rotates the x axis labels
#     leaf_font_size = 8,  # font size for the x axis labels
#     labels = df_Distances_joint_raw.columns,
#     # distance_sort = 'descending',
#     orientation = 'top',
#     color_threshold = .35,
#     above_threshold_color = '#7591a1', #bcbddc
#     ax = ax
# )
# ax.set_yticks(np.arange(.001, 1.1, .25))
# ax.set_yticklabels([0, .25, .5, .75, 1], fontsize = 6)
#plt.tight_layout()
# plt.show()

# plt.figure(figsize=(15,5))
# sns.barplot(df_Distances_joint_raw.mean(axis=0).sort_values())
# plt.xticks(rotation=70)
# plt.tight_layout()

# plt.figure(figsize=(15,15))
# sns.heatmap(df_Distances_joint_raw)
