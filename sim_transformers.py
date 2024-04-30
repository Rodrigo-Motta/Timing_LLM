
import nltk, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from dataset import DatasetLoader
import utils as ut

# nltk.download('wordnet')
# nltk.download('stopwords')

data = DatasetLoader()

# model_list = ['all-mpnet-base-v2', # our favorite
#               'all-MiniLM-L12-v2',
#               'gtr-t5-large',
#               'all-mpnet-base-v1',
#               'multi-qa-mpnet-base-dot-v1',
#               'multi-qa-mpnet-base-cos-v1',
#               'all-distilroberta-v1',
#               'all-MiniLM-L12-v1',
#               'multi-qa-distilbert-dot-v1',
#               'multi-qa-distilbert-cos-v1']

model_list = ['all-mpnet-base-v2']

# Joint scales_raw
data.scales_joint()

# Setting variables
num_refs = len(data.list_names)

# output = ut.PCA_embeddings(data,model_list,3)
#
# output = pd.DataFrame(output)
# output['Q'] = data.list_names
#
# #ut.plot_PCA(output)
# ut.plot_3D_PCA(output)

# output = ut.TSNE_embeddings(data, model_list, 3)
#
# output = pd.DataFrame(output)
# output['Q'] = data.list_names
#
# #ut.plot_PCA(output)
# ut.plot_3D_PCA(output)


# # Similarities loop
# distances_array_joint_raw = ut.similaraties(data, model_list, num_refs)
# #
# # Create Raw DataFrame
# df_Distances_joint_raw = pd.DataFrame(data=np.nanmean(distances_array_joint_raw, axis=0), columns=data.list_names,
#                                       index=data.list_names).replace(np.nan,0)
#
# # Convert upper triangular DataFrame to symmetric
# df_Distances_joint_raw = (df_Distances_joint_raw + df_Distances_joint_raw.T
#                           - np.fill_diagonal(np.diag(np.diag(df_Distances_joint_raw)),1))

# # Plots
# ut.plot_dendogram(df_Distances_joint_raw)
#
# ut.plot_heatmap(df_Distances_joint_raw)
#
# ut.plot_node_degree(df_Distances_joint_raw)



