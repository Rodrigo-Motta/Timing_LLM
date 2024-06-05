
import nltk, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from dataset import DatasetLoader
import utils as ut

# nltk.download('wordnet')
# nltk.download('stopwords')

data = DatasetLoader()

model_list = ['sentence-t5-large']
#model_list = ['all-distilroberta-v1']
#model_list = ['all-mpnet-base-v2']
#model_list = ['all-MiniLM-L6-v2']

# Joint scales_raw
data.scales_joint()
data.scramble_joint()

# Setting variables
num_refs = len(data.list_names)

################ Intra-Questionneire variance #################

#arr = ut.intra_similaraties(data, model_list, num_refs, num_scrambles=50)
#ut.plot_intra_barplot(arr, data)

########### K-Means Clusters information and plot ##############

# arr = ut.get_embedding(data, model_list, num_refs, num_scrambles=5)
# model = 0
# comps, variance = ut.PCA_embeddings(arr[:,model,:], n_comp=10)
# df = ut.clusters(arr[:, model,:], data, clusters=3)
# ut.plot_3D_PCA(comps, data.list_names, df['cluster'].values)

############ Hierarquical Clustering ###################

# model = 0
# arr = ut.get_embedding(data, model_list, num_refs, num_scrambles=5)
# groups = ut.hierarquical_clustering(arr[:, model,:], data)

################### Similarity matrix ######################
depression_scales = ['IDS','QIDS','BDI','CESD','SDS','MADRS','HDRS']
arr2 = np.array([ 0.61, 0.53, .26, .51, .29, .57, .61, .28, .43, .39, .5, .35,.50,.37,.42, .33, .38, .26, .35, .45, .31])
model = 0
arr = ut.get_embedding(data, model_list, num_refs, num_scrambles=5)
df = pd.DataFrame(data=np.corrcoef(arr[:, model,:]),columns=data.list_names,index=data.list_names)
df = df.loc[depression_scales,depression_scales]
df = ut.min_max_norm(df)

ut.plot_heatmap(df.replace(1.0 ,np.nan))
#ut.plot_node_degree(df)



# Repetitions

#average_dist, std_dist = ut.similaraties_average(data, model_list, num_refs, num_scrambles=10)

# Create Raw DataFrame
#df_Distances_joint_raw = ut.convert_arr_to_pandas(average_dist, data)

# Min Max Scaler
#df_Distances_joint_raw = ut.min_max_norm(df_Distances_joint_raw)

# Plots
#ut.plot_dendogram(df_Distances_joint_raw)

#ut.plot_heatmap(df_Distances_joint_raw.replace(1.0 ,np.nan))

#ut.plot_node_degree(df_Distances_joint_raw)
