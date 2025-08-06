#--------------------------------------------------Cluster--------------------------------------------------------------------------------#
import math
import pandas as pd
import scanpy as sc
import numpy as np
from collections import Counter
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score



data  = pd.read_csv('../raw_data.csv',index_col=0) 
label = pd.read_csv('../raw_label.csv',index_col=0) 

celltype_label  = label['label'].values         
unique_class    = np.unique(celltype_label)
celltype_num    = len(unique_class)            
print(Counter(celltype_label))

ngene,ncell  = data.shape[0],data.shape[1]
print("The number of genes is:{}".format(ngene)+"\n"+"The number of cells is:{}".format(ncell)+"\n"+"The number of cell types is:{}".format(celltype_num)+"\n"+"The sparsity is:{}".format(np.mean(data.values==0)))  

var_info = pd.DataFrame(index = data.index)          
obs_info = pd.DataFrame(index = data.columns)        
adata = sc.AnnData(np.array(data.T), obs = obs_info, var = var_info)    
adata.obs['label'] = pd.Categorical(label['label'])

min_cells_fliter = math.ceil(adata.n_obs * 0.01)   
print(min_cells_fliter)

min_genes_fliter = math.ceil(adata.n_vars * 0.01)  
print(min_genes_fliter)

sc.pp.filter_cells(adata, min_genes = min_genes_fliter)
sc.pp.filter_genes(adata, min_cells = min_cells_fliter)

adata_fliter = adata.to_df().T
adata_fliter.to_csv("../raw_data_fliter.csv")

label_fliter = pd.DataFrame(adata.obs['label'])
label_fliter.to_csv("../raw_label_fliter.csv")
