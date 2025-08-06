#--------------------------------------------------Cluster--------------------------------------------------------------------------------#
import pandas as pd
import scanpy as sc
import numpy as np
from collections import Counter
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


n_top_genes_   = 2000

data_dropout   = pd.read_csv('../Pollen_raw_data_fliter.csv',index_col=0)     
data_scGAAE    = pd.read_csv('../Pollen_imp_scGAAE.csv',index_col=0)

label          = pd.read_csv('../Pollen_raw_label_fliter.csv',index_col=0)
celltype_label = label['label'].values         
unique_class   = np.unique(celltype_label)
celltype_num   = len(unique_class)             
print(Counter(celltype_label))

ngene,ncell  = data_dropout.shape[0],data_dropout.shape[1]
print("The number of genes is:{}".format(ngene)+"\n"+"The number of cells is:{}".format(ncell)+"\n"+"The number of cell types is:{}".format(celltype_num)+"\n"+"The sparsity is:{}".format(np.mean(data_dropout.values==0)))  


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix1 = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix1, axis=0)) / np.sum(contingency_matrix1) 

def JaccardInd(ytrue,ypred):
    n = len(ytrue)
    a,b,c,d = 0,0,0,0
    for i in range(n-1):
        for j in range(i+1,n):
            if ((ypred[i] == ypred[j])&(ytrue[i]==ytrue[j])):
                a = a + 1
            elif ((ypred[i] == ypred[j])&(ytrue[i]!=ytrue[j])):
                b = b + 1
            elif ((ypred[i] != ypred[j])&(ytrue[i]==ytrue[j])):
                c = c + 1
            else:
                d = d + 1
    if (a==0)&(b==0)&(c==0):
        return 0
    else:
        return a/(a+b+c)


def cluster_metrics(data, label, n_top_genes_ = 2000, cellwise_norm=True, log1p=True, scale=True):
    
    df = pd.DataFrame()
    
    var_info = pd.DataFrame(index = data.index)                             
    obs_info = pd.DataFrame(index = data.columns)                           
    adata = sc.AnnData(np.array(data.T), obs = obs_info, var = var_info)    
    
    if cellwise_norm:
        sc.pp.normalize_total(adata, target_sum=1e4)                         
        
    if log1p:
        sc.pp.log1p(adata)                                                   
    
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = n_top_genes_) 
    adata = adata[:, adata.var.highly_variable]
    
    if scale:                                                                
        sc.pp.scale(adata)
    
    sc.tl.pca(adata, svd_solver='arpack')                                    
    
    sc.pp.neighbors(adata)   
    sc.tl.umap(adata)                                                        
    
    #sc.tl.tsne(adata)                                                        
    umap = adata.obsm['X_umap']                                                
    
    kmeans = KMeans(n_clusters = celltype_num,random_state=1).fit(umap)                
    cluster_label = kmeans.labels_
    
    df['ARI'] = [np.round(adjusted_rand_score(celltype_label,cluster_label),3)]         
    df['NMI'] = [np.round(normalized_mutual_info_score(celltype_label,cluster_label),3)]
    df['JI']  = [np.round(JaccardInd(celltype_label,cluster_label),3)]
    df['PS']  = [np.round(purity_score(celltype_label,cluster_label),3)]
    return df

os.environ['KMP_WARNINGS'] = '0'
warnings.filterwarnings("ignore")

Cluster_scGAAE = cluster_metrics(data_scGAAE, celltype_label, cellwise_norm=True, log1p=True, scale=True)
print("Cluster_scGAAE:{}".format(Cluster_scGAAE))
