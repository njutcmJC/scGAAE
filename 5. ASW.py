#--------------------------------------------------ASW--------------------------------------------------------------------------------#
from sklearn.metrics import silhouette_score

data_true      = pd.read_csv('../truecounts.csv',index_col=0) 
data_dropout   = pd.read_csv('../counts.csv',index_col=0)     
data_scGAAE    = pd.read_csv('../imp_scGAAE.csv',index_col=0)

label          = pd.read_csv('../cell_info.csv',index_col=0)
celltype_label = label['Group'].values          
unique_class   = np.unique(celltype_label)
celltype_num   = len(unique_class)     

var_info = pd.DataFrame(index = data_dropout.index)                             
obs_info = pd.DataFrame(index = data_dropout.columns)                           
adata    = sc.AnnData(np.array(data_dropout.T), obs = obs_info, var = var_info)  

sc.pp.normalize_total(adata, target_sum=1e4)                            
sc.pp.log1p(adata)  
sc.pp.scale(adata)
sc.tl.pca(adata, svd_solver='arpack') 
sc.pp.neighbors(adata)   
sc.tl.umap(adata)

umap = adata.obsm['X_umap']                              
    
for i in range(celltype_num):
    plt.scatter(umap[celltype_label == unique_class[i],0],
                umap[celltype_label == unique_class[i],1], 
                s=10,label = unique_class[i] ) 
plt.xticks([],[])
plt.yticks([],[])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

asw = np.round(silhouette_score(umap,celltype_label),3)

plt.title("Dropout" + '  ASW:{:.2f}'.format(asw), fontsize=15, color='black')