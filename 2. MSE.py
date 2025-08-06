#-----------------------------------------------------MSE-----------------------------------------------#
from sklearn.metrics import mean_squared_error

data_true      = pd.read_csv('../truecounts.csv',index_col=0) 
data_dropout   = pd.read_csv('../counts.csv',index_col=0)     
data_scGAAE    = pd.read_csv('../imp_scGAAE.csv',index_col=0)

label          = pd.read_csv('../cell_info.csv',index_col=0)
celltype_label = label['Group'].values          
unique_class   = np.unique(celltype_label)
celltype_num   = len(unique_class)     

def preprocess(data):                                 
    var_info = pd.DataFrame(index = data.index)                              
    obs_info = pd.DataFrame(index = data.columns)                           
    adata = sc.AnnData(np.array(data.T), obs = obs_info, var = var_info)    
    
    sc.pp.normalize_total(adata, target_sum=1e4)                            
    sc.pp.log1p(adata)                                                      
                                                               
    return adata.X

data_true_preprocess       = preprocess(data_true)        
data_dropout_preprocess    = preprocess(data_dropout)
data_scGAAE_preprocess     = preprocess(data_scGAAE) 

MSE_dropout = mean_squared_error(data_true_preprocess, data_dropout_preprocess, sample_weight=None, multioutput='uniform_average')
MSE_scGAAE  = mean_squared_error(data_true_preprocess, data_scGAAE_preprocess,  sample_weight=None, multioutput='uniform_average')

print("MSE_Dropout   :{}".format(MSE_dropout))
print("MSE_scGAAE    :{}".format(MSE_scGAAE))
