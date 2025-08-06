#--------------------------------------------------PCC--------------------------------------------------------------------------------#
data_true      = pd.read_csv('../truecounts.csv',index_col=0) 
data_dropout   = pd.read_csv('../counts.csv',index_col=0)     
data_scGAAE    = pd.read_csv('../imp_scGAAE.csv',index_col=0)

#label          = pd.read_csv('../cell_info.csv',index_col=0)
#celltype_label = label['Group'].values          
#unique_class   = np.unique(celltype_label)
#celltype_num   = len(unique_class)     


def select_HVG(data, n_top_genes = 1000, cellwise_norm=True, log1p=True):
    var_info = pd.DataFrame(index = data.index)                              
    obs_info = pd.DataFrame(index = data.columns)                            
    adata = sc.AnnData(np.array(data.T), obs = obs_info, var = var_info)    # 
    
    if cellwise_norm:
        sc.pp.normalize_total(adata, target_sum=1e4)                          
        
    if log1p:
        sc.pp.log1p(adata)                                                    
    
    sc.pp.highly_variable_genes(adata, n_top_genes = n_top_genes, min_mean=0.0125, max_mean=3, min_disp=0.5) 
    adata.HVG = adata[:, adata.var.highly_variable]
                                                        
    return adata.HVG.X

def pearsonr(x, y):
    """
    计算x和y的Pearson相关系数
    """
    n = len(x)
    sx = np.std(x, ddof=1)    #  ddof=1 时计算的是样本标准差
    sy = np.std(y, ddof=1)
    r = np.cov(x, y, ddof=1)[0, 1] / (sx * sy)       # cov/std * std
    return r

def PCC(True_data_, imp_data_):
    pcc_results = []
    for i in range(len(True_data_)):
        pcc_result = pearsonr(True_data_[i],imp_data_[i])
        pcc_results.append(pcc_result)
    pcc_results_array = np.array(pcc_results)
    pcc_results_array = pcc_results_array[~np.isnan(pcc_results_array)]
    
    mean_value = np.mean(pcc_results_array)
    std_value  = np.std(pcc_results_array)
    pcc_results_array_without_outliers = pcc_results_array[(pcc_results_array > (mean_value - 2 * std_value)) & (pcc_results_array < (mean_value + 2 * std_value))]
    median     = np.median(pcc_results_array_without_outliers)
    return median, pcc_results_array_without_outliers


n_top_genes_ = 1000         # 挑选高变异基因

data_true_HVG       = select_HVG(data_true,    n_top_genes = n_top_genes_, cellwise_norm=True, log1p=True) 
data_dropout_HVG    = select_HVG(data_dropout, n_top_genes = n_top_genes_, cellwise_norm=True, log1p=True) 
data_scGAAE_HVG     = select_HVG(data_scGAAE,  n_top_genes = n_top_genes_, cellwise_norm=True, log1p=True)   

os.environ['KMP_WARNINGS'] = '0'
warnings.filterwarnings("ignore")

PCC_Gene_wise_dropout,    PCC_Gene_wise_res_dropout    = PCC(data_true_HVG.T, data_dropout_HVG.T)
PCC_Gene_wise_scGAAE,     PCC_Gene_wise_res_scGAAE     = PCC(data_true_HVG.T, data_scGAAE_HVG.T)
print("Gene_wise的PCC中位数_Dropout   :{}".format(PCC_Gene_wise_dropout))
print("Gene_wise的PCC中位数_scGAAE    :{}".format(PCC_Gene_wise_scGAAE))

PCC_Cell_wise_dropout,     PCC_Cell_wise_res_dropout    = PCC(data_true_HVG, data_dropout_HVG)
PCC_Cell_wise_scGAAE,      PCC_Cell_wise_res_scGAAE     = PCC(data_true_HVG, data_scGAAE_HVG)
print("Cell_wise的PCC中位数_Dropout   :{}".format(PCC_Cell_wise_dropout))
print("Cell_wise的PCC中位数_scGAAE    :{}".format(PCC_Cell_wise_scGAAE))
