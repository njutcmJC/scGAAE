library(monocle)
library(TSCAN)

rm(list=ls())
deng        <- read.csv("../Deng_raw_data_fliter.csv", row.names = 1)     
deng_scGAAE <- read.csv("../Deng_scGAAE.csv", row.names = 1) 
deng_label  <- read.csv("../Deng_raw_label_fliter.csv", row.names = 1)  

deng_cellLabels = factor(deng_label$label,
                         levels=c('zygote', 'early 2-cell', 'mid 2-cell', 'late 2-cell',
                                  '4-cell', '8-cell', '16-cell', 'early blastocyst',
                                  'mid blastocyst', 'late blastocyst'))

my.monocle =  function(count, cellLabels){
  
  colnames(count) <- 1:ncol(count)    
  geneNames       <- rownames(count)                      
  rownames(count) <- 1:nrow(count)    
  
  pd <- data.frame(timepoint = cellLabels)
  pd <- new("AnnotatedDataFrame", data=pd)
  fd <- data.frame(gene_short_name = geneNames)
  fd <- new("AnnotatedDataFrame", data=fd)
  
  dCellData <- newCellDataSet(as.matrix(count), phenoData = pd, featureData = fd, expressionFamily = uninormal())  
  
  dCellData <- detectGenes(dCellData , min_expr = 0.1)
  expressed_genes <- row.names(subset(fData(dCellData),
                                      num_cells_expressed >= 50))     
  
  dCellData <- estimateSizeFactors(dCellData)              
  #dCellData <- estimateDispersions(dCellData, cores = 8)  
  
  diff_test_res <- differentialGeneTest(dCellData[expressed_genes,],
                                        fullModelFormulaStr = "~timepoint",
                                        cores = 8)                     
  ordering_genes <- row.names (subset(diff_test_res, qval < 0.01))    
  
  dCellData <- setOrderingFilter(dCellData, ordering_genes)           
  
  dCellData <- reduceDimension(dCellData, max_components = 2,
                               method = 'DDRTree', norm_method='none')   
  
  dCellData <- orderCells(dCellData)
  
  cor.kendall = cor(dCellData@phenoData@data$Pseudotime, as.numeric(dCellData@phenoData@data$timepoint), 
                    method = "kendall", use = "complete.obs")
  
  lpsorder2 = data.frame(sample_name = colnames(count), State= dCellData@phenoData@data$State, 
                         Pseudotime = dCellData@phenoData@data$Pseudotime, rank = rank(dCellData@phenoData@data$Pseudotime))
  
  lpsorder_rank = dplyr::arrange(lpsorder2, rank)
  
  lpsorder_rank$Pseudotime = lpsorder_rank$rank
  lpsorder_rank = lpsorder_rank[-4]
  
  #row.names(lpsorder_rank) = lpsorder_rank$sample_name
  #i <- sapply(lpsorder_rank, is.factor)
  
  lpsorder_rank[1] <- lapply(lpsorder_rank[1], as.character)
  
  subpopulation <- data.frame(cell = colnames(count), sub = as.numeric(cellLabels)-1)
  
  POS <- TSCAN::orderscore(subpopulation, lpsorder_rank)[1]
  
  #State
  p1 <- plot_cell_trajectory(dCellData, color_by = "State")
  
  #Pseudotime
  p2 <- plot_cell_trajectory(dCellData, color_by = "Pseudotime")
  
  #Celltype
  p3 <- plot_cell_trajectory(dCellData, color_by = "timepoint")   
  
  out = list(cor.kendall=cor.kendall, POS=POS, p1=p1, p2=p2, p3=p3)
  out
  
}

deng.dropout.monocle = my.monocle(deng, deng_cellLabels) 
deng.scGAAE.monocle  = my.monocle(deng_scGAAE, deng_cellLabels)

#-----------------------------------------------------------#
save(deng.dropout.monocle,
     deng.scGAAE.monocle,
     file = "../deng_monocle2.Rdata")


