rm(list=ls())
library(Seurat)
library(Matrix)
library(MAST)
library(DESeq2)

rm(list=ls())
foldChange <- 1     # 1, 1.2, 1.5

sc_true   <- read.csv("../truecounts.csv", row.names = 1);nrow(sc_true)
cell_info <- read.csv("../cell_info.csv", row.names = 1)
groups_lab<- cell_info$Group 

seu_true  <- CreateSeuratObject(counts = sc_true, project = "data", min.cells = 0, min.features = 0);seu_true     
Idents(seu_true) <- groups_lab
levels(seu_true)

seu_true <- NormalizeData(seu_true, normalization.method = "LogNormalize", scale.factor = 10000);seu_true
de.true  <- FindMarkers(seu_true, ident.1 = "Group1", ident.2 = "Group2", logfc.threshold = 0, test.use = 'MAST', min.pct = 0) # wilcox, t, MAST

de.true     <- de.true[complete.cases(de.true),];nrow(de.true)                 

de.true_subset1 <- de.true[de.true$p_val_adj <= 0.05,];nrow(de.true_subset1)
de.true.set1 <- rownames(de.true_subset1);length(de.true.set1)

de.true_subset2 <- de.true[(de.true$p_val_adj <= 0.05 & abs(de.true$avg_log2FC) > foldChange),];nrow(de.true_subset2)
de.true.set2 <- rownames(de.true_subset2);length(de.true.set2)


Eva_DEGs <- function(scRNA_data,name,method){
  res <- list()
  
  res$method <- method
  
  seu <- CreateSeuratObject(counts = scRNA_data, project = "data", min.cells = 0, min.features = 0);seu
  Idents(seu) <- cell_info$Group
  seu <- NormalizeData(seu, normalization.method = "LogNormalize", scale.factor = 10000);seu
  de.test  <- FindMarkers(seu, ident.1 = "Group1", ident.2 = "Group2", logfc.threshold = 0, test.use = method, min.pct = 0) # wilcox, t, MAST
  de.test  <- de.test[complete.cases(de.test),]                 
  de.test_subset1 <- de.test[de.test$p_val_adj <= 0.05,];nrow(de.test_subset1)
  de.test.set1 <- rownames(de.test_subset1);length(de.test.set1)
  
  de.test_subset2 <- de.test[(de.test$p_val_adj <= 0.05 & abs(de.test$avg_log2FC) > foldChange),];nrow(de.test_subset2)
  de.test.set2 <- rownames(de.test_subset2);length(de.test.set2)
  
  res$True_DEGs <- length(de.true.set2)
  res$test_DEGs <- length(de.test.set2)
  res$logFC     <- foldChange
  
  # DE result
  res$tp <- length(intersect(de.test.set2, de.true.set2))
  res$fp <- length(setdiff(de.test.set2, de.true.set2))
  res$fn <- length(setdiff(de.true.set2, de.test.set2))
  nde.truth <- setdiff(rownames(seu), de.true.set2)
  nde.test  <- setdiff(rownames(seu), de.test.set2)
  res$tn    <- length(intersect(nde.truth, nde.test))
  
  res$precision <- res$tp / (res$tp + res$fp); round(res$precision, 4)
  res$recall    <- res$tp / (res$tp + res$fn); round(res$recall, 4)
  res$tnr       <- res$tn / (res$tn + res$fn); round(res$tnr, 4)
  res$f1        <- 2 * res$precision * res$recall / (res$precision + res$recall); round(res$f1, 4)
  res$ACC       <- (res$tp + res$tn) / (res$tp + res$tn + res$fp + res$fn)
  
  de.true_subset1 <- de.true_subset1[order(de.true_subset1$p_val_adj),]
  de.test_subset1 <- de.test_subset1[order(de.test_subset1$p_val_adj),]
  
  top <- 100
  res$top100 <- length(intersect(rownames(de.test_subset1[1:top,]), rownames(de.true_subset1[1:top,])))
  
  top <- 200
  res$top200 <- length(intersect(rownames(de.test_subset1[1:top,]), rownames(de.true_subset1[1:top,])))
  
  top <- 300
  res$top300 <- length(intersect(rownames(de.test_subset1[1:top,]), rownames(de.true_subset1[1:top,])))
  
  top <- 400
  res$top400 <- length(intersect(rownames(de.test_subset1[1:top,]), rownames(de.true_subset1[1:top,])))
  
  top <- 500
  res$top500 <- length(intersect(rownames(de.test_subset1[1:top,]), rownames(de.true_subset1[1:top,])))
  
  res_df <- data.frame(res)
  rownames(res_df) <- name
  return(res_df)
}

sc_drop <- read.csv("../counts.csv", row.names = 1);nrow(sc_drop)
res_MAST   <- Eva_DEGs(sc_drop,"drop",'MAST')
res_wilcox <- Eva_DEGs(sc_drop,"drop",'wilcox')
res_T_test <- Eva_DEGs(sc_drop,"drop",'t')
res_drop <- rbind(res_MAST,
                  res_wilcox,
                  res_T_test)
res_drop 
rm(res_MAST,res_wilcox,res_T_test,sc_drop)
write.csv(res_drop,file = "../res_drop.csv")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#

sc_imp <- read.csv("../imp_scGAAE.csv", row.names = 1);nrow(sc_imp)
res_MAST   <- Eva_DEGs(sc_imp,"scGAAE",'MAST')
res_wilcox <- Eva_DEGs(sc_imp,"scGAAE",'wilcox')
res_T_test <- Eva_DEGs(sc_imp,"scGAAE",'t')
res_scGAAE <- rbind(res_MAST,
                    res_wilcox,
                    res_T_test)
res_scGAAE
rm(res_MAST,res_wilcox,res_T_test,sc_imp)
write.csv(res_scGAAE,file = "../res_scGAAE.csv")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------#