#--------------------------------------------------------------------------------------#
#-----------------------------Simulation-----------------------------------------------#
#--------------------------------------------------------------------------------------#

#install.packages("Rtools")
#if (!requireNamespace("BiocManager", quietly = TRUE))
#install.packages("BiocManager")
#BiocManager::install("splatter")

library(splatter)

rm(list = ls())
params.custom <- list(
  nGenes=1000, batchCells=500,
  group.prob=c(0.2,0.2,0.2,0.2,0.2),
  dropout.shape=-2.5, dropout.mid=2.5, 
  dropout.type="experiment"    
)
set.seed(1) 
params <- newSplatParams()
params <- setParams(params, update=params.custom)

sim <- splatSimulateGroups(params)
matrices <- assays(sim)

raw<- matrices[["counts"]]
truth <- matrices[["TrueCounts"]]
dropout <- matrices[["Dropout"]]

dropprob<-length(dropout[dropout==TRUE])/(nrow(dropout)*ncol(dropout))  
dropprob


# Write metadata
write.csv(colData(sim),"../cell_info.csv")
write.csv(rowData(sim),"../gene_info.csv")

# Write data
write.csv(raw,"../counts.csv")
write.csv(truth,"../truecounts.csv")
dropout<-as.data.frame(matrices$Dropout)                                
write.csv(dropout,"../dropout_mask.csv")
