library(SC3)
library(SingleCellExperiment)
library(yaml)
library(data.table)
library(scater)

sys_id <- as.numeric(Sys.getenv("SGE_TASK_ID"))
config_path <- './code/configs/'
config <- yaml.load_file(paste0(config_path,"config.yml"))

samp_info <- config$subsampling
num_rep   <- samp_info$num_rep
samp_path <- samp_info$subsample_path
#anno_file <- read.csv(samp_info$anno_file)
if (samp_info$is_fixed_ctype) {
  ntype <- samp_info$num_type
} else {
  ntype <- samp_info$num_type*sys_id
}

if (samp_info$is_fixed_cobs) {
  nobs <- 250 + samp_info$num_obs
} else {
  nobs <- 250 + samp_info$num_obs*sys_id
}
seed <- samp_info$seed

proc_info <- config$preprocess
min_cells <- proc_info$min_cells
min_genes <- proc_info$min_genes
# increase max_dim for large num cell types
multi   <- max(ntype%/%3 -1, 0)
max_dim <- proc_info$max_dim_base + multi *proc_info$max_dim_incre  #pca
jack_p <- proc_info$p_thresh
lab_path <- proc_info$label_path
pca_path <- proc_info$pca_proj_path



for (i in 1:num_rep) {
  file_name <- paste0(samp_path,'samp_',ntype,'type',nobs,'rep',i,'.csv')
  
  cdat      <- fread(file_name)
  cdat      <- data.frame(cdat)
  row.names(cdat)<-cdat$rn
  cdat <- cdat[,-1]
  
  sce <- SingleCellExperiment(
    assays = list(
      counts = as.matrix(cdat)#,
      #logcounts = log2(as.matrix(cdat) + 1)
    )
  )
  sce <- logNormCounts(sce)
  
  rowData(sce)$feature_symbol <- rownames(sce)
  #sce <- sce[!duplicated(rowData(sce)$feature_symbol), ]
  
  sce <- sc3_estimate_k(sce)
  k_est <- metadata(sce)$sc3$k_estimation
  sce <- sc3(sce, ks = k_est, rand_seed = seed)
  
  lab <- data.frame(colData(sce))
  lab$cell <- row.names(lab)
  idx <- grep('sc3_|cell',colnames(lab))
  out_lab <- lab[,idx]
  names(out_lab)[1] <-'label'
  
  write.csv(out_lab,paste0(lab_path,'sc3_',ntype,'type',nobs,'rep',i,'.csv'),row.names = F)
  
}

