library(yaml)
library(Seurat)
library(Matrix)
library(data.table)
sys_id <- as.numeric(Sys.getenv("SGE_TASK_ID"))

config_path <- './code/configs/'
config <- yaml.load_file(paste0(config_path,"config.yml"))

samp_info <- config$subsampling
num_rep   <- samp_info$num_rep
samp_path <- samp_info$subsample_path
anno_file <- read.csv(samp_info$anno_file)
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

get_pc_lab <- function(cmat, max_dim, min_cell, min_gene, alpha) {
  #log-normalize 
  rob <- CreateSeuratObject(counts = cmat, project = "scrna", min.cells = min_cell, min.features = min_gene)
  rob <- NormalizeData(rob, normalization.method = "LogNormalize", scale.factor = 10000)
  rob <- FindVariableFeatures(rob, selection.method = "vst", nfeatures = 2000)
  #standardize
  all.genes <- rownames(rob)
  rob       <- ScaleData(rob, features = all.genes)
  
  rob <- RunPCA(rob, features = VariableFeatures(object = rob),verbose = F,npcs = max_dim)
  rob <- JackStraw(object = rob, num.replicate = 100, dims = max_dim)
  rob <- ScoreJackStraw(rob,dim=1:max_dim)
  
  p_values <- data.frame(rob@reductions$pca@jackstraw@overall.p.values)
  # Bonferroni
  p_adj <- alpha/length(p_values$Score)
  index <- which(p_values$Score >= p_adj)[1]
  
  if (!is.na(index)) {
    max_pc <- index-1
  } else {
    max_pc <- max_dim
  }
  pca_obj<- rob[["pca"]]
  allpc <- data.frame(pca_obj@cell.embeddings)
  outpc <- allpc[,1:max_pc]
  
  rob <- FindNeighbors(rob, dims = 1:max_pc)
  rob <- FindClusters(rob, resolution = 0.4)
  
  out_lab <- as.data.frame(Idents(rob))
  names(out_lab) <-'label'
  out_lab$cell   <- row.names(out_lab)
  rownames(out_lab) <- NULL
  
  pl_list <- list(outpc,out_lab)
  return(pl_list)
}
# dat_paths <- list.files(path=samp_path, recursive=T, 
#                         pattern=paste0('samp_',ntype,'type',nobs),full.names=T)
set.seed(seed)

for (i in 1:num_rep) {
  file_name <- paste0(samp_path,'samp_',ntype,'type',nobs,'rep',i,'.csv')
  
  cdat      <- fread(file_name)
  cdat      <- data.frame(cdat)
  row.names(cdat)<-cdat$rn
  cdat <- cdat[,-1]
  dgc_mat <- Matrix(as.matrix(cdat),sparse=TRUE)
  
  pc_lab_ls <- get_pc_lab(cmat = dgc_mat, max_dim=max_dim, min_cell=min_cells,min_gene=min_genes,
                          alpha=jack_p)
  pcs <- pc_lab_ls[[1]]
  write.csv(pcs,paste0(pca_path,'pca_',ntype,'type',nobs,'rep',i,'.csv'))
  
  labs <- pc_lab_ls[[2]]
  write.csv(labs,paste0(lab_path,'seurat_',ntype,'type',nobs,'rep',i,'.csv'),row.names = F)
  
}

