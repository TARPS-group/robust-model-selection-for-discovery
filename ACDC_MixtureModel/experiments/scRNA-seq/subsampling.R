library(yaml)
library(data.table)
sys_id <- as.numeric(Sys.getenv("SGE_TASK_ID"))

##################################
##          samp func           ##
##################################

get_cell_df <- function(num_cell, num_type, dat_df, anno_f) {
  
  cell_names <- colnames(dat_df)
  anno_f1 <- anno_f[anno_f$cell %in% cell_names,]
  celltype_counts <- table(anno_f$cell_ontology_class)
  
  # restrict num_cell
  f1_types <- names(celltype_counts[celltype_counts >= num_cell])
  if (length(f1_types)==0){
    stop("no cell type has this many cells")
  }
  if (length(f1_types)< num_type){
    stop("not that many cell types")
  }
  anno_f2  <- anno_f1[anno_f1$cell_ontology_class %in% f1_types,]
  
  # restrict num_type
  f2_types <- sample(unique(anno_f2$cell_ontology_class), num_type)
  
  # for each cell type, sample xxx number of cell ids
  samp_ids <-  lapply(f2_types, function(cell_type) {
    # sample from one cell type
    cell_list <- anno_f2[anno_f2$cell_ontology_class==cell_type,]$cell
    cell_ids  <- sample(cell_list, num_cell)
    return(cell_ids)})
  out_ids <- unlist(samp_ids)
  return(dat_df[,colnames(dat_df)%in%out_ids])
}

##################################
##          load info           ##
##################################
config_path <- './code/configs/'
config <- yaml.load_file(paste0(config_path,"config.yml"))

samp_info <- config$subsampling
num_rep   <- samp_info$num_rep
out_path  <- samp_info$subsample_path

raw_data_file  <- samp_info$allsamp_file
a <- load(raw_data_file)
anno_file      <- read.csv(samp_info$anno_file)

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
set.seed(seed)

for (i in 1:num_rep) {
  samp <- get_cell_df(num_cell=nobs, num_type=ntype, dat_df=dat_all, anno_f=anno_file)
  samp <- as.data.table(samp, keep.rownames=TRUE)
  fwrite(samp, paste0(out_path,'samp_',ntype,'type',nobs,'rep',i,'.csv'))
}

