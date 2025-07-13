library(aricode)
lab_path ='./outputs/labels/'
anno_f <-  read.csv('./data/annotations_facs.csv')
anno <- anno_f[,3:4]

##############
# manually selected K
##############
prop_K <- c(13,16,15,11,7,15,8,11,8,7,16,8,11,8,10)
lab_path <- "./code/rho_cal/labels/prop/"
files <- list.files(path=lab_path, recursive=F, pattern='SARM',full.names=F)
df <- data.frame(files)
df$true_num_cell <- as.numeric(sub("SARM_(\\d+)type.*", "\\1", df$files))
df$obs <- as.numeric(sub(".*type(\\d+)\\.csv", "\\1", df$files))
df2 <- df[order(df$true_num_cell, df$obs), ]
df2$num_K <- prop_K

lab_info <- list()
for (lab_name in df2$files) {
  lab <- read.csv(paste0(lab_path,lab_name))
  index <- df2[df2$files==lab_name,]$num_K
  sarm_lab <- lab[,c(index, dim(lab)[2])]
  names(sarm_lab)[1] <- 'label'
  cell_obs <- df2[df2$files==lab_name,]$obs
  
  labels <- merge(anno,sarm_lab,by='cell')
  labels <- labels[complete.cases(labels), ]
  
  ari <- ARI(labels$cell_ontology_class, labels$label)
  ami <- AMI(labels$cell_ontology_class, labels$label)
  true_num_cell <- length(unique(labels$cell_ontology_class))
  est_num_cell  <- length(unique(labels$label))
  tuple <- list(list("obs"=cell_obs,'true_num_cell'=true_num_cell, 
                     'est_num_cell'=est_num_cell, 'ari'=ari, 'ami'=ami))
  lab_info <- append(lab_info,tuple)
}
prop_man <- data.frame(do.call(rbind, lapply(lab_info,unlist)))

write.csv(prop_man, "./code/as/manualK_sarms_prop_SumStats15.csv",row.names = F)

##############
# automated K
##############
rho_path <- "./code/rho_cal/rho_loss/prop/"
files <- list.files(path=rho_path, recursive=F, pattern='rho',full.names=F)
k_df <- data.frame(matrix(NA,nrow = 15,ncol=2))
names(k_df) <- c("file","num_K")

for (i in 1:length(files)) {
  
  file <- files[i]
  k_df$file[i] <- file
  dat <- read.csv(paste0(rho_path,file))
  r_info <- get_rho_start(dat,rho_len_thresh = 1, max_r = 20)
  rho_min <- r_info$rho_start
  name <- r_info$colname
  k <- as.integer(strsplit(name,split = "K")[[1]][2])
  k_df$num_K[i] <- k
}

k_df$true_num_cell <- as.numeric(sub("rho_(\\d+)type.*", "\\1", k_df$file))
k_df$obs <- as.numeric(sub(".*type(\\d+)\\.csv", "\\1", k_df$file))

lab_path ='./code/rho_cal/labels/prop/'
anno_f <-  read.csv('./data/annotations_facs.csv')
anno <- anno_f[,3:4]

files <- list.files(path=lab_path, recursive=F, pattern='SARM',full.names=F)
df <- data.frame(files)
df$true_num_cell <- as.numeric(sub("SARM_(\\d+)type.*", "\\1", df$files))
df$obs <- as.numeric(sub(".*type(\\d+)\\.csv", "\\1", df$files))
df2 <- merge(df,k_df,by=c("true_num_cell", "obs"))

lab_info <- list()
for (lab_name in df2$files) {
  lab <- read.csv(paste0(lab_path,lab_name))
  index <- df2[df2$files==lab_name,]$num_K
  sarm_lab <- lab[,c(index, dim(lab)[2])]
  names(sarm_lab)[1] <- 'label'
  cell_obs <- df2[df2$files==lab_name,]$obs
  labels <- merge(anno,sarm_lab,by='cell')
  labels <- labels[complete.cases(labels), ]
  ari <- ARI(labels$cell_ontology_class, labels$label)
  ami <- AMI(labels$cell_ontology_class, labels$label)
  true_num_cell <- length(unique(labels$cell_ontology_class))
  est_num_cell  <- length(unique(labels$label))
  tuple <- list(list("obs"=cell_obs,'true_num_cell'=true_num_cell, 
                     'est_num_cell'=est_num_cell, 'ari'=ari, 'ami'=ami))
  lab_info <- append(lab_info,tuple)
}
prop_auto <- data.frame(do.call(rbind, lapply(lab_info,unlist)))

write.csv(prop_auto, "./code/as/autoK_sarms_prop_SumStats15.csv",row.names = F)

names(prop_auto)[3:5] <- paste0("auto_",names(prop_auto)[3:5]) 
names(prop_man)[3:5] <- paste0("man_",names(prop_man)[3:5]) 

all <- merge(prop_auto,prop_man,by=c('true_num_cell','obs'))

outp <- "./code/as/plt/"
library(viridis)
pdf(paste0(outp,"comp_ari_hist.pdf"), width = 6, height = 3)
par(mfrow = c(1, 2), mar = c(5, 4, 1, 2) + 0.1)
hist(all$auto_ari - all$man_ari, 
     xlab = "ARI (auto - manual)", 
     main = NULL, 
     col = viridis(1), 
     border = "black", 
     cex.lab = 1.2, 
     cex.axis = 1.1)

hist(all$auto_ami - all$man_ami, 
     xlab = "AMI (auto - manual)", 
     main = NULL, 
     col = viridis(1), 
     border = "black", 
     cex.lab = 1.2, 
     cex.axis = 1.1)
dev.off()

############
# SC3
############
lab_path <- "./code/rho_cal/labels/prop/"
sc_stats <- get_stats(name='sc3',lab_path=lab_path,anno=anno)
write.csv(sc_stats, "./code/as/sc3_prop_SumStats15.csv",row.names = F)

############
# Seurat
############

seurat_stats <- get_stats(name='seurat',lab_path=lab_path,anno=anno)
write.csv(seurat_stats, "./code/as/seurat_prop_SumStats15.csv",row.names = F)

library(tidyverse)

propdat_man_auto <- ggplot() +
  geom_point(data = all, aes(x = true_num_cell, y = auto_est_num_cell,color='Auto K'), alpha = 1, position = "jitter",size = 1.5)+
  geom_point(data = all, aes(x = true_num_cell, y = man_est_num_cell,color='Manual K'), alpha = 1, position = "jitter",size = 1.5)+
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "dashed") +  
  labs(
    x = expression(K[o]),
    y = expression(hat(K)),
    color = "Method"
  ) +
  xlim(c(12, 17)) +
  ylim(c(6.5, 17.5)) +
  scale_color_manual(values = c("Auto K" = "#228833", "Manual K" = "#CC79A7")) +
  theme_minimal() +
  theme(
    strip.text = element_blank(),  # Remove subplot titles
    axis.title = element_text(size = 13),  # Increase font size of axis labels
    axis.text = element_text(size = 12),   # Increase font size of axis tick marks
    legend.text = element_text(size = 12), # Increase font size of legend labels
    legend.title = element_text(size = 12)
  )

ggsave(paste0(outp,"autoSARM_comp.pdf"), plot = propdat_man_auto, width = 5, height = 3, units = "in")

get_rho_start <- function(df, rho_len_thresh, max_r) {
  r_start <- NULL
  min_loss_colname <- NULL
  min_loss <- NULL
  for (i in 2: nrow(df)) {
    rho <- df$rho[i]
    losses <- df[i, 2:ncol(df)]
    if (is.null(r_start)& is.null(min_loss_colname)& is.null(min_loss)){
      is_converged <- sapply(2:ncol(df), function(col) {
        j <- i-1
        diff(df[j:i, col])==0
      })
      converged_ls <- losses[is_converged]
      if (length(converged_ls) > 0) {
        min_loss <- min(converged_ls)
        min_loss_colname <- names(losses)[is_converged & (losses == min_loss)]
        r_start <- df$rho[i-1]
      } 
      if (rho==max_r){
        min_loss_colname <- names(losses)[(losses == min_loss)]
        out_ls <- list('rho_start'= 1,'colname'= min_loss_colname)
        return(out_ls)
      }
    } else{
      # check if exist numb < min_loss
      if (sum(losses < min_loss)!= 0|rho==max_r) {
        rho_len <- rho - r_start
        if (rho_len>= rho_len_thresh){
          out_ls <- list('rho_start'= r_start,'colname'= min_loss_colname)
          return(out_ls)
        } else {
          # reset all, start looking for convergence again
          r_start <- NULL
          min_loss_colname <- NULL
          min_loss <- NULL
        }
      }
    }
  }
}


############################################################
# SARMS vs. other packages
############################################################


############
# SARMS auto
############
outp <- "./code/as/plt/"

rho_path <- "./code/rho_cal/rho_loss/"


files <- list.files(path=rho_path, recursive=F, pattern='rho',full.names=F)

k_df <- data.frame(matrix(NA,nrow = 80,ncol=2))
names(k_df) <- c("file","num_K")


for (i in 1:length(files)) {
  
  file <- files[i]
  k_df$file[i] <- file
  
  dat <- read.csv(paste0(rho_path,file))
  
  
  r_info <- get_rho_start(dat,rho_len_thresh = 1.2, max_r = 20)
  
  rho_min <- r_info$rho_start
  name <- r_info$colname
  
  k <- as.integer(strsplit(name,split = "K")[[1]][2])
  
  k_df$num_K[i] <- k
}

#Note: tend to overestimate for samples with large true num_type when rho_len_thresh is small
#      however, when thresh large, underestimate especially for numtype= 2



# get ARI 

k_df$true_num_cell <- as.numeric(sub("rho_(\\d+)type.*", "\\1", k_df$file))
k_df$rep <- as.numeric(sub(".*rep(\\d+)\\.csv", "\\1", k_df$file))



lab_path ='./outputs/labels/'
anno_f <-  read.csv('./data/annotations_facs.csv')
anno <- anno_f[,3:4]


files <- list.files(path=lab_path, recursive=T, pattern='SARM',full.names=F)
df <- data.frame(files)
df$true_num_cell <- as.numeric(sub("SARM_(\\d+)type.*", "\\1", df$files))
df$rep <- as.numeric(sub(".*rep(\\d+)\\.csv", "\\1", df$files))
df2 <- merge(df,k_df,by=c("true_num_cell", "rep"))

lab_info <- list()
for (lab_name in df2$files) {
  lab <- read.csv(paste0(lab_path,lab_name))
  index <- df2[df2$files==lab_name,]$num_K
  sarm_lab <- lab[,c(index, dim(lab)[2])]
  names(sarm_lab)[1] <- 'label'
  
  labels <- merge(anno,sarm_lab,by='cell')
  labels <- labels[complete.cases(labels), ]
  
  ari <- ARI(labels$cell_ontology_class, labels$label)
  ami <- AMI(labels$cell_ontology_class, labels$label)
  true_num_cell <- length(unique(labels$cell_ontology_class))
  est_num_cell  <- length(unique(labels$label))
  tuple <- list(list('true_num_cell'=true_num_cell, 
                     'est_num_cell'=est_num_cell, 'ari'=ari, 'ami'=ami))
  lab_info <- append(lab_info,tuple)
}
sarm_stats_auto <- data.frame(do.call(rbind, lapply(lab_info,unlist)))

write.csv(sarm_stats_auto, "./code/as/autoK_sarms_SumStats80.csv",row.names = F)


############
# SARMS manual
############

library(aricode)
lab_path ='./outputs/labels/'
anno_f <-  read.csv('./data/annotations_facs.csv')
anno <- anno_f[,3:4]

get_stats <- function(name,lab_path,anno){
  files <- list.files(path=lab_path, recursive=T, pattern=name,full.names=T)
  lab_info <- list()
  for (file in files){
    lab <- read.csv(file)
    labels <- merge(anno,lab,by='cell')
    labels <- labels[complete.cases(labels), ]
    
    ari <- ARI(labels$cell_ontology_class, labels$label)
    ami <- AMI(labels$cell_ontology_class, labels$label)
    true_num_cell <- length(unique(labels$cell_ontology_class))
    est_num_cell  <- length(unique(labels$label))
    tuple <- list(list('true_num_cell'=true_num_cell, 
                       'est_num_cell'=est_num_cell, 'ari'=ari, 'ami'=ami))
    lab_info <- append(lab_info,tuple)
  }
  
  df <- data.frame(do.call(rbind, lapply(lab_info,unlist)))
}


files <- list.files(path=lab_path, recursive=T, pattern='SARM',full.names=F)
df <- data.frame(files)
df$true_num_cell <- as.numeric(sub("SARM_(\\d+)type.*", "\\1", df$files))
df$rep <- as.numeric(sub(".*rep(\\d+)\\.csv", "\\1", df$files))
df2 <- df[order(df$true_num_cell, df$rep), ]

df2$num_K <- c(3,5,4,6,2,3,4,5,6,3,
               3,6,2,4,4,6,7,4,8,5,
               8,10,6,8,8,6,8,5,4,8,
               8,9,13,13,7,11,10,13,9,10,
               8,11,12,15,6,9,13,15,7,6,
               12,9,10,10,8,16,11,13,14,16,
               18,12,18,14,15,14,16,10,16,14,
               21,16,18,18,13,11,16,13,18,17)


lab_info <- list()
for (lab_name in df2$files) {
  lab <- read.csv(paste0(lab_path,lab_name))
  index <- df2[df2$files==lab_name,]$num_K
  sarm_lab <- lab[,c(index, dim(lab)[2])]
  names(sarm_lab)[1] <- 'label'
  
  labels <- merge(anno,sarm_lab,by='cell')
  labels <- labels[complete.cases(labels), ]
  
  ari <- ARI(labels$cell_ontology_class, labels$label)
  ami <- AMI(labels$cell_ontology_class, labels$label)
  true_num_cell <- length(unique(labels$cell_ontology_class))
  est_num_cell  <- length(unique(labels$label))
  tuple <- list(list('true_num_cell'=true_num_cell, 
                     'est_num_cell'=est_num_cell, 'ari'=ari, 'ami'=ami))
  lab_info <- append(lab_info,tuple)
}

sarm_man_stats <- data.frame(do.call(rbind, lapply(lab_info,unlist)))

write.csv(sarm_man_stats, "./code/as/manualK_sarms_SumStats80.csv",row.names = F)

############
# SC3
############
sc_stats <- get_stats(name='sc3',lab_path=lab_path,anno=anno)
write.csv(sc_stats, "./code/as/sc3_SumStats80.csv",row.names = F)

############
# Seurat
############

seurat_stats <- get_stats(name='seurat',lab_path=lab_path,anno=anno)
write.csv(seurat_stats, "./code/as/seurat_SumStats80.csv",row.names = F)


################## all

sarm_stats_auto <-read.csv("./code/as/autoK_sarms_SumStats80.csv")

sarm_stats <- read.csv("./code/as/manualK_sarms_SumStats80.csv")
sc_stats <- read.csv("./code/as/sc3_SumStats80.csv")
seurat_stats <- read.csv("./code/as/seurat_SumStats80.csv")

names(sarm_stats)[2:4] <- paste0('sarm_',colnames(sarm_stats)[2:4])
# names(seurat_stats)[2:4] <- paste0('seurat_',colnames(seurat_stats)[2:4])
# names(sc_stats)[2:4] <- paste0('sc_',colnames(sc_stats)[2:4])

library(tidyverse)

##########################

outp <- "./code/as/plt/"
library(viridis) 

tool_est <- ggplot() +
  geom_point(data = sarm_stats, aes(x = true_num_cell, y = sarm_est_num_cell, color = "Our method"), 
             alpha = 0.9, position = "jitter") +
  geom_point(data = sc_stats, aes(x = true_num_cell, y = sc_est_num_cell, color = "SC3"), 
             alpha = 0.9, position = "jitter") +
  geom_point(data = seurat_stats, aes(x = true_num_cell, y = seurat_est_num_cell, color = "Seurat"), 
             alpha = 0.9, position = "jitter") +
  scale_color_viridis_d(option = "D", name = "Tool") + # Using a discrete viridis palette
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "dashed") +  
  theme_minimal() + 
  labs(
    x = expression(K[o]),
    y = expression(hat(K)),
    color = "Tool"
  ) +
  theme(
    axis.title = element_text(size = 13),  # Increase font size of axis labels
    axis.text = element_text(size = 12),   # Increase font size of axis tick marks
    legend.text = element_text(size = 12), # Increase font size of legend labels
    legend.title = element_text(size = 12)
  )

ggsave(paste0(outp,"tool_comp_estk.pdf"), plot = tool_est , width = 5, height = 3, units = "in")

ggsave(paste0(outp,"tool_comp_estk2.pdf"), plot = tool_est , width = 5, height = 3, units = "in")

## ari

tool_ari <- ggplot() +
  geom_point(data = sarm_stats, aes(x = true_num_cell, y = sarm_ari, color = "Our method"), alpha = 0.9, position = "jitter") +
  geom_point(data = sc_stats, aes(x = true_num_cell, y = sc_ari, color = "SC3"), alpha = 0.9, position = "jitter") +
  geom_point(data = seurat_stats, aes(x = true_num_cell, y = seurat_ari, color = "Seurat"), alpha = 0.9, position = "jitter") +
  scale_color_viridis_d(option = "D", name = "Tool") +
  theme_minimal() + 
  theme_minimal() + 
  labs(
    x = expression(K[o]),
    y = 'ARI',
    color = "Tool"
  ) +
  theme(
    axis.title = element_text(size = 13),  
    axis.text = element_text(size = 12),   
    legend.text = element_text(size = 12), 
    legend.title = element_text(size = 12)
  )

ggsave(paste0(outp,"tool_comp_ari.pdf"), plot = tool_ari , width = 5, height = 3, units = "in")

ggsave(paste0(outp,"tool_comp_ari2.pdf"), plot = tool_ari , width = 5, height = 3, units = "in")


##  ami

tool_ami <- ggplot() +
  geom_point(data = sarm_stats, aes(x = true_num_cell, y = sarm_ami, color = "Our method"), alpha = 0.9, position = "jitter",size = 1.5) +
  geom_point(data = sc_stats, aes(x = true_num_cell, y = sc_ami, color = "SC3"), alpha = 0.9, position = "jitter",size = 1.5) +
  geom_point(data = seurat_stats, aes(x = true_num_cell, y = seurat_ami, color = "Seurat"), alpha = 0.9, position = "jitter",size = 1.5) +
  scale_color_viridis_d(option = "D", name = "Tool") +
  theme_minimal() + 
  theme_minimal() + 
  labs(
    x = expression(K[o]),
    y = 'AMI',
    color = "Tool"
  ) +
  theme(
    axis.title = element_text(size = 13),  
    axis.text = element_text(size = 12),   
    legend.text = element_text(size = 12), 
    legend.title = element_text(size = 12)
  )

ggsave(paste0(outp,"tool_comp_ami.pdf"), plot = tool_ami , width = 5, height = 3, units = "in")

ggsave(paste0(outp,"tool_comp_ami2.pdf"), plot = tool_ami , width = 5, height = 3, units = "in")


### get average
path <- "./code/as/"
seu <- read.csv(paste0(path,"seurat_SumStats80.csv"))
mean(seu$seurat_ari)
mean((seu$seurat_ami))
# > mean(seu$seurat_ari)
# [1] 0.702149
# > mean((seu$seurat_ami))
# [1] 0.7339123

sc3 <- read.csv(paste0(path, "sc3_SumStats80.csv"))
mean(sc3$sc_ari)
mean(sc3$sc_ami)
# > mean(sc3$sc_ari)
# [1] 0.4304533
# > mean(sc3$sc_ami)
# [1] 0.5694871

sarms_manual <- read.csv(paste0(path, "manualK_sarms_SumStats80.csv"))
mean(sarms_manual$ari)
mean(sarms_manual$ami)

######### 
outp <- "./code/as/plt/"
library(viridis) # for viridis palette

tool_est <- ggplot() +
  geom_point(data = sarm_stats, aes(x = true_num_cell, y = sarm_est_num_cell, color = "Manual"), 
             alpha = 0.9, position = "jitter") +
  geom_point(data = sarm_stats_auto, aes(x = true_num_cell, y = est_num_cell, color = "Auto"), 
             alpha = 0.9, position = "jitter") +
  scale_color_viridis_d(option = "D", name = "Tool") + # Using a discrete viridis palette
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "dashed") +  
  theme_minimal() + 
  labs(
    x = expression(K[o]),
    y = expression(hat(K)),
    color = "Tool"
  ) +
  theme(
    axis.title = element_text(size = 13),  # Increase font size of axis labels
    axis.text = element_text(size = 12),   # Increase font size of axis tick marks
    legend.text = element_text(size = 12), # Increase font size of legend labels
    legend.title = element_text(size = 12)
  )

ggsave(paste0(outp,"tool_comp_estk2.pdf"), plot = tool_est , width = 5, height = 3, units = "in")


############
# SARMS auto
############
outp <- "./code/as/plt/"

rho_path <- "./code/rho_cal/rho_loss/"


files <- list.files(path=rho_path, recursive=F, pattern='rho',full.names=F)

k_df <- data.frame(matrix(NA,nrow = 80,ncol=2))
names(k_df) <- c("file","num_K")


for (i in 1:length(files)) {
  
  file <- files[i]
  k_df$file[i] <- file
  
  dat <- read.csv(paste0(rho_path,file))
  
  
  r_info <- get_rho_start(dat,rho_len_thresh = 1.2, max_r = 20)
  
  rho_min <- r_info$rho_start
  name <- r_info$colname
  
  k <- as.integer(strsplit(name,split = "K")[[1]][2])
  
  k_df$num_K[i] <- k
}

#Note: tend to overestimate for samples with large true num_type when rho_len_thresh is small
#      however, when thresh large, underestimate especially for numtype= 2

# get ARI 
k_df$true_num_cell <- as.numeric(sub("rho_(\\d+)type.*", "\\1", k_df$file))
k_df$rep <- as.numeric(sub(".*rep(\\d+)\\.csv", "\\1", k_df$file))



lab_path ='./outputs/labels/'
anno_f <-  read.csv('./data/annotations_facs.csv')
anno <- anno_f[,3:4]


files <- list.files(path=lab_path, recursive=T, pattern='SARM',full.names=F)
df <- data.frame(files)
df$true_num_cell <- as.numeric(sub("SARM_(\\d+)type.*", "\\1", df$files))
df$rep <- as.numeric(sub(".*rep(\\d+)\\.csv", "\\1", df$files))
df2 <- merge(df,k_df,by=c("true_num_cell", "rep"))

lab_info <- list()
for (lab_name in df2$files) {
  lab <- read.csv(paste0(lab_path,lab_name))
  index <- df2[df2$files==lab_name,]$num_K
  rep_idx <- df2[df2$files==lab_name,]$rep
  sarm_lab <- lab[,c(index, dim(lab)[2])]
  names(sarm_lab)[1] <- 'label'
  
  labels <- merge(anno,sarm_lab,by='cell')
  labels <- labels[complete.cases(labels), ]
  
  ari <- ARI(labels$cell_ontology_class, labels$label)
  ami <- AMI(labels$cell_ontology_class, labels$label)
  true_num_cell <- length(unique(labels$cell_ontology_class))
  est_num_cell  <- length(unique(labels$label))
  tuple <- list(list('true_num_cell'=true_num_cell, 
                     'est_num_cell'=est_num_cell, 'ari'=ari, 'ami'=ami,
                     'rep'=rep_idx))
  lab_info <- append(lab_info,tuple)
}
sarm_stats_auto <- data.frame(do.call(rbind, lapply(lab_info,unlist)))

write.csv(sarm_stats_auto, "./code/as/autoK_sarms_SumStats80_repNum.csv",row.names = F)


############
# SARMS manual
############

library(aricode)
lab_path ='./outputs/labels/'
anno_f <-  read.csv('./data/annotations_facs.csv')
anno <- anno_f[,3:4]

files <- list.files(path=lab_path, recursive=T, pattern='SARM',full.names=F)
df <- data.frame(files)
df$true_num_cell <- as.numeric(sub("SARM_(\\d+)type.*", "\\1", df$files))
df$rep <- as.numeric(sub(".*rep(\\d+)\\.csv", "\\1", df$files))
df2 <- df[order(df$true_num_cell, df$rep), ]

df2$num_K <- c(3,5,4,6,2,3,4,5,6,3,
               3,6,2,4,4,6,7,4,8,5,
               8,10,6,8,8,6,8,5,4,8,
               8,9,13,13,7,11,10,13,9,10,
               8,11,12,15,6,9,13,15,7,6,
               12,9,10,10,8,16,11,13,14,16,
               18,12,18,14,15,14,16,10,16,14,
               21,16,18,18,13,11,16,13,18,17)


lab_info <- list()
for (lab_name in df2$files) {
  lab <- read.csv(paste0(lab_path,lab_name))
  index <- df2[df2$files==lab_name,]$num_K
  rep_idx <- df2[df2$files==lab_name,]$rep
  sarm_lab <- lab[,c(index, dim(lab)[2])]
  names(sarm_lab)[1] <- 'label'
  
  labels <- merge(anno,sarm_lab,by='cell')
  labels <- labels[complete.cases(labels), ]
  
  ari <- ARI(labels$cell_ontology_class, labels$label)
  ami <- AMI(labels$cell_ontology_class, labels$label)
  true_num_cell <- length(unique(labels$cell_ontology_class))
  est_num_cell  <- length(unique(labels$label))
  tuple <- list(list('true_num_cell'=true_num_cell, 
                     'est_num_cell'=est_num_cell, 'ari'=ari, 'ami'=ami,
                     'rep'=rep_idx))
  lab_info <- append(lab_info,tuple)
}

sarm_man_stats <- data.frame(do.call(rbind, lapply(lab_info,unlist)))

write.csv(sarm_man_stats, "./code/as/manualK_sarms_SumStats80_repNum.csv",row.names = F)

########################### plot
sarm_stats_auto <- read.csv("./code/as/autoK_sarms_SumStats80_repNum.csv")

sarm_stats <- read.csv("./code/as/manualK_sarms_SumStats80_repNum.csv")

names(sarm_stats)[2:4] <- paste0('man_',colnames(sarm_stats)[2:4])
names(sarm_stats_auto)[2:4] <- paste0('auto_',colnames(sarm_stats_auto)[2:4])

all <- merge(sarm_stats,sarm_stats_auto,by=c('true_num_cell','rep'))

outp <- "./code/as/plt/"

summary(all$auto_ari - all$man_ari)
summary(all$auto_ami - all$man_ami)

library(viridis)

pdf(paste0(outp,"comp_ari_hist_80uniform.pdf"), width = 6, height = 3)

par(mfrow = c(1, 2), mar = c(5, 4, 1, 2) + 0.1)

hist(all$auto_ari - all$man_ari, 
     xlab = "ARI (auto - manual)", 
     main = NULL, 
     col = viridis(1), 
     border = "black", 
     cex.lab = 1.2, 
     cex.axis = 1.1)

hist(all$auto_ami - all$man_ami, 
     xlab = "AMI (auto - manual)", 
     main = NULL, 
     col = viridis(1), 
     border = "black", 
     cex.lab = 1.2, 
     cex.axis = 1.1)
dev.off()


library(tidyverse)

unidat_man_auto <- ggplot() +
  geom_point(data = all, aes(x = true_num_cell, y = auto_est_num_cell,color='Auto K'), alpha = 1, position = "jitter",size = 1.5)+
  geom_point(data = all, aes(x = true_num_cell, y = man_est_num_cell,color='Manual K'), alpha = 1, position = "jitter",size = 1.5)+
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "dashed") +  
  labs(
    x = expression(K[o]),
    y = expression(hat(K)),
    color = "Method"
  ) +
  # xlim(c(12, 17)) +
  # ylim(c(6.5, 17.5)) +
  scale_color_manual(values = c("Auto K" = "#228833", "Manual K" = "#CC79A7")) +
  theme_minimal() +
  theme(
    strip.text = element_blank(),  # Remove subplot titles
    axis.title = element_text(size = 13),  # Increase font size of axis labels
    axis.text = element_text(size = 12),   # Increase font size of axis tick marks
    legend.text = element_text(size = 12), # Increase font size of legend labels
    legend.title = element_text(size = 12)
  )

ggsave(paste0(outp,"autoSARM_comp_uniform.pdf"), plot = unidat_man_auto, width = 5, height = 3, units = "in")

########## CI
t.test(all$man_ami, all$auto_ami, paired = TRUE)
t.test(all$man_ari, all$auto_ari, paired = TRUE)
t.test(all$auto_ami, all$man_ami, paired = TRUE)
t.test(all$auto_ari, all$man_ari, paired = TRUE)

plot(all$man_est_num_cell, all$auto_est_num_cell, 
     xlab = "man",
     ylab = "auto",
     main = NULL, 
     col = viridis(1), 
     # border = "black", 
     cex.lab = 1.2, 
     cex.axis = 1.1)

unidat_man_auto <- ggplot() +
  geom_point(data = all, aes(x = man_est_num_cell, y = auto_est_num_cell), color="#228833", alpha = 1, position = "jitter",size = 1.5)+
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "dashed") +  
  labs(
    x = expression(hat(K)[manual]),
    y = expression(hat(K)[auto]),
    color = "Method"
  ) +
  # xlim(c(12, 17)) +
  # ylim(c(6.5, 17.5)) +
  # scale_color_manual(values = c("Auto K" = "#228833", "Manual K" = "#CC79A7")) +
  theme_minimal() +
  theme(
    strip.text = element_blank(),  # Remove subplot titles
    axis.title = element_text(size = 13),  # Increase font size of axis labels
    axis.text = element_text(size = 12),   # Increase font size of axis tick marks
    legend.text = element_text(size = 12), # Increase font size of legend labels
    legend.title = element_text(size = 12)
  )

ggsave(paste0(outp,"automanSARM_comp_uniform.pdf"), plot = unidat_man_auto, width = 4, height = 3, units = "in")


################ 
sarm_stats <- read.csv("./code/as/manualK_sarms_prop_SumStats15.csv")

names(sarm_stats)[3:5] <- paste0("sarm_",names(sarm_stats)[3:5]) 
names(sc_stats)[2:4] <- paste0("sc_",names(sc_stats)[2:4]) 
names(seurat_stats)[2:4] <- paste0("seurat_",names(seurat_stats)[2:4]) 

outp <- "./code/as/plt/"
library(viridis)
library(ggplot2)
tool_est <- ggplot() +
  geom_point(data = sarm_stats, aes(x = true_num_cell, y = sarm_est_num_cell, color = "Our method"), 
             alpha = 0.9, position = "jitter") +
  geom_point(data = sc_stats, aes(x = true_num_cell, y = sc_est_num_cell, color = "SC3"), 
             alpha = 0.9, position = "jitter") +
  geom_point(data = seurat_stats, aes(x = true_num_cell, y = seurat_est_num_cell, color = "Seurat"), 
             alpha = 0.9, position = "jitter") +
  scale_color_viridis_d(option = "D", name = "Tool") + # Using a discrete viridis palette
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "dashed") +  
  theme_minimal() + 
  labs(
    x = expression(K[o]),
    y = expression(hat(K)),
    color = "Tool"
  ) +
  theme(
    axis.title = element_text(size = 13),  # Increase font size of axis labels
    axis.text = element_text(size = 12),   # Increase font size of axis tick marks
    legend.text = element_text(size = 12), # Increase font size of legend labels
    legend.title = element_text(size = 12)
  )

ggsave(paste0(outp,"tool_comp_prop_estk.pdf"), plot = tool_est , width = 5, height = 3, units = "in")

## ari

tool_ari <- ggplot() +
  geom_point(data = sarm_stats, aes(x = true_num_cell, y = sarm_ari, color = "Our method"), alpha = 0.9, position = "jitter") +
  geom_point(data = sc_stats, aes(x = true_num_cell, y = sc_ari, color = "SC3"), alpha = 0.9, position = "jitter") +
  geom_point(data = seurat_stats, aes(x = true_num_cell, y = seurat_ari, color = "Seurat"), alpha = 0.9, position = "jitter") +
  scale_color_viridis_d(option = "D", name = "Tool") +
  theme_minimal() + 
  theme_minimal() + 
  labs(
    x = expression(K[o]),
    y = 'ARI',
    color = "Tool"
  ) +
  theme(
    axis.title = element_text(size = 13),  
    axis.text = element_text(size = 12),   
    legend.text = element_text(size = 12), 
    legend.title = element_text(size = 12)
  )

ggsave(paste0(outp,"tool_comp_prop_ari.pdf"), plot = tool_ari , width = 5, height = 3, units = "in")

##  ami
tool_ami <- ggplot() +
  geom_point(data = sarm_stats, aes(x = true_num_cell, y = sarm_ami, color = "Our method"), alpha = 0.9, position = "jitter",size = 1.5) +
  geom_point(data = sc_stats, aes(x = true_num_cell, y = sc_ami, color = "SC3"), alpha = 0.9, position = "jitter",size = 1.5) +
  geom_point(data = seurat_stats, aes(x = true_num_cell, y = seurat_ami, color = "Seurat"), alpha = 0.9, position = "jitter",size = 1.5) +
  scale_color_viridis_d(option = "D", name = "Tool") +
  theme_minimal() + 
  theme_minimal() + 
  labs(
    x = expression(K[o]),
    y = 'AMI',
    color = "Tool"
  ) +
  theme(
    axis.title = element_text(size = 13),  
    axis.text = element_text(size = 12),   
    legend.text = element_text(size = 12), 
    legend.title = element_text(size = 12)
  )

ggsave(paste0(outp,"tool_comp_prop_ami.pdf"), plot = tool_ami , width = 5, height = 3, units = "in")

###########################  non-unif auto vs man plot
sarm_stats_auto <- read.csv("./code/as/autoK_sarms_prop_SumStats15.csv")

sarm_stats <- read.csv("./code/as/manualK_sarms_prop_SumStats15.csv")

names(sarm_stats)[3:5] <- paste0('man_',colnames(sarm_stats)[3:5])
names(sarm_stats_auto)[3:5] <- paste0('auto_',colnames(sarm_stats_auto)[3:5])

all <- merge(sarm_stats,sarm_stats_auto,by=c('true_num_cell','obs'))

outp <- "./code/as/plt/"

library(tidyverse)

plot(all$man_est_num_cell, all$auto_est_num_cell, 
     xlab = "man",
     ylab = "auto",
     main = NULL, 
     col = viridis(1), 
     # border = "black", 
     cex.lab = 1.2, 
     cex.axis = 1.1)

non_unidat_man_auto <- ggplot() +
  geom_point(data = all, aes(x = man_est_num_cell, y = auto_est_num_cell), color="#228833", alpha = 1, position = "jitter",size = 1.5)+
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "dashed") +  
  labs(
    x = expression(hat(K)[manual]),
    y = expression(hat(K)[auto]),
    color = "Method"
  ) +
  # xlim(c(12, 17)) +
  # ylim(c(6.5, 17.5)) +
  # scale_color_manual(values = c("Auto K" = "#228833", "Manual K" = "#CC79A7")) +
  theme_minimal() +
  theme(
    strip.text = element_blank(),  # Remove subplot titles
    axis.title = element_text(size = 13),  # Increase font size of axis labels
    axis.text = element_text(size = 12),   # Increase font size of axis tick marks
    legend.text = element_text(size = 12), # Increase font size of legend labels
    legend.title = element_text(size = 12)
  )

ggsave(paste0(outp,"automanSARM_comp_prop.pdf"), plot = non_unidat_man_auto, width = 4, height = 3, units = "in")

########## CI
t.test(all$man_ami, all$auto_ami, paired = TRUE)
summary(all$man_ami- all$auto_ami)
summary(all$man_ari- all$auto_ari)

t.test(all$man_ari, all$auto_ari, paired = TRUE)

summary(all$auto_ami - all$man_ami)
summary(all$auto_ari - all$man_ari)

t.test(all$auto_ami, all$man_ami, paired = TRUE)
t.test(all$auto_ari, all$man_ari, paired = TRUE)

hist(all$auto_ari - all$man_ari, 
     xlab = "ARI (auto - manual)", 
     main = NULL, 
     col = viridis(1), 
     border = "black", 
     cex.lab = 1.2, 
     cex.axis = 1.1)

hist(all$auto_ami - all$man_ami, 
     xlab = "AMI (auto - manual)", 
     main = NULL, 
     col = viridis(1), 
     border = "black", 
     cex.lab = 1.2, 
     cex.axis = 1.1)

### get average
path <- "./code/as/"
seu <- read.csv(paste0(path,"seurat_prop_SumStats15.csv"))
sc3 <- read.csv(paste0(path, "sc3_prop_SumStats15.csv"))
sarms_manual <- read.csv(paste0(path, "manualK_sarms_prop_SumStats15.csv"))

########################### center the zeros on the auto vs. manual plots
sarm_stats_auto <- read.csv("./code/as/autoK_sarms_SumStats80_repNum.csv")

sarm_stats <- read.csv("./code/as/manualK_sarms_SumStats80_repNum.csv")

names(sarm_stats)[2:4] <- paste0('man_',colnames(sarm_stats)[2:4])
names(sarm_stats_auto)[2:4] <- paste0('auto_',colnames(sarm_stats_auto)[2:4])

all <- merge(sarm_stats,sarm_stats_auto,by=c('true_num_cell','rep'))

outp <- "./code/as/plt/"

summary(all$auto_ari - all$man_ari)
summary(all$auto_ami - all$man_ami)

library(ggplot2)
library(viridis)
library(gridExtra)  

ari_diff <- all$auto_ari - all$man_ari
ami_diff <- all$auto_ami - all$man_ami

df_ari <- data.frame(diff = ari_diff)
df_ami <- data.frame(diff = ami_diff)

binwidth_val <- 0.1

# ARI plot
p1 <- ggplot(df_ari, aes(x = diff)) +
  geom_histogram(binwidth = binwidth_val, center = 0,
                 fill = viridis(1), color = "black") +
  labs(x = "ARI (auto - manual)", y = "Count") +
  theme_minimal(base_size = 12)+
  theme(
    axis.title = element_text(size = 13),  # Increase font size of axis labels
    axis.text = element_text(size = 12),   # Increase font size of axis tick marks
    legend.text = element_text(size = 12), # Increase font size of legend labels
    legend.title = element_text(size = 12)
  )

# AMI plot
p2 <- ggplot(df_ami, aes(x = diff)) +
  geom_histogram(binwidth = binwidth_val, center = 0,
                 fill = viridis(1), color = "black") +
  labs(x = "AMI (auto - manual)", y = "Count") +
  theme_minimal(base_size = 12)+
  theme(
    axis.title = element_text(size = 13),  # Increase font size of axis labels
    axis.text = element_text(size = 12),   # Increase font size of axis tick marks
    legend.text = element_text(size = 12), # Increase font size of legend labels
    legend.title = element_text(size = 12)
  )

pdf(paste0(outp, "comp_ari_hist_80uniform_0cent.pdf"), width = 6, height = 3)
grid.arrange(p1, p2, ncol = 2)
dev.off()

# non-unif
sarm_stats_auto <- read.csv("./code/as/autoK_sarms_prop_SumStats15.csv")

sarm_stats <- read.csv("./code/as/manualK_sarms_prop_SumStats15.csv")

names(sarm_stats)[3:5] <- paste0('man_',colnames(sarm_stats)[3:5])
names(sarm_stats_auto)[3:5] <- paste0('auto_',colnames(sarm_stats_auto)[3:5])

all <- merge(sarm_stats,sarm_stats_auto,by=c('true_num_cell','obs'))

outp <- "./code/as/plt/"

ari_diff <- all$auto_ari - all$man_ari
ami_diff <- all$auto_ami - all$man_ami

df_ari <- data.frame(diff = ari_diff)
df_ami <- data.frame(diff = ami_diff)

binwidth_val <- 0.015

# ARI plot
p1 <- ggplot(df_ari, aes(x = diff)) +
  geom_histogram(binwidth = binwidth_val, center = 0,
                 fill = viridis(1), color = "black") +
  labs(x = "ARI (auto - manual)", y = "Count") +
  theme_minimal(base_size = 12)+
  theme(
    axis.title = element_text(size = 13),  # Increase font size of axis labels
    axis.text = element_text(size = 12),   # Increase font size of axis tick marks
    legend.text = element_text(size = 12), # Increase font size of legend labels
    legend.title = element_text(size = 12)
  )

# AMI plot
p2 <- ggplot(df_ami, aes(x = diff)) +
  geom_histogram(binwidth = binwidth_val, center = 0,
                 fill = viridis(1), color = "black") +
  labs(x = "AMI (auto - manual)", y = "Count") +
  theme_minimal(base_size = 12)+
  theme(
    axis.title = element_text(size = 13),  
    axis.text = element_text(size = 12),   
    legend.text = element_text(size = 12), 
    legend.title = element_text(size = 12)
  )

pdf(paste0(outp, "comp_ari_hist_prop_0cent.pdf"), width = 6, height = 3)
grid.arrange(p1, p2, ncol = 2)
dev.off()
