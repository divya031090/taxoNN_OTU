OTU_data_sim<-read.csv('OTU_orig.csv',header = TRUE, stringsAsFactors = FALSE,sep=",")


data <- read.csv("T2D_clusterwise_heatmap.csv", header = FALSE, stringsAsFactors = FALSE,sep=",")

#data <- read.csv("Cirr_clusterwise_heatmap.csv", header = FALSE, stringsAsFactors = FALSE,sep=",")




T2D_data_firmicutes=data.matrix(data[62:129,3:346], rownames.force = NA)

T2D_data_firmicutes=data.matrix(data[62:129,3:346], rownames.force = NA)

T2D_data_firmicutes_transpose<-(T2D_data_firmicutes)




library("pdist")
d <- pdist(T2D_data_firmicutes_transpose,T2D_data_firmicutes_transpose)
dis_mat<-matrix(unlist(d@dist),ncol=68,byrow=T)
#totalDist = colSums (dis_mat, na.rm = FALSE)
dist_OTUs<-dis_mat
dist_OTUs<-NULL

totalDist = rowMeans(dis_mat)

which(totalDist == min(totalDist), arr.ind = TRUE)


#distance of all OTUs from medoid

for (i in 1:68)
{
dist_OTUs[i]<-(dist(rbind(T2D_data_firmicutes[i,],T2D_data_firmicutes[59,])))
}

dis_OTU<-as.data.frame(as.numeric(dist_OTUs))
dis_OTU$ID<-1:nrow(dis_OTU) 
sorted_dis<-dis_OTU[order(dis_OTU$`as.numeric(dist_OTUs)`),]


#correlation based ordering

corr_mat=cor(t(T2D_data_firmicutes),method="spearman",use = "complete.obs")




corr_sum<-rowSums(corr_mat)
corr_sum_df<-as.data.frame(as.numeric(corr_sum))
corr_sum_df$ID<-1:nrow(corr_sum_df) 
corr_sum_sorted<-corr_sum_df[order(corr_sum_df$`as.numeric(corr_sum)`),]

corr_mat=cor(t(T2D_data_firmicutes),method="spearman",use = "complete.obs")

corr_mat=cor(t(corr_sum_sorted$ID), t(sorted_dis$ID),  method = c("spearman"))


