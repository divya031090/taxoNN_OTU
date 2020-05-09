data <- read.csv("T2D_up.csv", header = TRUE, stringsAsFactors = FALSE,sep=",")

#data <- read.csv("OTU_orig.csv", header = FALSE, stringsAsFactors = FALSE,sep=",")



head(data)
rownames(data) <- data$OTUname
#rownames(data) <- data$X.OTUname
data$OTUname<- NULL
#data$X.OTUname <- NULL
head(data)
#data_transpose <- as.data.frame(t(as.matrix(data)))
a=data.matrix(data[52:116,1:502], rownames.force = NA)

corr_mat=cor(t(a),method="spearman",use = "complete.obs")

library(reshape2)
co=melt(corr_mat)
head(co)

corr_sum<-rowSums(corr_mat)
corr_sum_sorted<-sort(corr_sum,decreasing=TRUE)

corr_mat_sorted<-corr_mat[order(corr_sum, decreasing = TRUE),order(corr_sum, decreasing = TRUE)]
corr_mat_sorted_sum<-rowSums(corr_mat_sorted)
#new<-order(corr_sum, decreasing = TRUE)
#o=co[1:1000,1:3]
#a<-order(corr_sum)

library(reshape2)
co=melt(t(corr_mat_sorted))
head(co)
#co=co[1:1000,]

library(ggplot2)
library(scales) # for muted function
ggplot(co, aes(Var1, Var2)) + # x and y axes => Var1 and Var2
  geom_tile(aes(fill = value)) + # background colours are mapped according to the value column
  geom_text(aes(fill = co$value, label = round(co$value, 2)),size=2) + # write the values
  scale_fill_gradient2(low = muted("darkred"), 
                       mid = "white", 
                       high = muted("blue"), 
                       midpoint = 0) + # determine the colour
  theme(panel.grid.major.x=element_blank(), #no gridlines
        panel.grid.minor.x=element_blank(), 
        panel.grid.major.y=element_blank(), 
        panel.grid.minor.y=element_blank(),
        panel.background=element_rect(fill="white"), # background=white
        axis.text.x = element_text(angle=90, hjust = 1,vjust=1,size = 10,face = "bold"),
        plot.title = element_text(size=10,face="bold"),
        axis.text.y = element_text(size = 10,face = "bold")) + 
  ggtitle("Correlation Plot") + 
  theme(legend.title=element_text(face="bold", size=10)) + 
  scale_x_discrete(name="") +
  scale_y_discrete(name="") +
  labs(fill="Corr. Coef.")





