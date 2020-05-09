

data_frame <- read.csv("cirrhosis_boxplot.csv", header = FALSE, stringsAsFactors = FALSE)
#data_frame_t<-as.data.frame(t((data_frame)))
p <- ggplot(data_frame, aes(data_frame$V1,data_frame$V2))

q<-          p + geom_boxplot(outlier.alpha = 0.5)
          q+   ylim(0,0.1)
            

  #phylum box plot       
          data_frame <- read.csv("T2D_phylumboxplot.csv", header = FALSE, stringsAsFactors = FALSE)
          #data_frame_t<-as.data.frame(t((data_frame)))
          ggplot(data_frame, aes(data_frame$V1,data_frame$V2,fill = data_frame$V1))+ xlab("Phylum")+ ylab("Relative abundance %")+
          #theme(legend.position="none")+
            #theme(axis.text=element_text(size=12),
             #       axis.title=element_text(size=14,face="bold"))+
            theme(axis.text.x = element_text(angle = 90, hjust = 1, size=8,face="bold" )) +
          geom_boxplot(alpha = 0.5) + scale_fill_brewer(palette="Paired")
          
          data_frame <- read.csv("Cirrhosis_phylumboxplot.csv", header = FALSE, stringsAsFactors = FALSE)
          #data_frame_t<-as.data.frame(t((data_frame)))
          ggplot(data_frame, aes(data_frame$V1,data_frame$V2,fill = data_frame$V1,))+ xlab("Phylum")+ ylab("Relative abundance %")+
            #theme(legend.position="none")+
            theme(axis.text.x = element_text(angle = 90, hjust = 1, size=8,face="bold" )) +
            geom_boxplot(alpha = 0.5) + scale_fill_brewer(palette="Paired")
          
          
  #genus boxplot T2D
          
          data <- read.csv("T2D_clusterwise_heatmap.csv", header = FALSE, stringsAsFactors = FALSE,sep=",")
          #cluster_actino
          a<-(data[3:33,1:346])
          
          a$group <- a$V2
          a.m <- melt(a[1:31,3:347], id.vars = "group")
          ggplot(a.m, aes(group, value)) + 
             xlab("Genus")+ ylab("Relative abundance %")+
            theme(axis.text.x = element_text(angle = 90, hjust = 1, size=10,face="bold")) +
            geom_boxplot() + scale_fill_brewer(palette="Dark2")
          
          
          #cluster_firmi
          a<-(data[62:129,1:346])
          
          a$group <- a$V2
          a.m <- melt(a[1:68,3:347], id.vars = "group")
          ggplot(a.m, aes(group, value)) + 
            xlab("Genus")+ ylab("Relative abundance %")+
            theme(axis.text.x = element_text(angle = 90, hjust = 1,size=10,face="bold")) +
          geom_boxplot() + scale_fill_brewer(palette="Dark2")
          
          
          
         #cluster_proteo
          a<-(data[135:188,1:346])
          
          a$group <- a$V2
          a.m <- melt(a[1:53,3:347], id.vars = "group")
          ggplot(a.m, aes(group, value)) + 
            xlab("Genus")+ ylab("Relative abundance %")+
            theme(axis.text.x = element_text(angle = 90, hjust = 1,size=10,face="bold")) +
            geom_boxplot() + scale_fill_brewer(palette="Dark2")
          
          
          #genus boxplot Cirr        
          
          data <- read.csv("/Users/divvi/Documents/postdoc/Analysis_OTU/Analysis_ROC/Cirr_clusterwise_heatmap.csv", header = FALSE, stringsAsFactors = FALSE,sep=",")
          #cluster_actino
          a<-(data[2:26,1:234])
          
          a$group <- a$V2
          a.m <- melt(a[1:25,3:235], id.vars = "group")
          ggplot(a.m, aes(group, value)) + 
            xlab("Genus")+ ylab("Relative abundance %")+
            theme(axis.text.x = element_text(angle = 90, hjust = 1,size=10,face="bold")) +
            geom_boxplot() + scale_fill_brewer(palette="Dark2")
          
          
          #cluster_firmi
          a<-(data[55:122,1:234])
          
          a$group <- a$V2
          a.m <- melt(a[1:67,3:235], id.vars = "group")
          ggplot(a.m, aes(group, value)) + 
            xlab("Genus")+ ylab("Relative abundance %")+
            theme(axis.text.x = element_text(angle = 90, hjust = 1,size=10,face="bold")) +
            geom_boxplot() + scale_fill_brewer(palette="Dark2")
          
          
          
          #cluster_proteo
          a<-(data[127:178,1:234])
          
          a$group <- a$V2
          a.m <- melt(a[1:51,3:235], id.vars = "group")
          ggplot(a.m, aes(group, value)) + 
            xlab("Genus")+ ylab("Relative abundance %")+
            theme(axis.text.x = element_text(angle = 90, hjust = 1,size=10,face="bold")) +
            geom_boxplot() + scale_fill_brewer(palette="Dark2")
          
          