data <- read.csv("OTU_orig.csv", header = TRUE, stringsAsFactors = FALSE,sep=",")
data_cirr<-read.csv('Cirrhosis_OTU.csv',header = TRUE, stringsAsFactors = FALSE,sep=",")
data_T2D<-read.csv('T2D_OTU.csv',header = TRUE, stringsAsFactors = FALSE,sep=",")


data_cluster_1<-data[ , grepl( "p__Actino" , colnames( data) ) ]
data_cluster_2<-data[ , grepl( "p__Proteo" , colnames( data) ) ]
data_cluster_3<-data[ , grepl( "p__Firmicutes" , colnames( data) ) ]


data_cluster_1<-data_cirr[ , grepl( "p__Actino" , colnames( data_cirr) ) ]
data_cluster_2<-data_cirr[ , grepl( "p__Proteo" , colnames( data_cirr) ) ]
data_cluster_3<-data_cirr[ , grepl( "p__Firmicutes" , colnames( data_cirr) ) ]

data_cluster_1<-data_T2D[ , grepl( "p__Actino" , colnames( data_T2D) ) ]
data_cluster_2<-data_T2D[ , grepl( "p__Proteo" , colnames( data_T2D) ) ]
data_cluster_3<-data_T2D[ , grepl( "p__Firmicutes" , colnames( data_T2D) ) ]


