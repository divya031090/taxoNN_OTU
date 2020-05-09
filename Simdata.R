data <- read.csv("OTU_orig.csv", header = TRUE, stringsAsFactors = FALSE,sep=",")
noise<-read.csv("Noise.csv", header = FALSE, stringsAsFactors = FALSE,sep=",")
env_vars<-read.csv("Env_var.csv", header = TRUE, stringsAsFactors = FALSE,sep=",")


zeroes<-rowSums(data == 0)



nonzero<-which(data[1,]!=0,arr.ind = T)

nonzerodata_firstrow<-data[1,]
nonzerodata_firstrow[1,nonzero[1:(length(nonzero[,2])/2),2]]<-data[1, nonzero[1:(length(nonzero[,2])/2),2]]+noise[1,1]


a<-(length(nonzero[,2])/2)+1
b<-length(nonzero[,2])
nonzerodata_firstrow[1, nonzero[a:b,2]]<-data[1, nonzero[a:b,2]]-noise[1,1]


#applying on all rows producing new values
data_new<-data
for (j in 1:50)
{
for (i in 1:nrow(data))
{
  nonzero<-which(data[i,]!=0,arr.ind = T)
  
  nonzerodata<-data[i,]
  nonzerodata[1,nonzero[1:(length(nonzero[,2])/2),2]]<-data[i, nonzero[1:(length(nonzero[,2])/2),2]]+noise[i,1]
  
  
  a<-(length(nonzero[,2])/2)+1
  b<-length(nonzero[,2])
  nonzerodata[1, nonzero[a:b,2]]<-data[i, nonzero[a:b,2]]-noise[i,1]
  data_new<-rbind(data_new,nonzerodata)
  
}
}

choose_32<-data_new[, sample(ncol(data_new), 32)]
choose_3<-choose_32[, sample(ncol(choose_32), 3)]

beta<-seq(1, 2, length.out = 32)
beta_int<-c(1,1.5,2)

m <- matrix(0, ncol = 1, nrow = nrow(choose_32))
sum_beta<-as.data.frame(m)
for (k in 1:32)
{
sum_beta[k,1]<-sum_beta[k,1]+(beta[k]*sum(choose_32[,k]))
}


p_exp<-exp(-2.5+sum_beta+(beta_int[1]*choose_3[,1]*choose_3[,2])+(beta_int[2]*choose_3[,2]*choose_3[,3])+(beta_int[3]*choose_3[,1]*choose_3[,3]))

p<-p_exp/(1+p_exp)

colnames(p)<-c("prob")


a<-which(p>=0.0758583,arr.ind = T)
cases<-data_new[a[,1],]
b<-which(p<=0.0758583,arr.ind = T)
controls<-data_new[b[,1],]


choose_200_cases<-cases[sample(nrow(cases), 200), ]
choose_200_controls<-cases[sample(nrow(controls), 200), ]


