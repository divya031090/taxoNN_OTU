# TaxoNN_OTU

An ensemble learning based method for training Convolutional Neural Networks on OTU data after stratification based on phyla. This is a deep learning methodology to arrange OTU data for CNN modelling based on similarity between OTUs in phylum level of the taxonomy tree.


Three datasets are used: 1) Simulation study 2) T2D study by Qin et al., 2012 and 3) Cirrhosis study by Qin et al., 2014. The files NN_Sim.py, NN_T2D.py and NN_Cirr.py are the main files. Relative abundance in OTUs are present in rows for each individual in the files Sim_OTU.csv, T2D_OTU.csv and Cirr_OTU.csv. 
The datasets are stored in T2D.zip, Cirrhosis.zip and Simulation Data.zip. 

**Prerequisites**

1.	Python 2.7
2.	CUDA
3.	cuDNN
4.	Conda
5.	TensorFlow 
6.	NumPy pandas 
7.	Keras

**Citation**

If you find our work useful in your research, please cite our work:

**TaxoNN: Ensemble of Neural Networks on Stratified Microbiome Data for Disease Prediction, Bioinformatics, May 2020**

Divya Sharma, Andrew D Paterson, Wei Xu, TaxoNN: ensemble of neural networks on stratified microbiome data for disease prediction, Bioinformatics, Volume 36, Issue 17, 1 September 2020, Pages 4544–4550, https://doi.org/10.1093/bioinformatics/btaa542

**Bibtex:**
 
  @article{sharma2020taxonn,
   title={TaxoNN: Ensemble of Neural Networks on Stratified Microbiome Data for Disease Prediction},
   author={Sharma, Divya and Paterson, Andrew D and Xu, Wei},  
   journal={Bioinformatics},  
   volume = {36},  
   number = {17},  
   pages = {4544-4550}, 
   year={2020},  
   month = {05}
   }
