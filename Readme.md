# Multi-Translation fingerprint

Multi-Translation (MT)  is a Transformer-based molecular representation learning model which translated SMILES into two binary molecule fingerprints (PubChem fingerprints and pharmacophore fingerprint), SMILES and InChI. Latent representations MT-FP were extracted from the feature output by the encoder and here were utilized to conduct various molecule property prediction tasks and drug virtual screening for NLRP3 inflammasome.  

# Requirement

Programs were run in the linux operating system Ubuntu 22.04 LTS with python 3.8. This project requires the following main libraries and details can be seen in "requirements.txt" which lists all the libraries in our environment. 

NumPy
Pandas
Pytorch
tqdm
RDKit
DeepChem
ChemBench

# Dataset

Canonical SMILES of 10 million molecules were filtered according to specific rules and used to pretrained  the model. Data source Link:[https://huggingface.co/datasets/sagawa/pubchem-10m-canonicalized](https://huggingface.co/datasets/sagawa/pubchem-10m-canonicalized)

# Training

Demo provides an example for data preparation("DemoData-pretreat.ipynb") and traning("translation_smi4decoder_with_mask.py"). Click the Run button to run the program.

# Training

Files under “train” are codes for training. After data preparing (BigData-pretreat.ipynb), run:

```python
$ python train/translation_smi4decoder_with_mask.py
```

Pre-trained model are provided—”trfm_new_98_10000.pkl”.

# Translation Test
The preparation of test-set data and the translation results can be seen under the file “translation_test”.

# Performance comparison of different fingerprints
The acquisition of molecular fingerprints and the test results can be seen under the file “test”. And results of MT-FP can be seen in file “MT_performance.ipynb” .

# Results of t-SNE
MT-FPs were visualized and the results can be found under the file “interpret”.

# Drug Screening
Data and codes for the screening of NLRP3 inflammasome inhibitors can be found under the file “drug_screening”.