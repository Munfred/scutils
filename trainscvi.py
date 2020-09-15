### FILENAME CHANGE BEFORE RUNNING ###
# sbatch -A SternbergGroup --mem=64000 -t 15:00:00 --ntasks 10 --nodes 1 --job-name "taylorpatrick" --wrap "python taylorpatrick.py"

import sys


model_name= 'taylorpatrick.0915'
adata_file='../taylorpatrick.h5ad'
######################################

import warnings; warnings.simplefilter('ignore')
import os
import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
from scvi.dataset import GeneExpressionDataset
from scvi.models import VAE
from scvi.inference import UnsupervisedTrainer
import torch
import anndata
import scvi 

import plotly.express as px
import plotly.graph_objects as go
from anndata import AnnData
from umap import UMAP
from fastTSNE import TSNE
from fastTSNE.callbacks import ErrorLogger
import plotnine as p

##### 
print('Reading adata...')
adata = anndata.read(adata_file)
#################### vvvvvv SOUP FILTER vvvvvv #############################
print(adata)
print('🥣🥣🥣🥣🥣🥣 FILTERING SOUP GENES 🥣🥣🥣🥣🥣🥣🥣🥣')
soupgenes = pd.read_csv('../taylorsoupgenes.csv')
adata.var['wbgene']=adata.var.index.values
adata.var['nosoup']= ~pd.Series(adata.var.wbgene).isin(soupgenes.gene_id.values).values
adata.var['soup']= pd.Series(adata.var.wbgene).isin(soupgenes.gene_id.values).values
adata=adata[:,adata.var.nosoup].copy()
print(adata)
#################### ^^^^^ SOUP FILTER ^^^^^^^ #############################

adata.obs['library_id_codes'] =  adata.obs['experiment'].cat.codes


print(adata)

gene_dataset = scvi.dataset.AnnDatasetFromAnnData(adata,
                                    ctype_label='cluster_label',
                                    batch_label='library_id_codes'
                                  )
print('**** FILTERING GENES WITH 0 COUNTS IN AT LEAST ONE BATCH ****')
gene_dataset.filter_genes_by_count(min_count = 1, per_batch=False)
# print('**** FILTERING GENES WITH <100 COUNTS IN AT LEAST ONE BATCH ****')
gene_dataset.filter_genes_by_count(min_count = 100, per_batch=False)
gene_dataset.filter_cells_by_count()
print(gene_dataset)
print('Number of batches:', gene_dataset.n_batches)
print('Selecting genes ...')
gene_dataset.subsample_genes(1000, batch_correction= True,mode="poisson_zeros")
exit
vae_params={
    'n_input':gene_dataset.nb_genes, 
    'n_batch':gene_dataset.n_batches, 
    'dispersion':'gene-batch', 
    'n_layers':1, 
    'n_hidden':128,
    'reconstruction_loss':'nb'   
    }
print(vae_params)
with open(model_name+'.json', 'w') as json_file:
    json.dump(vae_params, json_file)
    
# for this dataset 10 epochs is sufficient 
n_epochs =100
lr = 1e-3
use_cuda = False # we are loading a CPU trained model so this should be false

# set the VAE to perform batch correction
vae = VAE(**vae_params)

trainer = UnsupervisedTrainer(
    vae,
    gene_dataset,
    train_size=0.75, # number between 0 and 1, default 0.8
    use_cuda=use_cuda,
    frequency=1,
#     n_epochs_kl_warmup = None,
)

# check if a previously trained model already exists, if yes load it
save_path='./'


full_file_save_path = os.path.join(save_path, model_name + '.vae')

if os.path.isfile(full_file_save_path):
    print('Found file: ', model_name)
    trainer.model.load_state_dict(torch.load(full_file_save_path))
    trainer.model.eval()
else:
    print('Starting training for new model:', model_name)
    trainer.train(n_epochs=n_epochs, lr=lr)
    torch.save(trainer.model.state_dict(), full_file_save_path)

    train_test_results = pd.DataFrame(trainer.history).rename(columns={'elbo_train_set':'Train', 'elbo_test_set':'Test'})

    train_test_results.to_csv(model_name + '.training.csv')
    print(train_test_results)
    
######## Get the posterior ########


full_posterior = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
latent, batch_indices, labels = full_posterior.sequential().get_latent()
batch_indices = batch_indices.ravel()
full_posterior.save_posterior(model_name+'.posterior')

#replace adata with the one from the saved posterior
adata=anndata.read(model_name+'.posterior/anndata_dataset.h5ad')
### Make t-SNE ###
tsne = TSNE(negative_gradient_method='fft',
            initialization='random',
            neighbors='approx',
            n_iter=2000,
            callbacks=ErrorLogger(),
            callbacks_every_iters=100,
            n_jobs=-1, 
            learning_rate=300)


latent_tsne = tsne.fit(np.squeeze(np.asarray(latent)))


adata.obs['tsne1']=latent_tsne[:,1]
adata.obs['tsne0']=latent_tsne[:,0]


### Make UMAP ###
latent_umap = UMAP(spread=2).fit_transform(latent)


adata.obs['umap1']=latent_umap[:,1]
adata.obs['umap0']=latent_umap[:,0]

adata.obs['log10umi']=np.log10(np.squeeze(np.asarray(adata.X.sum(1))))

adata.obs.to_csv(model_name+'.obs.csv')
print('saved adata!')

### TSNES
wraplot(adata=adata, filename=model_name, embedding='tsne',feature='experiment',size=(12, 12), save=True)
derplot(adata=adata, filename=model_name, embedding='tsne',feature='experiment',size=(12, 12), save=True)


### UMAPS
wraplot(adata=adata, filename=model_name, embedding='umap',feature='experiment',size=(12, 12), save=True)
derplot(adata=adata, filename=model_name, embedding='umap',feature='experiment',size=(12, 12), save=True)

### Depth heatmaps
wraplot(adata=adata, filename=model_name, embedding='tsne',feature='experiment',size=(12, 12), save=True, color = 'log10umi')
wraplot(adata=adata, filename=model_name, embedding='umap',feature='experiment',size=(12, 12), save=True, color = 'log10umi')


derplot(adata=adata, filename=model_name, embedding='tsne',feature='study',size=(12, 12), save=True)
wraplot(adata=adata, filename=model_name, embedding='tsne',feature='study',size=(12, 6), save=True)
derplot(adata=adata, filename=model_name, embedding='umap',feature='study',size=(12, 12), save=True)
wraplot(adata=adata, filename=model_name, embedding='umap',feature='study',size=(12, 6), save=True)
derplot(adata=adata, filename=model_name, embedding='tsne',feature='cell_type',size=(12, 12), save=True)
derplot(adata=adata, filename=model_name, embedding='umap',feature='cell_type',size=(12, 12), save=True)
