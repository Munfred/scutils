
### FILENAME CHANGE BEFORE RUNNING ###
# sbatch -A SternbergGroup --gres gpu:1 --mem=64000 -t 15:00:00 --ntasks 10 --nodes 1 --job-name "F3samples.100umis" --wrap "python F3samples.100umis.py"
print('Starting...')
import plotnine as p
import scvi
import IPython
import warnings; warnings.simplefilter('ignore')
import os
import numpy as np
import pandas as pd
import torch
import anndata
from tqdm import tqdm
from umap import UMAP
from anndata import AnnData
from umap import UMAP
from fastTSNE import TSNE
from fastTSNE.callbacks import ErrorLogger
import datetime
import os
print('FINISHED IMPORTS')
model_name=os.path.basename(__file__)
model_name= model_name.strip('.py')
print('MODEL NAME:', model_name)

adata_file='../adata.h5ad'
print('READING: ', adata_file )
adata = anndata.read(adata_file)
adata.obs['experiment']=adata.obs['shortname']
print(adata)

### FILTER MINIMUM GENES AND CELLS
min_gene_counts = 200
print('REMOVING GENES WITH LESS THAN ', min_gene_counts, ' COUNTS')
adata = adata[:, adata.X.sum(0) > min_gene_counts].copy()
print(adata)

min_cell_umi_counts = 100
print('REMOVING CELLS WITH LESS THAN ', min_cell_umi_counts, ' UMI COUNTS')
adata = adata[adata.X.sum(1) > min_cell_umi_counts].copy()
print(adata)
###


### DEFINE PLOTTING FUNCTIONS

def derplot(adata=None, filename='derplot',embedding='tsne',feature='sample_type_tech',
            size=(12, 12), save=False, draw=False, psize=1):
    start = datetime.datetime.now()
    p.options.figure_size = size
    savename=filename +'.' + embedding + '.' + feature + '.derplot.png'
    print(start.strftime("%H:%M:%S"), 'Starting ... \t',savename, )
    p.theme_set(p.theme_classic())
    pt = \
    p.ggplot(p.aes(embedding +'0', embedding + '1', color=feature), adata.obs) \
        + p.geom_point(size=psize, alpha = 1, stroke = 0 ) \
        + p.guides(color = p.guide_legend(override_aes={'size': 15})) 
    
    if save: pt.save(savename, format='png', dpi=200)
    end = datetime.datetime.now()
    delta = end-start
    print(start.strftime("%H:%M:%S"),   str(int(delta.total_seconds())), 's to make: \t', savename)
    

def wraplot(adata=None, filename='wraplot',embedding='tsne',feature='sample_type_tech',
            size=(12, 12), color=None, save=False, draw=False, psize=1):
    start = datetime.datetime.now()
    p.options.figure_size = size
    savename = filename +'.' + embedding + '.' + feature + '.' + str(color) + '.png'
    if color==None: 
        color=feature
        savename = filename +'.' + embedding + '.' + feature + '.wraplot.png'
    print(start.strftime("%H:%M:%S"), 'Starting ... \t',savename, )
    
    pt = (
        p.ggplot(p.aes(x= embedding+'0', y=embedding+'1', color=color), adata.obs) 
        + p.geom_point(color='lightgrey', shape = '.', data=adata.obs.drop(feature, axis = 1)) 
        + p.geom_point(shape='.', size=psize, alpha = 1, stroke = 0 ) 
        + p.theme_minimal()
        + p.facet_wrap('~' + feature )
        + p.guides(color = p.guide_legend(override_aes={'size': 10}))
    )
    
    if save: pt.save(savename, format='png', dpi=200)
    
    end = datetime.datetime.now()
    delta = end-start
    print(start.strftime("%H:%M:%S"),   str(int(delta.total_seconds())), 's to make: \t', savename)
   
    
    
### PREPARE DATA
adata.layers["counts"] = adata.X.copy() # preserve counts
scvi.data.setup_anndata(adata, layer="counts", batch_key='experiment_id') 

### DEFINE MODEL
model = scvi.model.SCVI(adata,  
                        n_hidden=128, 
                        n_layers=2,
                        gene_likelihood='nb',
                        dispersion='gene-batch'
                        )

# MODEL TRAINING
model.train(frequency=1, 
            n_epochs = 15,
            lr=2e-3,
            n_epochs_kl_warmup=None)

train_test_results = pd.DataFrame(model.trainer.history).rename(columns={'elbo_train_set':'Train', 'elbo_test_set':'Test'})
print(train_test_results)
train_test_results.to_csv('train_test_results.csv')

### SAVE MODEL
print('SAVING MODEL: ', model_name)
model.save(model_name)

### LOAD MODEL
# model = scvi.model.SCVI.load(adata, model_name, use_cuda=True)
print('SAVED MODEL')

### GET LATENT
latent = model.get_latent_representation() # get latent

### MAKE TSNE
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
adata.obs.to_csv('obs.csv')
adata.write(model_name+'.h5ad')


### MAKE PLOTS
derplot(adata=adata, filename=model_name, embedding='tsne',feature='experiment_type',size=(10, 10), save=True)
derplot(adata=adata, filename=model_name, embedding='tsne',feature='experiment',size=(10, 10), save=True)

wraplot(adata=adata, filename=model_name, embedding='tsne',feature='experiment',size=(10, 10), save=True)
wraplot(adata=adata, filename=model_name, embedding='tsne',feature='experiment_type',size=(10, 10), save=True)

adata.obs['log10umi']=np.log10(np.squeeze(np.asarray(adata.X.sum(1))))
wraplot(adata=adata, filename=model_name, embedding='tsne',feature='experiment',size=(10, 10), save=True, color = 'log10umi')
wraplot(adata=adata, filename=model_name, embedding='tsne',feature='experiment_type',size=(10, 10), save=True, color = 'log10umi')
