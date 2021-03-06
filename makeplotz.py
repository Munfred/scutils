### EXAMPLE JOB SUBMISISON ###
# sbatch -A SternbergGroup  --gres gpu --mem=64000 -t 15:00:00 --ntasks 10 --nodes 1 --job-name "bcbg" --wrap "python bcbg.py"


### FILENAME CHANGE BEFORE RUNNING ###
model_name= 'blah'
adata_file='../blah.h5ad'

######################################
import sys
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
import datetime
import plotly.express as px
import plotly.graph_objects as go
from anndata import AnnData
from umap import UMAP
from fastTSNE import TSNE
from fastTSNE.callbacks import ErrorLogger
import plotnine as p

print('Starting makeplotz with model:', model_name)

##### PLOTTING FUNCTIONS ###### 

def isnotebook():
    # return false if not running on a notebook to avoid drawing and wasting time
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
def derplot(adata=None, filename='derplot',embedding='tsne',feature='sample_type_tech',
            size=(12, 12), save=False, draw=False, psize=3):
    start = datetime.datetime.now()
    
    p.options.figure_size = size
    savename=filename +'.' + embedding + '.' + feature + '.png'
    print(start.strftime("%H:%M:%S"), 'Starting ... \t',savename, )
    p.theme_set(p.theme_classic())
    pt = \
    p.ggplot(p.aes(embedding +'0', embedding + '1', color=feature), adata.obs) \
        + p.geom_point(size=psize, alpha = 1, stroke = 0 ) \
        + p.guides(color = p.guide_legend(override_aes={'size': 15})) 
    
    if isnotebook() and draw: pt.draw()
    if save: pt.save(savename, format='png', dpi=200)
        
    end = datetime.datetime.now()
    delta = end-start
    print(start.strftime("%H:%M:%S"),   str(int(delta.total_seconds())), 's to make: \t', savename)
    
    return(pt)


def wraplot(adata=None, filename='wraplot',embedding='tsne',feature='sample_type_tech',
            size=(12, 12), color=None, save=False, draw=False, psize=3):
    start = datetime.datetime.now()
    p.options.figure_size = size
    savename = filename +'.' + embedding + '.' + feature + '.' + str(color) + '.png'
    if color==None: 
        color=feature
        savename = filename +'.' + embedding + '.' + feature + '.png'
    print(start.strftime("%H:%M:%S"), 'Starting ... \t',savename, )
    
    pt = (
        p.ggplot(p.aes(x= embedding+'0', y=embedding+'1', color=color), adata.obs) 
        + p.geom_point(color='lightgrey', shape = '.', data=adata.obs.drop(feature, axis = 1)) 
        + p.geom_point(shape='.', size=psize, alpha = 1, stroke = 0 ) 
        + p.theme_minimal()
        + p.facet_wrap('~' + feature )
        + p.guides(color = p.guide_legend(override_aes={'size': 10}))
    )
    
    if isnotebook() and draw: pt.draw()
    if save: pt.save(savename, format='png', dpi=200)
    
    end = datetime.datetime.now()
    delta = end-start
    print(start.strftime("%H:%M:%S"),   str(int(delta.total_seconds())), 's to make: \t', savename)
    
    return(pt)
