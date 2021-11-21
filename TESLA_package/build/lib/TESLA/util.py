import scanpy as sc
import pandas as pd
import numpy as np
import scipy
import os
from anndata import AnnData,read_csv,read_text,read_mtx
from scipy.sparse import issparse

def prefilter_cells(adata,min_counts=None,max_counts=None,min_genes=200,max_genes=None):
    if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[0],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_genes=min_genes)[0]) if min_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_genes=max_genes)[0]) if max_genes is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_cells(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_obs(id_tmp)
    adata.raw=sc.pp.log1p(adata,copy=True) #check the rowname 
    print("the var_names of adata.raw: adata.raw.var_names.is_unique=:",adata.raw.var_names.is_unique)
   

def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)


def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2)
    adata._inplace_subset_var(id_tmp)

def relative_func(expres):
    #expres: an array counts expression for a gene
    maxd = np.max(expres) - np.min(expres)
    min_exp=np.min(expres)
    rexpr = (expres - min_exp)/maxd
    return rexpr

def plot_relative_exp(input_adata, gene, x_name, y_name,color,use_raw=False, spot_size=200000):
    adata=input_adata.copy()
    if use_raw:
        X=adata.raw.X
    else:
        X=adata.X
    if issparse(X):
        X=pd.DataFrame(X.A)
    else:
        X=pd.DataFrame(X)
    X.index=adata.obs.index
    X.columns=adata.var.index
    rexpr=relative_func(X.loc[:,gene])
    adata.obs["rexpr"]=rexpr
    fig=sc.pl.scatter(adata,x=x_name,y=y_name,color="rexpr",title=gene+"_rexpr",color_map=color,show=False,size=spot_size/adata.shape[0])
    return fig

def plot_log_exp(input_adata, gene, x_name, y_name,color,use_raw=False):
    adata=input_adata.copy()
    if use_raw:
        X=adata.X
    else:
        X=adata.raw.X
    if issparse(X):
        X=pd.DataFrame(X.A)
    else:
        X=pd.DataFrame(X)
    X.index=adata.obs.index
    X.columns=adata.var.index
    adata.obs["log"]=np.log((X.loc[:,gene]+1).tolist())
    fig=sc.pl.scatter(adata,x=x_name,y=y_name,color="log",title=gene+"_log",color_map=color,show=False,size=200000/adata.shape[0])
    return fig

def refine_clusters(pred, resize_height, resize_width, threshold, radius):
    pixel_num=pd.Series(pred).value_counts()
    clusters=pixel_num.index.tolist()
    reorder_map={}
    for i in range(pixel_num.shape[0]):
        reorder_map[clusters[i]]=i
    pred_reordered=pd.Series(pred).replace(reorder_map).to_numpy()
    pixel_num=pd.Series(pred_reordered).value_counts()
    # Number of clusters
    nLabels = len(np.unique(pred_reordered))
    # Number of main clusters
    mainLabels=(pd.Series(pred_reordered).value_counts()>=threshold).sum()
    #------------- Refine clusters ---------------------
    main_clusters=pixel_num.index[pixel_num>=threshold].tolist()
    minor_clusters=pixel_num.index[pixel_num<threshold].tolist()
    pred_reordered_img = pred_reordered.reshape( (resize_height, resize_width))
    max_x, max_y=resize_width, resize_height
    replace_map={}
    for i in minor_clusters:
        nbs=[]
        xy=np.where(pred_reordered_img==i)
        for j in range(len(xy[0])):
            x, y=xy[0][j], xy[1][j]
            nbs=nbs+pred_reordered_img[max(0,x-radius):min(max_x,x+radius+1),max(0,y-radius):min(max_y,y+radius+1)].flatten().tolist()
            nbs_num=pd.Series(nbs).value_counts()
            if sum(nbs_num.index.isin(main_clusters))>0:
                replace_map[i]=nbs_num.index[ nbs_num.index.isin(main_clusters) ][ 0 ]
    pred_refined=pd.Series(pred_reordered).replace(replace_map).to_numpy()
    return pred_refined
