import os, sys, csv,re, time, random
import cv2
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from . util import *
from . contour_util import *
from . ConnectedComponents import Graph


def leading_edge_detection(img, pred_refined, resize_factor, target_clusters, binary):
	resize_width=int(img.shape[1]*resize_factor)
	resize_height=int(img.shape[0]*resize_factor)
	img_new = img.copy()
	img_new_resized = cv2.resize(img_new, (resize_width, resize_height))
	for t in target_clusters:
		target_img=(1*(pred_refined==t)).reshape(resize_height, resize_width)
		cnt_info=cv2_detect_contour((target_img==0).astype(np.uint8), apertureSize=5,L2gradient = True, all_cnt_info=True)
		cnt_info=list(filter(lambda x: (x[2] > 2000), cnt_info))
		for i in range(len(cnt_info)):
			cv2.drawContours(img_new_resized, [cnt_info[i][0]], -1, ([0, 0, 0]), thickness=2) #BGR
	return img_new_resized


def tumor_edge_core_separation(img,  binary, resize_factor, pred_refined, target_clusters, sudo_adata, res, shrink_rate=0.8):
	resize_width=int(img.shape[1]*resize_factor)
	resize_height=int(img.shape[0]*resize_factor)
	#Extract target pixels
	target_img=(1*(np.isin(pred_refined, target_clusters))).reshape(resize_height, resize_width)
	#Filter using binary filter
	binary_resized=cv2.resize(binary, (resize_width, resize_height), interpolation = cv2.INTER_AREA)
	target_img=target_img*(binary_resized!=0)
	tumor_binary=cv2.resize(target_img.astype(np.uint8), ((img.shape[1], img.shape[0])))
	tumor_index=[]
	for index, row in sudo_adata.obs.iterrows():
		x=int(row["x"])
		y=int(row["y"])
		if tumor_binary[x, y]!=0: tumor_index.append(index)
	sudo_tumor=sudo_adata[sudo_adata.obs.index.isin(tumor_index)]
	print("Running Connected Components ...")
	adj=pd.DataFrame(np.zeros([sudo_tumor.shape[0], sudo_tumor.shape[0]]), columns=sudo_tumor.obs.index.tolist(), index=sudo_tumor.obs.index.tolist())
	for index, row in sudo_tumor.obs.iterrows():
		x, y=row["x"], row["y"]
		nbr=sudo_tumor.obs[((sudo_tumor.obs["x"]==x) & (np.abs(sudo_tumor.obs["y"]-y)<=res))| ((sudo_tumor.obs["y"]==y) & (np.abs(sudo_tumor.obs["x"]-x)<=res))]
		adj.loc[index, nbr.index.tolist()]=1
	sys.setrecursionlimit(adj.shape[0])
	g = Graph(V=adj.index.tolist(), adj=adj)
	c_c = g.ConnectedComponents()
	cc_info=[]
	for c in c_c:
		cc_info.append((c,len(c)))
	cc_info = sorted(cc_info, key=lambda c: c[1], reverse=True)
	print("Running Select biggest Tumor region ...")
	#cc_info[0][0] contains the index of all superpiexls in the biggest tumor region
	sudo_tumor_sub=sudo_tumor[sudo_tumor.obs.index.isin(cc_info[0][0])]
	adj_sub=adj.loc[cc_info[0][0], cc_info[0][0]]
	adj_sub=adj_sub[np.sum(adj_sub, 1)>2]
	sudo_tumor_sub=sudo_tumor_sub[sudo_tumor_sub.obs.index.isin(adj_sub.index.tolist())]
	binary_tumor = np.zeros(img.shape[0:2]).astype(np.uint8)
	for index, row in sudo_tumor_sub.obs.iterrows():
		x=int(row["x"])
		y=int(row["y"])
		binary_tumor[int(x-res/2):int(x+res/2), int(y-res/2):int(y+res/2)]=1
	cnt_tumor=cv2_detect_contour((binary_tumor==0).astype(np.uint8), apertureSize=5,L2gradient = True)
	print("Running Core and edge separation ...")
	cnt_tumor_reduced=scale_contour(cnt_tumor, shrink_rate)
	binary_core = np.zeros(img.shape[0:2]).astype(np.uint8)
	cv2.drawContours(binary_core, [cnt_tumor_reduced], -1, (1), thickness=-1)
	binary_core=binary_core*binary_tumor
	cnt_core=cv2_detect_contour((binary_core==0).astype(np.uint8), apertureSize=5,L2gradient = True)
	print("Running Create Gene Expression adata for tumor edge vs. core ...")
	region=[]
	for index, row in sudo_tumor_sub.obs.iterrows():
		x=int(row["x"])
		y=int(row["y"])
		region.append(("edge","core",)[binary_core[x, y]])
	sudo_tumor_sub.obs["region"]=region
	return binary_tumor, binary_core, sudo_tumor_sub


def plot_tumor_edge_core(img, resize_factor, binary, binary_tumor, binary_core, color_edge=[66, 50, 225], color_core=[62, 25, 53]):
	resize_width=int(img.shape[1]*resize_factor)
	resize_height=int(img.shape[0]*resize_factor)
	white_ratio=0.5
	alpha=0.8
	ret_img=img.copy()
	cnt_tumor=cv2_detect_contour((binary_tumor==0).astype(np.uint8), apertureSize=5,L2gradient = True)
	cnt_core=cv2_detect_contour((binary_core==0).astype(np.uint8), apertureSize=5,L2gradient = True)
	ret_img[binary!=0]=ret_img[binary!=0]*(1-white_ratio)+np.array([255, 255, 255])*(white_ratio)
	#Core and edge region
	target_img = np.zeros(img.shape, dtype=np.uint8)
	cv2.drawContours(target_img, [cnt_tumor], -1, (color_edge), thickness=-1) #BGR
	cv2.drawContours(target_img, [cnt_core], -1, (color_core), thickness=-1) #BGR
	#Overlay
	ret_img[binary_tumor!=0]=ret_img[binary_tumor!=0]*(1-alpha)+target_img[binary_tumor!=0]*alpha
	ret_img=cv2.resize(ret_img, (resize_width, resize_height), interpolation = cv2.INTER_AREA)
	return ret_img


def tumor_edge_core_DE(sudo_core_edge):
	sc.tl.rank_genes_groups(sudo_core_edge, groupby="region",reference="rest",n_genes=sudo_core_edge.shape[1], method='wilcoxon')
	pvals_adj1=[i[1] for i in sudo_core_edge.uns['rank_genes_groups']["pvals_adj"]]
	pvals_adj0=[i[0] for i in sudo_core_edge.uns['rank_genes_groups']["pvals_adj"]]
	genes1=[i[1] for i in sudo_core_edge.uns['rank_genes_groups']["names"]]
	genes0=[i[0] for i in sudo_core_edge.uns['rank_genes_groups']["names"]]
	if issparse(sudo_core_edge.X):
		obs_tidy=pd.DataFrame(sudo_core_edge.X.A)
	else:
		obs_tidy=pd.DataFrame(sudo_core_edge.X)
	obs_tidy.index=sudo_core_edge.obs["region"].tolist()
	obs_tidy.columns=sudo_core_edge.var.index.tolist()
	#--------Edge enriched genes
	obs_tidy1=obs_tidy.loc[:,genes1]
	# 1. compute mean value
	mean_obs1 = obs_tidy1.groupby(level=0).mean()
	mean_all1=obs_tidy1.mean()
	# 2. compute fraction of cells having value >0
	obs_bool1 = obs_tidy1.astype(bool)
	fraction_obs1 = obs_bool1.groupby(level=0).sum() / obs_bool1.groupby(level=0).count()
	# compute fold change.
	fold_change1 = (mean_obs1.loc["edge"] / (mean_obs1.loc["core"]+ 1e-9)).values
	df1 = {'edge_genes': genes1, 
		'in_group_fraction': fraction_obs1.loc["edge"].tolist(), 
		"out_group_fraction":fraction_obs1.loc["core"].tolist(),
		"in_out_group_ratio":(fraction_obs1.loc["edge"]/fraction_obs1.loc["core"]).tolist(),
		"in_group_mean_exp": mean_obs1.loc["edge"].tolist(), 
		"out_group_mean_exp": mean_obs1.loc["core"].tolist(),
		"fold_change":fold_change1.tolist(), 
		"pvals_adj":pvals_adj1,
		"all_mean_exp": mean_all1}
	df1 = pd.DataFrame(data=df1)
	df1=df1[(df1["pvals_adj"]<0.05)]
	#--------Core enriched genes
	obs_tidy0=obs_tidy.loc[:,genes0]
	# 1. compute mean value
	mean_obs0 = obs_tidy0.groupby(level=0).mean()
	mean_all0=obs_tidy0.mean()
	# 2. compute fraction of cells having value >0
	obs_bool0 = obs_tidy0.astype(bool)
	fraction_obs0 = obs_bool0.groupby(level=0).sum() / obs_bool0.groupby(level=0).count()
	# compute fold change.
	fold_change0 = (mean_obs0.loc["core"] / (mean_obs0.loc["edge"]+ 1e-9)).values
	df0 = {'edge_genes': genes0, 
		'in_group_fraction': fraction_obs0.loc["core"].tolist(), 
		"out_group_fraction":fraction_obs0.loc["edge"].tolist(),
		"in_out_group_ratio":(fraction_obs0.loc["core"]/fraction_obs0.loc["edge"]).tolist(),
		"in_group_mean_exp": mean_obs0.loc["core"].tolist(), 
		"out_group_mean_exp": mean_obs0.loc["edge"].tolist(),
		"fold_change":fold_change0.tolist(), 
		"pvals_adj":pvals_adj0,
		"all_mean_exp": mean_all0}
	df0 = pd.DataFrame(data=df0)
	df0=df0[(df0["pvals_adj"]<0.05)]
	return df0, df1

def filter_DE_genes(df, min_all_mean_exp=0.1, min_in_out_group_ratio=1, min_in_group_fraction=0.5, min_fold_change=1.2):
	df_filtered=df.copy()
	df_filtered=df_filtered[(df_filtered["all_mean_exp"]>min_all_mean_exp) &
	                            (df_filtered["in_out_group_ratio"]>=min_in_out_group_ratio) &
	                            (df_filtered["in_group_fraction"]>min_in_group_fraction) &
	                            (df_filtered["fold_change"]>min_fold_change)]
	return df_filtered

def plot_edge_core_enrichd_genes(img, resize_factor, binary, binary_tumor, sudo_core_edge, genes, cnt_color, plot_dir, res):
	resize_width=int(img.shape[1]*resize_factor)
	resize_height=int(img.shape[0]*resize_factor)
	if issparse(sudo_core_edge.X):sudo_core_edge.X=sudo_core_edge.X.A
	res=res*1.5
	white_ratio=0.5
	for g in genes:
		sudo_core_edge.obs["exp"]=sudo_core_edge.X[:,sudo_core_edge.var.index==g]
		sudo_core_edge.obs["relexp"]=(sudo_core_edge.obs["exp"]-np.min(sudo_core_edge.obs["exp"]))/(np.max(sudo_core_edge.obs["exp"])-np.min(sudo_core_edge.obs["exp"]))
		img_new=np.zeros(img.shape, dtype=np.uint8)
		for _, row in sudo_core_edge.obs.iterrows():
			x=row["x"]
			y=row["y"]
			c=row["relexp"]
			img_new[int(x-res/2):int(x+res/2), int(y-res/2):int(y+res/2),:]=((np.array(cnt_color(int(c*255)))[0:3])*255).astype(int)
		img_new=cv2.cvtColor(img_new, cv2.COLOR_RGB2BGR)
		#Background
		ret_img=img.copy()
		ret_img[binary!=0]=ret_img[binary!=0]*(1-white_ratio)+np.array([255, 255, 255])*(white_ratio)
		#Overlay
		ret_img[binary_tumor!=0]=img_new[binary_tumor!=0]
		ret_img=cv2.resize(ret_img, (resize_width, resize_height), interpolation = cv2.INTER_AREA)
		cv2.imwrite(plot_dir+g+'.jpg', ret_img)



