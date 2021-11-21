import os,csv,re, time
import cv2
import pandas as pd
import numpy as np
from . util import *
from . contour_util import *
from . calculate_dis import *

def imputation(img, raw, cnt, genes, shape="None", res=50, s=1, k=2, num_nbs=10):
	binary=np.zeros((img.shape[0:2]), dtype=np.uint8)
	cv2.drawContours(binary, [cnt], -1, (1), thickness=-1)
	#Enlarged filter
	cnt_enlarged = scale_contour(cnt, 1.05)
	binary_enlarged = np.zeros(img.shape[0:2])
	cv2.drawContours(binary_enlarged, [cnt_enlarged], -1, (1), thickness=-1)
	x_max, y_max=img.shape[0], img.shape[1]
	x_list=list(range(int(res), x_max, int(res)))
	y_list=list(range(int(res), y_max, int(res)))
	x=np.repeat(x_list,len(y_list)).tolist()
	y=y_list*len(x_list)
	sudo=pd.DataFrame({"x":x, "y": y})
	sudo=sudo[sudo.index.isin([i for i in sudo.index if (binary_enlarged[sudo.x[i], sudo.y[i]]!=0)])]
	b=res
	sudo["color"]=extract_color(x_pixel=sudo.x.tolist(), y_pixel=sudo.y.tolist(), image=img, beta=b, RGB=True)
	z_scale=np.max([np.std(sudo.x), np.std(sudo.y)])*s
	sudo["z"]=(sudo["color"]-np.mean(sudo["color"]))/np.std(sudo["color"])*z_scale
	sudo=sudo.reset_index(drop=True)
	#------------------------------------Known points---------------------------------#
	known_adata=raw[:, raw.var.index.isin(genes)]
	known_adata.obs["x"]=known_adata.obs["pixel_x"]
	known_adata.obs["y"]=known_adata.obs["pixel_y"]
	known_adata.obs["color"]=extract_color(x_pixel=known_adata.obs["pixel_x"].astype(int).tolist(), y_pixel=known_adata.obs["pixel_y"].astype(int).tolist(), image=img, beta=b, RGB=False)
	known_adata.obs["z"]=(known_adata.obs["color"]-np.mean(known_adata.obs["color"]))/np.std(known_adata.obs["color"])*z_scale
	#-----------------------Distance matrix between sudo and known points-------------#
	start_time = time.time()
	dis=np.zeros((sudo.shape[0],known_adata.shape[0]))
	x_sudo, y_sudo, z_sudo=sudo["x"].values, sudo["y"].values, sudo["z"].values
	x_known, y_known, z_known=known_adata.obs["x"].values, known_adata.obs["y"].values, known_adata.obs["z"].values
	print("Total number of sudo points: ", sudo.shape[0])
	for i in range(sudo.shape[0]):
		if i%1000==0:print("Calculating spot", i)
		cord1=np.array([x_sudo[i], y_sudo[i], z_sudo[i]])
		for j in range(known_adata.shape[0]):
			cord2=np.array([x_known[j], y_known[j], z_known[j]])
			dis[i][j]=distance(cord1, cord2)
	print("--- %s seconds ---" % (time.time() - start_time))
	dis=pd.DataFrame(dis, index=sudo.index, columns=known_adata.obs.index)
	#-------------------------Fill gene expression using nbs---------------------------#
	sudo_adata=AnnData(np.zeros((sudo.shape[0], len(genes))))
	sudo_adata.obs=sudo
	sudo_adata.var=known_adata.var
	#Impute using all spots, weighted
	for i in range(sudo_adata.shape[0]):
		if i%1000==0:print("Imputing spot", i)
		index=sudo_adata.obs.index[i]
		dis_tmp=dis.loc[index, :].sort_values()
		nbs=dis_tmp[0:num_nbs]
		dis_tmp=(nbs.to_numpy()+0.1)/np.min(nbs.to_numpy()+0.1) #avoid 0 distance
		if isinstance(k, int):
			weights=((1/(dis_tmp**k))/((1/(dis_tmp**k)).sum()))
		else:
			weights=np.exp(-dis_tmp)/np.sum(np.exp(-dis_tmp))
		row_index=[known_adata.obs.index.get_loc(i) for i in nbs.index]
		sudo_adata.X[i, :]=np.dot(weights, known_adata.X[row_index,:])
	return sudo_adata

