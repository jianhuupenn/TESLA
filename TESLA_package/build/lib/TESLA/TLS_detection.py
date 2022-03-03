import os, sys, csv,re, time, random
import cv2
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from . util import *
from . contour_util import *

def TLS_detection( pred_refined_list, cluster_density_list, num_required, cnt_color, pooling="min"):
	pred_TLS=np.zeros([len(pred_refined_list[0]), len(pred_refined_list)])
	for i in range(len(pred_refined_list)):
		tmp=np.zeros(pred_refined_list[i].shape)
		for k, v in cluster_density_list[i].items():
			tmp[pred_refined_list[i]==k]=v/np.max(list(cluster_density_list[i].values()))
		pred_TLS[:,i]=tmp
	target = np.partition(pred_TLS, -num_required, axis=1)[:,-num_required:] #Select top num_required
	if pooling=="mean":
		target=np.mean(target, axis=1)
	elif pooling=="min":
		target=np.min(target, axis=1)
	else:
		print("Error! Pooling logic not understood.")
	target=(target-np.min(target))/(np.max(target)-np.min(target))
	target[target<0.5]=0
	return target


def plot_TLS_score(img, resize_factor, binary,target, cnt_color):
	resize_width=int(img.shape[1]*resize_factor)
	resize_height=int(img.shape[0]*resize_factor)
	binary_resized=cv2.resize(binary, (resize_width, resize_height))
	img_resized =cv2.resize(img_new, (resize_width, resize_height))
	target_img=target.reshape(resize_height, resize_width)
	target_img_rgb=(cnt_color((target*255).astype("int"))[:, 0:3]*255).reshape(resize_height, resize_width,3).astype( np.uint8 )
	target_img_rgb=(cnt_color((target*255).astype("int"))[:, 0:3]*255).reshape(resize_height, resize_width,3).astype( np.uint8 )
	target_img_rgb=cv2.cvtColor(target_img_rgb, cv2.COLOR_RGB2BGR)
	ret_img=img_resized.copy()
	#Whiten
	white_ratio=0.5
	ret_img[binary_resized!=0]=ret_img[binary_resized!=0]*(1-white_ratio)+np.array([255, 255, 255])*(white_ratio)
	ret_img[target_img!=0]=target_img_rgb[target_img!=0]
	ret_img[binary_resized==0]=255
	return ret_img





