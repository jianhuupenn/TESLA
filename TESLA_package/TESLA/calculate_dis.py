import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import matplotlib.colors as clr
import matplotlib.pyplot as plt

def distance(t1,t2):
	sum=((t1-t2)**2).sum()
	return math.sqrt(sum)

def calculate_dis_matrix(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
	#x,y,x_pixel, y_pixel are lists
	dis=np.zeros((len(x),len(x)))
	if histology:
		assert (x_pixel is not None) & (x_pixel is not None) & (image is not None)
		assert (len(x)==len(x_pixel)) & (len(y)==len(y_pixel))
		print("Calculateing distance matrix using histology image...")
		#beta to control the range of neighbourhood when calculate grey vale for one spot
		#alpha to control the color scale
		beta_half=round(beta/2)
		g=[]
		for i in range(len(x_pixel)):
			max_x=image.shape[0]
			max_y=image.shape[1]
			nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
			g.append(np.mean(np.mean(nbs,axis=0),axis=0))
		c0, c1, c2=[], [], []
		for i in g:
			c0.append(i[0])
			c1.append(i[1])
			c2.append(i[2])
		c0=np.array(c0)
		c1=np.array(c1)
		c2=np.array(c2)
		print("Var of c0,c1,c2 = ", np.var(c0),np.var(c1),np.var(c2))
		c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
		c4=(c3-np.mean(c3))/np.std(c3)
		z_scale=np.max([np.std(x), np.std(y)])*alpha
		z=c4*z_scale
		z=z.tolist()
		print("Var of x,y,z = ", np.var(x),np.var(y),np.var(z))
		for i in range(len(x)):
			if i%500==0:
				print("Calculating spot ", i, "/", len(x))
			for j in range(i, len(x)):
				cord1=np.array([x[i],y[i],z[i]])
				cord2=np.array([x[j],y[j],z[j]])
				dis[i][j]=dis[j][i]=distance(cord1, cord2)
		return dis, z
	else:
		print("Calculateing distance matrix using xy only...")
		for i in range(len(x)):
			if i%50==0:
				print("Calculating spot ", i, "/", len(x))
			for j in range(i, len(x)):
				cord1=np.array([x[i],y[i]])
				cord2=np.array([x[j],y[j]])
				dis[i][j]=dis[j][i]=distance(cord1, cord2)
		return dis

def extract_color(x_pixel=None, y_pixel=None, image=None, beta=49, RGB=True):
	if RGB:
		#beta to control the range of neighbourhood when calculate grey vale for one spot
		beta_half=round(beta/2)
		g=[]
		for i in range(len(x_pixel)):
			max_x=image.shape[0]
			max_y=image.shape[1]
			nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
			g.append(np.mean(np.mean(nbs,axis=0),axis=0))
		c0, c1, c2=[], [], []
		for i in g:
			c0.append(i[0])
			c1.append(i[1])
			c2.append(i[2])
		c0=np.array(c0)
		c1=np.array(c1)
		c2=np.array(c2)
		c3=(c0*np.var(c0)+c1*np.var(c1)+c2*np.var(c2))/(np.var(c0)+np.var(c1)+np.var(c2))
	else:
		beta_half=round(beta/2)
		g=[]
		for i in range(len(x_pixel)):
			max_x=image.shape[0]
			max_y=image.shape[1]
			nbs=image[max(0,x_pixel[i]-beta_half):min(max_x,x_pixel[i]+beta_half+1),max(0,y_pixel[i]-beta_half):min(max_y,y_pixel[i]+beta_half+1)]
			g.append(np.mean(nbs))
		c3=np.array(g)
	return c3
