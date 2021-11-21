import numpy as np
import pandas as pd
import time
import cv2
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init
from torch.autograd import Variable
from . models import Spatial_CNN

class TESLA(object):
	def __init__(self):
		super(TESLA, self).__init__()

	def train(self, input,
		use_cuda=False, 
		train_refine=True, 
		radius=3, 
		nChannel=100, 
		lr=0.1, 
		minLabels=20, 
		maxIter=30, 
		stepsize_sim=1, 
		stepsize_con=5, 
		threshold=1000,
		plot_intermedium=False,
		plot_dir="./"):
		self.use_cuda=use_cuda 
		self.train_refine=train_refine 
		self.radius=radius 
		self.nChannel=nChannel 
		self.lr=lr 
		self.minLabels=minLabels 
		self.maxIter=maxIter 
		self.stepsize_sim=stepsize_sim 
		self.stepsize_con=stepsize_con 
		self.threshold=threshold
		self.plot_intermedium=plot_intermedium
		self.resize_height=input.shape[0]
		self.resize_width=input.shape[1]
		input = torch.from_numpy( np.array([input.transpose( (2, 0, 1) ).astype('float32')]) )
		if use_cuda:
			input = input.cuda()
		input = Variable(input)
		#--------------------------------------- Train ---------------------------------------
		self.model = Spatial_CNN(input.size(1), nConv=2, nChannel=self.nChannel, kernel_size_list=[5, 5],stride_list=[1, 1], padding_list=[2, 2])
		if use_cuda:
			self.model.cuda()
		# Similarity loss definition
		loss_fn = torch.nn.CrossEntropyLoss()
		# Continuity loss definition
		loss_hpy = torch.nn.L1Loss(size_average = True)
		loss_hpz = torch.nn.L1Loss(size_average = True)
		HPy_target = torch.zeros(self.resize_height-1, self.resize_width, nChannel)
		HPz_target = torch.zeros(self.resize_height, self.resize_width-1, nChannel)
		if use_cuda:
			HPy_target = HPy_target.cuda()
			HPz_target = HPz_target.cuda()
		optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
		label_colours = np.random.randint(255,size=(100,3))
		start_time = time.time()
		self.model.train()
		for batch_idx in range(maxIter):
			# forwarding
			optimizer.zero_grad()
			output = self.model( input )[ 0 ]
			output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
			outputHP = output.reshape( (self.resize_height, self.resize_width, nChannel) )
			HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
			HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
			lhpy = loss_hpy(HPy,HPy_target)
			lhpz = loss_hpz(HPz,HPz_target)
			_, target = torch.max( output, 1 )
			img_target = target.data.cpu().numpy()
			# Total number of clusters
			nLabels = len(np.unique(img_target))
			# Number of main clusters
			mainLabels=(pd.Series(img_target).value_counts()>=threshold).sum()
			#--------------------------Refine during training----------------------------------------
			if train_refine:
				pixel_num=pd.Series(img_target).value_counts()
				main_clusters=pixel_num.index[pixel_num>=threshold].tolist()
				minor_clusters=pixel_num.index[pixel_num<threshold].tolist()
				b_refine = img_target.reshape( (self.resize_height, self.resize_width))
				max_x, max_y=self.resize_width, self.resize_height
				replace_map={}
				for i in minor_clusters:
					nbs=[]
					xy=np.where(b_refine==i)
					for j in range(len(xy[0])):
						x, y=xy[0][j], xy[1][j]
						nbs=nbs+b_refine[max(0,x-radius):min(max_x,x+radius+1),max(0,y-radius):min(max_y,y+radius+1)].flatten().tolist()
						nbs_num=pd.Series(nbs).value_counts()
						if nbs_num.index[nbs_num.index.isin(main_clusters)].shape[0]>0:
							replace_map[i]=nbs_num.index[nbs_num.index.isin(main_clusters)][0]
				target=torch.from_numpy(pd.Series(img_target).replace(replace_map).to_numpy())
			#------------------------------------------------------------------
			loss1 = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz)
			loss2 = stepsize_con * (lhpy + lhpz)
			loss=loss1+loss2
			loss.backward()
			optimizer.step()
			print (batch_idx, '/', maxIter, '|', ' label num :', nLabels, '|',' main clusters :',mainLabels,' | feature loss :', loss1.item(),' | spatial loss :', loss2.item())
			print("--- %s seconds ---" % (time.time() - start_time))
			if batch_idx%2==0:
				output = self.model(input)[0]
				output = output.permute(1, 2, 0).contiguous().view(-1, nChannel)
				_, target = torch.max(output, 1)
				img_target = target.data.cpu().numpy()
				img_target_rgb = np.array([label_colours[ c %  label_colours.shape[0]] for c in img_target])
				img_target_rgb = img_target_rgb.reshape(self.resize_height, self.resize_width, 3).astype( np.uint8 )
				if self.plot_intermedium:
					cv2.imwrite(plot_dir+str(batch_idx)+"_nL"+str(nLabels)+"_mL"+str(mainLabels)+".png", img_target_rgb)
			if mainLabels <= minLabels:
				print ("mainLabels", mainLabels, "reached minLabels", minLabels, ".")
				break

	def predict(self, input):
		input = torch.from_numpy( np.array([input.transpose( (2, 0, 1) ).astype('float32')]) )
		output = self.model( input )[ 0 ]
		output = output.permute( 1, 2, 0 ).contiguous().view( -1, self.nChannel )
		prob=output.data.cpu().numpy()
		prob=np.divide(np.exp(prob), np.sum(np.exp(prob), 1).reshape(-1, 1))
		_, pred = torch.max( output, 1 )
		pred = pred.data.cpu().numpy()
		return prob, pred








