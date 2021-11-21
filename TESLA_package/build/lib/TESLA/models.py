#from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.init

class Spatial_CNN(nn.Module):
	def __init__(self,input_dim, nChannel=100, nConv=2, kernel_size_list=[3,3],stride_list=[1,1], padding_list=[1,1]):
		super(Spatial_CNN, self).__init__()
		assert nConv==len(kernel_size_list)==len(stride_list)==len(padding_list)
		self.nChannel=nChannel
		self.nConv=nConv
		self.conv1 = nn.Conv2d(input_dim, self.nChannel, kernel_size=kernel_size_list[0], stride=stride_list[0], padding=padding_list[0] )
		self.bn1 = nn.BatchNorm2d(self.nChannel)
		self.conv2 = nn.ModuleList()
		self.bn2 = nn.ModuleList()
		for i in range(self.nConv-1):
			self.conv2.append( nn.Conv2d(self.nChannel, self.nChannel, kernel_size=kernel_size_list[i+1], stride=stride_list[i+1], padding=padding_list[i+1] ) )
			self.bn2.append( nn.BatchNorm2d(self.nChannel) )
		self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
		self.bn3 = nn.BatchNorm2d(self.nChannel)
	def forward(self, x):
		x = self.conv1(x)
		x = F.relu( x )
		x = self.bn1(x)
		for i in range(self.nConv-1):
			x = self.conv2[i](x)
			x = F.relu( x )
			x = self.bn2[i](x)
		x = self.conv3(x)
		x = self.bn3(x)
		return x
