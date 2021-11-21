#Get connected components in an undirected graph
import pandas as pd
import numpy as np
class Graph:
	# init function to declare class variables
	def __init__(self, V, adj):
		self.V = V
		self.adj = adj
	def DFSUtil(self, v, tmp, visited):
		# Mark the current vertex as visited
		visited[v] = 1
		# Store the vertex to list
		tmp.append(v)
		# Repeat for all vertices adjacent to this vertex v
		nbr=self.adj[v][self.adj[v]==1].index.tolist()
		for i in nbr:
			if visited[i] == 0:
				tmp = self.DFSUtil(i, tmp, visited)
		return tmp
	def ConnectedComponents(self):
		visited = pd.Series([0]* len(self.V), index=self.V)
		cc = []
		for v in self.V:
			if visited[v] == 0:
				tmp = []
				cc.append(self.DFSUtil(v, tmp, visited))
		return cc
