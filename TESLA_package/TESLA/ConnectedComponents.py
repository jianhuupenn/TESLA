#Get connected components in an undirected graph
import pandas as pd
import numpy as np
from collections import deque
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
	def ConnectedComponentsDFS(self):
		visited = pd.Series([0]* len(self.V), index=self.V)
		cc = []
		for v in self.V:
			if visited[v] == 0:
				tmp = []
				cc.append(self.DFSUtil(v, tmp, visited))
		return cc

	def ConnectedComponents(self):
		visited = pd.Series([0] * len(self.V), index=self.V)
		cc = []
		for v in self.V:
			if visited[v] == 1:
				continue

			queue = deque([v])
			visited[v] = 1
			tmp = [v]
			while len(queue) > 0:
				elem = queue.pop()
				nbrs = self.adj[elem][self.adj[elem] == 1].index.tolist()

				for nbr in nbrs:
					if visited[nbr] == 0:
						queue.append(nbr)
						visited[nbr] = 1
						tmp.append(nbr)

			if len(tmp) > 0:
				cc.append(tmp)

		return cc
