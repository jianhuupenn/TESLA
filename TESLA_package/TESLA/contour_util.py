import pandas as pd
import numpy as np
import cv2

def scan_contour(spots, scan_x=True, shape="hexagon"):
	#shape="hexagon" # For 10X Vsium, shape="square" for ST data
	if scan_x:
		array_a="array_x"
		array_b="array_y"
		pixel_a="pixel_x"
		pixel_b="pixel_y"
	else:
		array_a="array_y"
		array_b="array_x"
		pixel_a="pixel_y"
		pixel_b="pixel_x"
	upper, lower={}, {}
	uniq_array_a=sorted(set(spots[array_a]))
	if shape=="hexagon":
		for i in range(len(uniq_array_a)-1):
			a1=uniq_array_a[i]
			a2=uniq_array_a[i+1]
			group=spots.loc[spots[array_a].isin([a1, a2]),:]
			lower[np.min(group[pixel_a])]=np.min(group[pixel_b])
			upper[np.min(group[pixel_a])]=np.max(group[pixel_b])
			#print(a1, lower[np.min(group[pixel_a])], upper[np.min(group[pixel_a])])
		a1=uniq_array_a[-1]
		a2=uniq_array_a[-2]
		group=spots.loc[spots[array_a].isin([a1, a2]),:]
		lower[np.min(group[pixel_a])]=np.min(group[pixel_b])
		upper[np.min(group[pixel_a])]=np.max(group[pixel_b])
		#print(a1, lower[np.min(group[pixel_a])], upper[np.min(group[pixel_a])])
	elif shape=="square":
		for i in range(len(uniq_array_a)-1):
			a1=uniq_array_a[i]
			group=spots.loc[spots[array_a]==a1,:]
			lower[np.min(group[pixel_a])]=np.min(group[pixel_b])
			upper[np.min(group[pixel_a])]=np.max(group[pixel_b])
		a1=uniq_array_a[-1]
		group=spots.loc[spots[array_a]==a1,:]
		lower[np.min(group[pixel_a])]=np.min(group[pixel_b])
		upper[np.min(group[pixel_a])]=np.max(group[pixel_b])
	else:
		print("Error, unknown shape, pls specify 'square' or 'hexagon'.")
	lower=np.array(list(lower.items())[::-1]).astype("int32")
	upper=np.array(list(upper.items())).astype("int32")
	cnt=np.concatenate((upper, lower), axis=0)
	cnt=cnt.reshape(cnt.shape[0], 1, 2)
	if scan_x:
		cnt=cnt[:, : , [1, 0]]
	return cnt

def cv2_detect_contour(img, 
	CANNY_THRESH_1 = 100,
	CANNY_THRESH_2 = 200,
	apertureSize=5,
	L2gradient = True,
	all_cnt_info=False):
	if len(img.shape)==3:
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	elif len(img.shape)==2:
		gray=(img*((1, 255)[np.max(img)<=1])).astype(np.uint8)
	else:
		print("Image format error!")
	edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2,apertureSize = apertureSize, L2gradient = L2gradient)
	edges = cv2.dilate(edges, None)
	edges = cv2.erode(edges, None)
	cnt_info = []
	cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	for c in cnts:
		cnt_info.append((c,cv2.isContourConvex(c),cv2.contourArea(c),))
	cnt_info = sorted(cnt_info, key=lambda c: c[2], reverse=True)
	cnt=cnt_info[0][0]
	if all_cnt_info:
		return cnt_info
	else:
		return cnt


def cut_contour_boundary(cnt, x_min, x_max, y_min, y_max, enlarge):
	ret=cnt.copy()
	ret[:, : , 0][ret[:, : , 0]>y_max]=y_max
	ret[:, : , 0][ret[:, : , 0]<y_min]=y_min
	ret[:, : , 1][ret[:, : , 1]>x_max]=x_max
	ret[:, : , 1][ret[:, : , 1]<x_min]=x_min
	return ret

def scale_contour(cnt, scale):
	M = cv2.moments(cnt)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	cnt_norm = cnt - [cx, cy]
	cnt_scaled = cnt_norm * scale
	cnt_scaled = cnt_scaled + [cx, cy]
	cnt_scaled = cnt_scaled.astype(np.int32)
	return cnt_scaled








