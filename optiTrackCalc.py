#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from scipy import optimize
from numpy import*


R_N_CLUSTERS = 4
L_N_CLUSTERS = 4
C_PLAIN = [100,100,100,100]
PLAIN_NORMAL = np.array([C_PLAIN[0],C_PLAIN[1],C_PLAIN[2]])
PLAIN_PT = (C_PLAIN[0]*C_PLAIN[3], C_PLAIN[1]*C_PLAIN[3], C_PLAIN[2]*C_PLAIN[3])

project_path = "/home/yagi/CLionProjects/homographyStepEstimation/projects"
project_name = "optitrack"
video_name = "021501"
txt_path = project_path + "/" + project_name + "/results/" + video_name + "/"

r_result_path = txt_path + 'Rresult.txt'
l_result_path = txt_path + 'Lresult.txt'

def loadData(path):
    # with open(path) as f:
    #     for s_line in f:
    #         words = s_line.split(" ")
    #         point = []
    #         point.append(int(words[0]))
    #         point.append(int(words[1]))
    #         point.append(int(words[2]))
    #         ptList.append(point)
    #         frameList.append(int(words[2]))
    #         if words[0] == "Frame":
    #             break
    #
    # lineNum = 0
    # with open(path) as f:
    #     for s_line in f:
    #         if(lineNum >= 8):
    #             words = s_line.split(" ")
    #         else:
    #             lineNum+=1
    # return np.array(ptList)

def distFromPlain(pt):
    PA = np.array([pt.x - PLAIN_PT.x, pt.y - PLAIN_PT.y, pt.z - PLAIN_PT.z])
    return(np.dot(PA, PLAIN_NORMAL))

def kmeans(features, N_CLUSTERS):
    # クラスタリングする
    cls = KMeans(n_clusters=N_CLUSTERS)
    pred = cls.fit_predict(features)

    # 各要素をラベルごとに色付けして表示する
    for i in range(N_CLUSTERS):
        labels = features[pred == i]
        plt.scatter(labels[:, 0], labels[:, 1])

    # クラスタのセントロイド (重心) を描く
    centers = cls.cluster_centers_

    # 重心をソート
    centers.sort(axis=0)
    plt.scatter(centers[:, 0], centers[:, 1], s=100,
                facecolors='none', edgecolors='black')
    plt.show()

    return centers

#Least squares method with scipy.optimize
def fit_func(parameter,x,y):
    a = parameter[0]
    b = parameter[1]
    residual = y-(a*x+b)
    return residual

def linearApproximation(data):
    xdata = data[:, 0]
    ydata = data[:, 1]
    print(xdata)
    print(ydata)
    parameter0 = [0., 0.]
    result = optimize.leastsq(fit_func, parameter0, args=(xdata, ydata))
    a_fit = result[0][0]
    b_fit = result[0][1]

    print(a_fit, b_fit)
    return a_fit, b_fit

def getStrideLength(data):
    steps = data[:,:2]
    slList = []
    for i in range(steps.shape[0] - 1):
        a = np.array([steps[i,0],steps[i,1]])
        b = np.array([steps[i+1,0],steps[i+1,1]])
        u = b - a
        sl = np.linalg.norm(u)
        slList.append(sl)
    return slList

def getStrideWidth(steps, a, b):
    pt1 = (0, b)
    pt2 = (500, 500*a+b)
    swList = []
    for i in range(steps.shape[0]):
        sw = distance_l(pt1, pt2, (steps[i,0],steps[i,1]))
        swList.append(sw)
    return swList

def distance_l(a,b,c):
    u = np.array([b[0]-a[0],b[1]-a[1]])
    v = np.array([c[0]-a[0],c[1]-a[1]])
    L = abs(cross(u,v)/linalg.norm(u))
    return L

def output_result(steps, sl, sw, result_path):
    with open(result_path, mode='w') as f:

        f.write("step positions: \n")
        for i in range(steps.shape[0]):
            f.write(str(steps[i,0]) + " " + str((steps[i,1])))
            f.write("\n")

        f.write("step timing: \n")
        for i in range(steps.shape[0]):
            f.write(str(steps[i, 2]))
            f.write("\n")

        f.write("stride length: \n")
        for a in sl:
            f.write(str(a))
            f.write("\n")

        f.write("stride width: \n")
        for a in sw:
            f.write(str(a))
            f.write("\n")

def estimateParams(r, l):
    rsl = getStrideLength(r)
    lsl = getStrideLength(l)
    ra, rb = linearApproximation(r)
    la, lb = linearApproximation(l)
    rsw = getStrideWidth(r,la,lb)
    lsw = getStrideWidth(l,ra,rb)
    output_result(r, rsl, rsw, r_result_path)
    output_result(l, lsl, lsw, l_result_path)


if __name__ == '__main__':
    Rsteps = loadData(txt_path + "RstepPoints.csv")
    Lsteps = loadData(txt_path + "LstepPoints.csv")
    Rcenters = kmeans(Rsteps, R_N_CLUSTERS)
    Lcenters = kmeans(Lsteps, L_N_CLUSTERS)
    estimateParams(Rcenters, Lcenters)


