#%% imports
import pyclustering
from pyclustering.cluster import xmeans
import numpy as np
import pylab
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

dataCsv = open("/home/yagi/CLionProjects/homographyStepEstimation/projects/optitrack/RstepPoints.txt", "r")
# dataCsv = open("/home/yagi/CLionProjects/homographyStepEstimation/projects/optitrack/csv/021501left.csv", "r")
classNum = 5
data = np.array([])
lineNum = 0
for line in dataCsv:
    words = line.split(" ")
    pt = np.array([float(words[0]), float(words[1]), float(words[2])])
    data = np.append(data, pt)
    lineNum+=1
data = np.reshape(data, (lineNum, 3))
dataCsv.close()

# #%% create data 3d
# data = np.concatenate((
#     np.random.uniform(-10, 0,    (100, 3)),
#     np.random.uniform(-20, -10,  (100, 3)) + [0, 50, 0],
#     np.random.uniform(-30, -20,  (100, 3)),
#     np.random.uniform(-40, -30,  (100, 3)) + [-50, 20, 30],
#     np.random.uniform(-50, -40,  (100, 3)),
#     np.random.uniform(-60, -50,  (100, 3)) + [50, 0, 100],
#     np.random.uniform(-70, -60,  (100, 3)),
#     np.random.uniform(-80, -70,  (100, 3)) + [-30, 10, -30],
#     np.random.uniform(-90, -80,  (100, 3)) + [0, 0, 50],
#     np.random.uniform(-100, -90, (100, 3)) + [70, 0, -30],
#     ), axis=0)
print(data)

X = data[..., 0]
Y = data[..., 1]
Z = data[..., 2]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X, Y, Z, s=0.8)
# mappable = ax.contour(X, Y, Z, cmap="Blues")
# fig.colorbar(mappable)
# mappable0 = ax[0].pcolor(X, Y, Z, cmap="Blues")
# fig.colorbar(mappable0, ax=ax[0])
# fig.tight_layout() # これが無いと表示が少し崩れる
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
# ax.plot(, "o-", color="#00aa00", ms=4, mew=0.5)
# ax.scatter3D(data[..., 0], data[..., 1], data[..., 2], "o-", color="#00aa00", ms=4, mew=0.5)
plt.show()

pylab.savefig("./data.png")

#%% clustering
# init_center = pyclustering.cluster.xmeans.kmeans_plusplus_initializer(data, 5).initialize()
# xm = pyclustering.cluster.xmeans.xmeans(data, init_center, 5, ccore=False)
# xm.process()
# clusters = xm.get_clusters()

# create instance of K-Means algorithm with prepared centers
initial_centers = pyclustering.cluster.xmeans.kmeans_plusplus_initializer(data, classNum).initialize()
k_means_instance = pyclustering.cluster.xmeans.kmeans(data, initial_centers, ccore=False)
k_means_instance.process()

clusters = k_means_instance.get_clusters()
pyclustering.utils.draw_clusters(data, clusters, display_result=False)
pylab.savefig("./data2.png")
pylab.show()

# final_centers = kmeans_instance.get_centers()
