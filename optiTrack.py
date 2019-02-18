import pyclustering
from pyclustering.cluster import xmeans
import numpy as np
import pylab
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
dataCsv = open("/home/yagi/CLionProjects/homographyStepEstimation/projects/optitrack/csv/021501left.csv", "r")
lineNum = 1
xList = []
yList = []
zList = []
for line in dataCsv:
    words = line.split(",")
    if len(words[0]) > 2:
        zList.append(float(words[0]))
        xList.append(float(words[1]))
        yList.append(float(words[2].strip()))
        lineNum+=1
dataCsv.close()
arr_1 = np.array(xList)
arr_2 = np.array(yList)
arr_3 = np.array(zList)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(arr_1,arr_2,arr_3, s=0.8)
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