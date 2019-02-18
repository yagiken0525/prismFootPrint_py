import numpy as np
import matplotlib.pyplot as plt

dataCsv = open("/home/yagi/CLionProjects/homographyStepEstimation/projects/optitrack/RstepNumList.txt", "r")
lineNum = 1
xList = []
yList = []
for line in dataCsv:
    words = line.split(" ")
    yList.append(float(words[0]))
    xList.append(lineNum)
    lineNum+=1
dataCsv.close()
arr_1 = np.array(xList)
arr_2 = np.array(yList)

fig, ax = plt.subplots()
ax.plot(arr_1,arr_2)
plt.show()