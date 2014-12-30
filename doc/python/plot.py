#coding: utf-8

import re

class Result(object):
    """docstring for Result"""
    def __init__(self, fileName):
        f = open(fileName)
        s = f.readline()
        self.pts = int(float(re.search(r'\d+\.\d+', s).group(0))*1000000)
        self.nameArr = []
        self.cpuTimeArr = {}
        self.gpuTimeArr = {}
        for i in xrange(6):
            s = f.readline()
            name = re.search(r'\S+::\S+', s).group(0)
            self.nameArr.append(name)
            f.readline()
            s = f.readline()
            self.gpuTimeArr[name] = float(s.split('\t')[1].split('\n')[0][:-1])
            s = f.readline()
            self.cpuTimeArr[name] = float(s.split('\t')[1].split('\n')[0][:-1])
        s = f.readline()
        self.shapesGPU = int(s.split(' ')[3])
        s = f.readline()
        self.gpuTimeArr["Total"] = float(s.split('\t')[1].split('\n')[0][:-1])
        s = f.readline()
        self.shapesCPU = int(s.split(' ')[4])
        self.cpuTimeArr["Total"] = float(s.split(' ')[8])
        self.nameArr.append("Total")

    def __str__(self):
        s = ""
        s += str(self.pts) + "\n"
        s += str(self.nameArr) + '\n'
        s += str(self.cpuTimeArr) + '\n'
        s += str(self.gpuTimeArr) + '\n'
        return s

template = "./res/image0.txt"

resArr = []

for x in xrange(10):
    try:
        r = Result("./res/image{0}.txt".format(x))
        resArr.append(r)
    except Exception, e:
        pass

print resArr

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
# import math

pltDict = {}

class PltModel(object):
    """docstring for PltModel"""
    def __init__(self):
        self.cpuPts = []
        self.cpuTime = []
        self.gpuPts = []
        self.gpuTime = []

    def plot(self, title):
        # plt.xkcd()

        plt.subplot(1,2,1)
        plt.title(title)
        fcpu = self.plotWithCubic(self.cpuPts, self.cpuTime, 'rd', 'r--')
        fgpu = self.plotWithCubic(self.gpuPts, self.gpuTime, 'bx', 'b:')
        # f2 = interp1d(self.cpuPts, self.cpuTime, kind='cubic')
        # xnew = np.linspace(self.cpuPts[0], self.cpuPts[-1])
        # plt.plot(self.cpuPts, self.cpuTime, 'rd', xnew, f2(xnew),'--')
        # plt.plot(self.gpuPts, self.gpuTime, 'bx:')
        
        plt.legend(['CPU Time Data Point', 'CPU Time', 'GPU Time Data Point', 'GPU Time'], loc='best')
        # print (self.cpuPts, self.cpuTime, 'r--', self.gpuPts, self.gpuTime, 'bs')
        plt.xlabel("Number of keypoints")
        plt.ylabel("Time (s)")
        plt.subplot(1,2,2)
        xnew = np.linspace(max(self.cpuPts[0], self.gpuPts[0]), 
            min(self.cpuPts[-1], self.gpuPts[-1]))
        plt.plot(xnew, fcpu(xnew)/fgpu(xnew), 'r-')
        plt.xlabel("Number of keypoints")
        plt.ylabel("Speedup Ratio")
        plt.legend(['Speedup Ratio'], loc='best')
        plt.show()

    def plotWithCubic(self, arrSrc , arrDes, styPt, styLi):
        f = interp1d(arrSrc, arrDes, kind='slinear')
        xnew = np.linspace(arrSrc[0], arrSrc[-1])
        plt.plot(arrSrc, arrDes, styPt, xnew, f(xnew), styLi)
        return f

    def addGpuPtsTime(self, pts, time):
        self.insertTime(self.gpuPts, self.gpuTime, pts, time)

    def addCpuPtsTime(self, pts, time):
        self.insertTime(self.cpuPts, self.cpuTime, pts, time)

    def insertTime(self, ptsArr, timeArr, pts, time):
        for i in xrange(len(ptsArr)):
            if pts < ptsArr[i]:
                ptsArr.insert(i, pts)
                timeArr.insert(i, time)
                return
        ptsArr.append(pts)
        timeArr.append(time)

for eachRes in resArr:
    names = eachRes.nameArr
    for eachName in names:
        if eachName not in pltDict:
            pltDict[eachName] = PltModel()
        pltDict[eachName].addCpuPtsTime(eachRes.shapesCPU, eachRes.cpuTimeArr[eachName])
        pltDict[eachName].addGpuPtsTime(eachRes.shapesGPU, eachRes.gpuTimeArr[eachName])

for eachName in pltDict:
    pltDict[eachName].plot(eachName)
