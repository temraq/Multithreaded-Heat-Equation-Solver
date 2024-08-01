from __future__ import division
import time as tm
import threading as thr

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import numpy as np

import VarDir

a = 5
b = 5
Nx = 50
Ny = 50
T = 1.0
Nt = 10
C1 = 1
C2 = 1
alpha = 0.25


f = open("input1.txt", "r")
a = float(f.readline().split(" ")[0])
b = float(f.readline().split(" ")[0])
Nx = int(f.readline().split(" ")[0])
Ny = int(f.readline().split(" ")[0])
T = float(f.readline().split(" ")[0])
Nt = int(f.readline().split(" ")[0])
C1 = float(f.readline().split(" ")[0])
C2 = float(f.readline().split(" ")[0])
alpha = float(f.readline().split(" ")[0])
f.close()


RS = VarDir.solverRS(a=a, b=b, Nx=Nx, Ny=Ny, T=T, Nt=Nt, C1=C1, C2=C2, alpha= alpha ,test=0)
t = thr.Thread(target = RS.solve,)
t.start()
# wait until the second thread is over then visualize
sec_thread_state = False
count = 1
while(sec_thread_state == False):
    count += 1
    RS.L1.acquire()
    sec_thread_state = RS.end_second_threat
    RS.L1.release()
print("final")
print("Количество итераций в цикле первичного потока: ", count)
RS.draw()



