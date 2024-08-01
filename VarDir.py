from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import numpy as np
import time as tm

import threading as thr

class solverRS:
    # a - предельное значение по оси x.
    # b - предельное значение по оси y.
    # Nx - число шагов по оси x.
    # Ny - число шагов по оси y.
    # T - предельное значение на оси времени.
    # Nt - число шагов по оси времени.
    def __init__(self, a, b, Nx, Ny, T, Nt, C1, C2, alpha, test = 0):
        self.L1 = thr.Lock() # Для блокировки флага о завершнении вторичного потока.

        self.a = a
        self.b = b
        self.Nx = Nx
        self.Ny = Ny
        self.T = T
        self.Nt = Nt
        self.C1 = C1
        self.C2 = C2
        self.alpha = alpha

        self.Hx = a/Nx
        self.Hy = b/Ny
        self.Ht = T/Nt

        if test == 0:
            self.f = lambda x, y, t: self.C1*t*math.exp(-((x-3*a/8)**2 + (y-7*b/8)**2)*self.alpha)
        elif test == 1:
            self.f = lambda x, y, t: 0.0
        elif test == 2:
            self.f = lambda x, y, t: 0.3
        elif test == 3:
            self.f = lambda x, y, t: x + y

        self.v = lambda x, y: self.C2*x*y/self.b
        self.g1 = lambda x, y, t: 0
        self.g2 = lambda x, y, t: 0
        self.g3 = lambda x, y, t: 0
        self.g4 = lambda x, y, t: 0
        # выделение памяти под сеточные фунцкии.
        self.Ygrid = np.zeros((Nx + 1, Ny + 1), dtype=np.float64)  # Матрица для хранения сетки на n-ом "слое".

        for i in range(1, self.Nx-1):
            x_tmp = i*self.Hx
            for j in range(1, self.Ny-1):
                y_tmp = j*self.Hy
                self.Ygrid[i][j] = self.v(x_tmp, y_tmp)
        self.Ygrid_help = np.zeros((Nx + 1, Ny + 1), dtype=np.float64)  # Матрица для хранения сетки на n+1/2-ом "слое".
        # функции на границах
        for i in range(0, Nx+1, 1): # g1.
            x_tmp = self.Hx*i
            self.Ygrid[i][0] = self.g1(x_tmp, 0, -1)
            #self.Ygrid[i][0] = 1
        #print("g1: ", self.Ygrid[:,0])
        for i in range(0, Nx+1, 1): # g3.
            x_tmp = self.Hx*i
            self.Ygrid[i][self.Ny] = self.g3(x_tmp, self.b, -1)
            #self.Ygrid[i][self.Ny] = 3
        #print("g3: ", self.Ygrid[:,self.Ny])
        for j in range(0, Ny+1, 1): # g2.
            y_tmp = self.Hy*i
            self.Ygrid[0][j] = self.g2(0, y_tmp, -1)
            #self.Ygrid[0][j] = 2
        #print("g2: ", self.Ygrid[0])
        for j in range(0, Ny+1, 1): # g4.
            y_tmp = self.Hy*i
            self.Ygrid[self.Nx][j] = self.g2(self.a, y_tmp, -1)
            #self.Ygrid[self.Nx][j] = 4
        #print("g4: ", self.Ygrid[self.Nx])
        # Доопределить значений сеточных функций на границах для слоя n+1/2.
        # т.е. работаю с self.Ygrid_help.
        # Использую то, что в нашей задаче эти значения равны нулю, вне зависимости от слоя.
        for j in range(1, self.Ny):
            self.Ygrid_help[0][j] = 0
            self.Ygrid_help[self.Nx][j] = 0
        for i in range(1, self.Nx):
            self.Ygrid_help[i][0] = 0
            self.Ygrid_help[i][self.Ny] = 0
        # Для вычисления правых частей уравнений в СЛАУ для метода прогонки, ур-ие (14).
        self.F = lambda i, j, t_i: self.Ygrid[i][j] + 0.5 * self.Ht * \
                         ((self.Ygrid[i][j+1] - 2 * self.Ygrid[i][j] + self.Ygrid[i][j-1]) / \
                          (self.Hx ** 2)) + \
                              self.f(i*self.Hx, j*self.Hy, (t_i+1/2)*self.Ht)
        #Для вычисления правых частей уравнений в СЛАУ для метода прогонки, ур-ие (15).
        self.Phi = lambda i, j, t_i: self.Ygrid_help[i][j] + 0.5*self.Ht* \
                                (self.Ygrid_help[i+1][j] - 2*self.Ygrid_help[i][j] + self.Ygrid_help[i-1][j])/ \
                                (self.Hy**2)+\
                                self.f(i*self.Hx, j*self.Hy, (t_i+1)*self.Ht)


    # Сбрасывает вычисленную сеточную функцию к начальному значению
    def reset(self):
        for i in range(1, self.Nx-1):
            x_tmp = i*self.Hx
            for j in range(1, self.Ny-1):
                y_tmp = j*self.Hy
                self.Ygrid[i][j] = self.v(x_tmp, y_tmp)
    # Метод прогонки для (14).
    # t_i - номер последнего сформированного слоя.
    # тут применяется одномерная прогонка по переменной x при фиксированных j.
    def TomasMethod14(self, t_i):
        A = 0.5*self.Ht/(self.Hx**2)
        B = -(1+self.Ht/(self.Hx**2))

        # Массив для хранения прогоночных коэффициентов.
        koef = np.empty((self.Nx , 2), dtype=np.float64)
        for j in range(1, self.Ny):
            # Прямой ход.
            koef[0, 0] = -A/B
            koef[0, 1] = -self.F(1, j, t_i)/B
            for i in range(1, self.Nx): # ? -> Nx-1.
                koef[i, 0] = -A/(B+A*koef[i-1, 0])
                koef[i, 1] =(-self.F(i, j, t_i) - A*koef[i-1, 1])/(B+A*koef[i-1,0])
            koef[self.Nx-1, 1] = (-self.F(self.Nx-1, j, t_i) + A*koef[self.Nx-2, 1])/ \
                                 (B + A*koef[self.Nx-2, 0])
            # Обратный ход.
            self.Ygrid_help[self.Nx-1, j] = koef[self.Nx-1, 1]
            for i in range(self.Nx-1, 0, -1):
                self.Ygrid_help[i, j] = koef[i-1, 0]*self.Ygrid_help[i+1, j] + koef[i-1, 1]

    # Метод прогонки для (15).
    # t_i - номер последнего сформированного слоя.
    # тут применяется одномерная прогонка по переменной y при фиксированных i.
    def TomasMethod15(self, t_i):
        A = 0.5 * self.Ht / (self.Hy ** 2)
        B = -(1 + self.Ht / (self.Hy ** 2))
        # Массив для хранения прогоночных коэффициентов.
        koef = np.empty((self.Ny, 2), dtype=np.float64)
        for i in range(1, self.Nx):
            # Прямой ход.
            koef[0, 0] = -A/B
            koef[0, 1] = -self.Phi(i, 1, t_i)/B
            for j in range(1, self.Ny):
                koef[j, 0] = -A/(B+A*koef[j-1, 0])
                koef[j, 1] =(-self.Phi(i, j, t_i) - A*koef[j-1, 1])/(B+A*koef[j-1,0])
            koef[self.Ny - 1, 1] = (-self.Phi(i, self.Ny-1, t_i) + A * koef[self.Ny - 2, 1]) / \
                                   (B + A * koef[self.Ny - 2, 0])
            # Обратный ход.
            self.Ygrid[i, self.Ny - 1] = koef[self.Ny - 1, 1]
            for j in range(self.Ny - 1, 0, -1):
                self.Ygrid[i, j] = koef[j - 1, 0] * self.Ygrid[i, j+1] + koef[j - 1, 1]


    # Решить уравнение методом переменных направлений.
    def solve(self):
        self.L1.acquire()
        self.end_second_threat = False;
        self.L1.release()
        count = 0
        for t_i in np.arange(0, self.Nt+1):  # Количество расчитываемых слоёв.
            count +=1
            #print("t_i = ", t_i)
            # 1. Прогонка по переменной x, при фиксированных значениях j = 1, 0, ... , N-1.
            # т.е. решение систем из уравнения (14) (слой t+1/2).
            tm.sleep(0.5)
            self.TomasMethod14(t_i)
            # 2. Прогонка по переменной y, при фиксированных значениях j = 1, 0, ... , N-1.
            # т.е. решение систем из уравнения (15) (слой t+1).
            self.TomasMethod15(t_i)
            #self.draw_help()
            #self.draw()
        print("Количество итераций в цикле вторичного потока: ", count)
        self.L1.acquire()
        self.end_second_threat = True;
        self.L1.release()

    # 3d отображение найденного решения.
    def draw(self):
        # График приближенного решения.
        # Creating dataset
        x = np.zeros(self.Nx + 1)
        for i in range(0, self.Nx + 1):
            x[i] = self.Hy * i
        x2 = np.outer(x, np.ones(self.Ny + 1))
        y = np.zeros(self.Ny + 1)
        for i in range(0, self.Ny + 1):
            y[i] = self.Hx * i
        y2 = np.outer(y, np.ones(self.Nx + 1))
        y2 = y2.T
        # self.calculateGridExplisit()
        # print("x2: ", x2.shape)
        # print("y2: ", y2.shape)
        u = self.Ygrid
        # print("u: ", u.shape)

        # Creating figure
        fig = plt.figure(figsize=(14, 9))
        ax = plt.axes(projection='3d')
        # Creating color map
        my_cmap = plt.get_cmap('hot')
        # Creating plot
        surf = ax.plot_surface(y2, x2, u, cmap=my_cmap,
                               edgecolor='none')
        fig.colorbar(surf, ax=ax,
                     shrink=0.5, aspect=5)
        ax.set_xlabel('y-axis')
        ax.set_ylabel('x-axis')
        ax.set_zlabel('u-axis')
        plt.show()

    # 3d отображение промежуточного слоя.
    def draw_help(self):
        # График приближенного решения.
        # Creating dataset
        y = np.zeros(self.Ny + 1)
        for i in range(0, self.Ny + 1):
            y[i] = self.Hy * i
        y2 = np.outer(y, np.ones(self.Nx + 1))
        x = np.zeros(self.Nx + 1)
        for i in range(0, self.Nx + 1):
            x[i] = self.Hx * i
        x2 = np.outer(x, np.ones(self.Ny + 1))
        x2 = x.T
        # self.calculateGridExplisit()

        u = self.Ygrid_help

        # Creating figure
        fig = plt.figure(figsize=(14, 9))
        ax = plt.axes(projection='3d')
        # Creating color map
        my_cmap = plt.get_cmap('hot')
        # Creating plot
        surf = ax.plot_surface(x2, y2, u, cmap=my_cmap,
                                   edgecolor='none')
        fig.colorbar(surf, ax=ax,
                         shrink=0.5, aspect=5)
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('u-axis')
        plt.show()
    # 3d отображение функции f.
    def draw_f(self, t = 1):
        # График приближенного решения.
        # Creating dataset
        x = np.zeros(self.Nx + 1)
        for i in range(0, self.Nx + 1):
            x[i] = self.Hx * i
        x2 = np.outer(x, np.ones(self.Ny + 1))

        y = np.zeros(self.Ny + 1)
        for i in range(0, self.Ny + 1):
            y[i] = self.Hy * i
        y2 = np.outer(y, np.ones(self.Nx + 1))
        y2 = y2.T
        # self.calculateGridExplisit()

        fGrid = np.zeros((self.Nx+1, self.Ny+1), dtype=np.float64)
        for j in range(self.Ny+1):
            ytmp = 0 + j*self.Hy
            for i in range(self.Nx+1):
                xtmp = 0 + i*self.Hx
                fGrid[i][j] = self.f(xtmp, ytmp, t)

        # Creating figure
        fig = plt.figure(figsize=(14, 9))
        ax = plt.axes(projection='3d')
        # Creating color map
        my_cmap = plt.get_cmap('hot')
        # Creating plot
        surf = ax.plot_surface(x2, y2, fGrid, cmap=my_cmap,
                                   edgecolor='none')
        fig.colorbar(surf, ax=ax,
                         shrink=0.5, aspect=5)
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('f-axis')
        plt.show()