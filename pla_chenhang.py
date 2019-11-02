#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'Mr.R'

import numpy as np
import matplotlib.pyplot as plt
import time


np.random.seed(1)
sample_num = 1000
lim = 9999
learning_rate = 0.1


def generator(n, t):
    X = np.random.random((n, 2))
    Y = np.zeros(n)
    if t == 'linear':
        Y = np.where(X[:, 0] + 5 * X[:, 1] > 2, 1, -1)
    if t == 'nonlinear':
        Y = np.where((X[:, 0]-0.2)**2 + (X[:, 1]-0.2)**2 > 0.5, 1, -1)
    return X, Y


def plot_scatter(data, t):
    train_X = data[0]
    train_Y = data[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(t)
    plt.xlabel('x')
    plt.ylabel('y')

    idx_1 = np.where(train_Y == 1)
    p1 = ax.scatter(train_X[idx_1, 0], train_X[idx_1, 1],
                    marker='o', color='g', label=1, s=20)
    idx_2 = np.where(train_Y == -1)
    p2 = ax.scatter(train_X[idx_2, 0], train_X[idx_2, 1],
                    marker='x', color='r', label=2, s=20)

    plt.legend(loc='upper right')
    plt.draw()
    plt.pause(3)
    plt.close()


def plot_line(data, t, wb):
    train_X = data[0]
    train_Y = data[1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(t)
    plt.xlabel('x')
    plt.ylabel('y')

    idx_1 = np.where(train_Y == 1)
    p1 = ax.scatter(train_X[idx_1, 0], train_X[idx_1, 1],
                    marker='o', color='g', label=1, s=20)
    idx_2 = np.where(train_Y == -1)
    p2 = ax.scatter(train_X[idx_2, 0], train_X[idx_2, 1],
                    marker='x', color='r', label=2, s=20)

    line_x = np.arange(0, 1, 0.01)
    line_y = -(wb[2]+line_x*wb[0])/wb[1]
    p3 = ax.plot(line_x, line_y, label="line")

    plt.legend(loc='upper right')
    plt.draw()
    plt.pause(3)
    plt.close()

def polt_precision(data):
    x = [i[0] for i in data]
    y = [i[1] for i in data]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('precision')
    plt.xlabel('n')
    plt.ylabel('correct')

    p1 = ax.plot(x, y)
    plt.draw()
    plt.pause(3)
    plt.close()

def POCKET_train(sample_num, t, learning_rate, lim):
    dataSet = generator(sample_num, t)
    train_X = dataSet[0]
    train_Y = dataSet[1]

    w = np.zeros(train_X.shape[1])
    b = 0
    n = 0
    pocket = np.append(w, b)
    temp = []
    while True:
        flag = 1
        for i in range(sample_num):
            a = np.dot(w, train_X[i]) + b
            if train_Y[i] * a <= 0:
                flag = 0
                w = w + train_Y[i] * train_X[i] * learning_rate
                b = b + train_Y[i] * learning_rate

                correct_pocket = sum(
                    np.where(train_Y * (np.dot(pocket[:2], train_X.T) + pocket[2]) > 0, 1, 0))
                correct_new = sum(
                    np.where(train_Y * (np.dot(w, train_X.T) + b) > 0, 1, 0))
                if correct_new > correct_pocket:
                    pocket = np.append(w, b)
                    temp.append([n, float(correct_new/sample_num)])
            n += 1
        if flag == 1 or n > lim:
            print("iterate_num = ", n)
            break

    plot_line(dataSet, t, pocket)
    polt_precision(temp)


if __name__ == "__main__":
    start = time.clock()
    POCKET_train(sample_num, 'nonlinear', learning_rate, lim)
    end = time.clock()
    print('finish all in %s' % str(end - start))