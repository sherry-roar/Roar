#!/usr/bin/env python 
# -*- coding:utf-8 -*-

__author__ = 'Mr.R'


import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
# import sklearn
import matplotlib.pyplot as plt
# import pylab as pl
import time



np.random.seed(1)


sample_num=1000
t='nonlinear'
k=5
c=[1e-7,1e-5,1e-3,1e-1]

def generator(n, t):
    X = 10*np.random.random((n, 2))
    Y = np.zeros(n)
    if t == 'linear':
        Y = np.where(X[:, 0] -  X[:, 1] > 0, 1,-1)
    if t == 'nonlinear':
        Y = np.where((X[:, 0]-4)**2 + X[:, 1]-8 >0 , 1, -1)
    return X, Y

def cross_va(X,Y,k,i):
    # k fold cross validate model
    data=np.column_stack((X,Y))
    np.random.seed(5)
    np.random.shuffle(data)
    # m,n=np.shape(X)
    # start=(i-1)/k*m
    # end=(i*m/k)
    # sl=slice(0,20,1)
    # data_train=data[sl,:]
    kf = KFold( n_splits=k, shuffle=False)
    kf.get_n_splits(data)
    for (train_index, test_index) in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        # print(X_train, X_test)
        # print(Y_train, Y_test)
    return X_test,X_train,Y_test,Y_train


def show_acc(y_hat,y,t):
    num=np.shape(y)
    err=sum(y_hat-y)
    acc=err/num
    if t=='train':
        print('train accuracy=',acc)
    if t=='test':
        print('test accuracy=', acc)

def plot_scatter(x, y , t):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(t)
    plt.xlabel('x')
    plt.ylabel('y')

    idx_1 = np.where(y == 1)
    p1 = ax.scatter(x[idx_1, 0], x[idx_1, 1],
                    marker='o', color='g', label=1, s=20)
    idx_2 = np.where(y == -1)
    p2 = ax.scatter(x[idx_2, 0], x[idx_2, 1],
                    marker='x', color='r', label=2, s=20)

    plt.legend(loc='upper right')
    plt.draw()
    plt.pause(3)
    plt.close()


def plot_line(x,y, tt):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_title(tt)
    idx_1 = np.where(y == 1)
    p1 = ax.scatter(x[idx_1, 0], x[idx_1, 1],
                    marker='o', color='g', label=1, s=20)
    idx_2 = np.where(y == -1)
    p2 = ax.scatter(x[idx_2, 0], x[idx_2, 1],
                    marker='x', color='r', label=2, s=20)


    if t=='linear' :
        line_x = np.arange(0, 10, 0.1)
        line_y = line_x
        p3 = ax.plot(line_x, line_y, label="line")

    elif t=='nonlinear':
        line_x = np.arange(1, 7, 0.1)
        line_y = -(line_x-4)**2+8
        p3 = ax.plot(line_x, line_y, label="line")


    plt.legend(loc='upper right')
    plt.draw()
    plt.pause(3)
    plt.close()



def svm_test(x_train, x_test, y_train, y_test,c):
    # X, Y is training data,c is Penalty factor

    clf=svm.SVC(C=c,kernel='rbf',decision_function_shape='ovo')
    clf.fit(x_train,y_train)
    train_acc=clf.score(x_train, y_train)
    test_acc=clf.score(x_test, y_test)
    # print('train accuracy=', train_acc) # 精度
    # y_hat_train = clf.predict(x_train)
    # plot_line(x_train, y_hat, 'train')
    # show_acc(y_hat, y_train, 'train')
    # print('test accuracy=',test_acc)
    y_hat_test = clf.predict(x_test)
    # plot_line(x_test, y_hat, 'test')
    # show_acc(y_hat, y_test, 'test')
    total_acc=(train_acc+test_acc)/2
    # w = clf.coef_[0]
    # a = -w[0] / w[1]
    # b=clf.intercept_[0]
    # xx = np.linspace(-5, 5)
    # yy = a * xx - intercept / w[1]

    # plot the parallels to the separating hyperplane that pass through the support vectors
    # a=clf.support_vectors_[:]
    # b = clf.support_vectors_[0]
    # yy_down = a * xx + (b[1] - a * b[0])
    # b = clf.support_vectors_[-1]
    # yy_up = a * xx + (b[1] - a * b[0])

    # print("w: ", w)
    # print("a: ", a)

    # print "xx: ", xx
    # print "yy: ", yy
    # print("support_vectors_: ", clf.support_vectors_)
    # print("clf.coef_: ", clf.coef_)
    # plot the line, the points, and the nearest vectors to the plane
    # pl.plot(xx, yy, 'k-')
    # pl.plot(xx, yy_down, 'k--')
    # pl.plot(xx, yy_up, 'k--')
    #
    # pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
    #            s=80, facecolors='none')
    # pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
    #
    # pl.axis('tight')
    # pl.show()


    return total_acc,w,a,b,y_hat_test





if __name__ == "__main__":
    start = time.clock()
    X,Y=generator(sample_num, t)
    plot_line(X, Y, t)
    # svm_test(X,Y)
    # data=cross_va(X,Y,5,1)
    data=np.column_stack((X,Y))
    np.random.seed(5)
    np.random.shuffle(data)
    kf = KFold( n_splits=k, shuffle=False)
    kf.get_n_splits(data)

    # factor initiate
    acc=0
    a=0
    b=0
    w=0

    for (train_index, test_index) in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        for i in c:

            acct,w_t,a_t,b_t,y_hat_test=svm_test(X_train,X_test,Y_train,Y_test,i)
            # print(acct)
            if acc > acct :
                continue
            else :
                acc=acct
                # w,a,b=w_t,a_t,b_t
                XX=X_test
                YY=y_hat_test
                cc=i

    print('Total accuracy =',acc)
    print('C =',cc)
    plot_line(XX, YY, 'test')



    # xx = np.linspace(1, 7)
    # yy = -(w[0]/w[1]) * xx - b / w[1]
    # a1 = a[0]
    # yy_down = -(w[0]/w[1]) * xx + (a1[1] +(w[0]/w[1]) * a1[0])
    # a2 = a[-1]
    # yy_up = -(w[0]/w[1]) * xx + (a2[1] +(w[0]/w[1]) * a2[0])
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # ax.set_title('final')
    # plt.plot(xx, yy, 'k-')
    # plt.plot(xx, yy_down, 'k--')
    # plt.plot(xx, yy_up, 'k--')
    # idx_1 = np.where(YY == 1)
    # p1 = ax.scatter(XX[idx_1, 0], XX[idx_1, 1],
    #                 marker='o', color='g', label=1, s=20)
    # idx_2 = np.where(YY == -1)
    # p2 = ax.scatter(XX[idx_2, 0], XX[idx_2, 1],
    #                 marker='x', color='r', label=2, s=20)
    # plt.scatter(a[:, 0], a[:, 1],
    #             s=80, facecolors='none')
    # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)



    plt.legend(loc='upper right')
    plt.draw()
    plt.pause(3)
    plt.close()



    end = time.clock()
    print('finish all in %s' % str(end - start))

