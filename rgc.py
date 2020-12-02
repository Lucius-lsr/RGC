# -*- coding: utf-8 -*-

import numpy as np


class RGC:
    def __init__(self, k, alpha, beta, mu):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.mu = mu

        self.S = None

    def graph_construct(self, X):
        '''

        :param X: data array with size of (n x m), indicating n data with m features
        '''
        Y1 = np.zeros_like(X)
        Y2 = np.zeros_like(X)
        E = np.zeros_like(X)
        Z = np.array(X, copy=True)

        distX = self.L2_distance(X)
        sorted_distX = self.sort_dis(distX)
        gamma = self.cal_gamma(X, sorted_distX, self.beta, self.k)
        for _ in range(1000):
            D = self.update_D(E, X, Y1, Y2, self.mu, Z)
            distD = self.L2_distance(D)
            s_distD, s_idx = self.sort_dis(distD)
            gamma = self.cal_gamma(D, s_distD, self.beta, self.k)
            E = self.update_E(D, X, Y1, self.mu, self.alpha)
            S = self.update_S(X, s_distD, s_idx, self.k)
            S = (S + np.transpose(S))/2
            L = np.diag(np.sum(S, axis=0)) - S
            Z = self.update_Z(L, self.beta, self.mu, D, Y2)
            Y1 = Y1 + self.mu * (D + E - X)
            Y2 = Y2 + self.mu * (D - Z)
            print(np.linalg.norm(D + E - X, ord=1), np.linalg.norm(D - Z, ord=1))
            if np.linalg.norm(D + E - X, ord=1) < 1e-5 and np.linalg.norm(D - Z, ord=1) < 1e-5:
                self.S = S
                break

    @staticmethod
    def L2_distance(A):
        if A.shape[1] == 1:
            A_expansion = np.zeros((A.shape[0], 1))
            A = np.hstack((A, A_expansion))
        M = np.dot(A, A.T)
        t = np.diag(np.dot(A, A.T))
        t = np.reshape(np.repeat(t, M.shape[0]), M.shape)
        sq = -2 * M + t + t.T
        return sq

    @staticmethod
    def sort_dis(distX):
        '''

        :param distX: an (n x n) symmetric distance matrix
        :return:
        '''
        sorted_disX = np.sort(distX, axis=1)
        sorted_idx = np.argsort(distX, axis=1)
        return sorted_disX, sorted_idx

    @staticmethod
    def cal_gamma(X, sorted_distX, beta, k):
        '''

        :param X: matrix X with size of (n x m)
        :param sorted_distX: sorted distance matrix with size of (n x n)
        :param beta: hyper-parameter beta
        :param k: hyper-parameter k
        :return:
        '''
        data_num = X.shape[0]
        oneRow = np.zeros(data_num)
        oneRow[1:k + 1] = -1
        oneRow[k + 1] = k
        oneRow = np.tile(oneRow, (data_num, 1))
        gamma = (beta / (4 * data_num)) * np.sum(sorted_distX * oneRow)
        return gamma

    @staticmethod
    def update_D(E, X, Y1, Y2, mu, Z):
        '''

        :param E: matrix E with size of (n x m)
        :param X: matrix X with size of (n x m)
        :param Y1: matrix Y1 with size of (n x m)
        :param Y2: matrix Y2 with size of (n x m)
        :param mu: hyper-parameter mu
        :param Z: matrix Z with size of (n x m)
        :return: new D
        '''
        H = (X + Z - E - (Y1 + Y2) / mu) / 2
        U, sigma, Vt = np.linalg.svd(H)
        sigma = np.maximum(sigma - 1 / (2 * mu), 0)
        row = U.shape[0]
        col = Vt.shape[0]
        sing_num = len(sigma)
        smat = np.zeros((row, col))
        smat[:sing_num, :sing_num] = np.diag(sigma)
        D = np.matmul(np.matmul(U, smat), Vt)
        return D

    @staticmethod
    def update_E(D, X, Y1, mu, alpha):
        '''

        :param D: matrix D with size of (n x m)
        :param X: matrix X with size of (n x m)
        :param Y1: matrix Y1 with size of (n x m)
        :param mu: hyper-parameter mu
        :param alpha: hyper-parameter alpha
        :return: E
        '''
        G = X - D - Y1 / mu
        signG = np.copy(G)
        signG[np.where(G > 0)] = 1
        signG[np.where(G < 0)] = -1
        E = np.maximum(np.abs(G) - alpha / mu, 0) * signG
        return E

    @staticmethod
    def update_S(X, sorted_distX, sorted_idx, k):
        '''

        :param X:  matrix X with size of (n x m)
        :param sorted_distX: sorted distance matrix with size of (n x n)
        :param sorted_idx: distance sorting index, the first of each row is always the row number, with size of (n x n)
        :param k: hyper-parameter k
        :return: new S
        '''
        n = X.shape[0]
        S = np.zeros((n, n))
        for i in range(n):
            dis_i = sorted_distX[i][1:k + 2]  # dis_i has k+1 elements
            sum_k = np.sum(dis_i[:-1])
            for j in range(k):
                S[i][sorted_idx[i][j + 1]] = (dis_i[k] - dis_i[j]) / (k * dis_i[k] - sum_k)

        return S

    @staticmethod
    def update_Z(L, beta, mu, D, Y2):
        '''

        :param L: matrix L with size of (n x m)
        :param beta: hyper-parameter beta
        :param mu: hyper-parameter mu
        :param D: matrix D with size of (n x m)
        :param Y2: matrix Y2 with size of (n x m)
        :return: new Z
        '''
        L_size = L.shape[0]
        Z = np.dot(np.linalg.inv(2 * beta * L + mu * np.identity(L_size)), mu * D + Y2)
        return Z
