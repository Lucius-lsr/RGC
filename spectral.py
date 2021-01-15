import math

import numpy as np
from sklearn.cluster import KMeans
from Hungarian import Hungarian

from data import get_data
from rgc import RGC


def calLapMat(S):
    """
    计算拉普拉斯矩阵
    :param S: adjacent matrix with size n*n
    :return: Laplacian matrix L
    """
    D = np.sum(S, axis=1)
    L = np.diag(D) - S
    sqrtD = np.diag(1.0 / (D ** 0.5))
    L = np.dot(np.dot(sqrtD, L), sqrtD)
    return L


def L2Dist(a, b):
    """
    calculate the L2 distance between a and b
    """
    return np.sqrt(np.sum((np.abs(a - b)) ** 2))


def get_closest_dist(point, centroids):
    """
    :param point: a vector of k dims
    :param centroids: a t * k matrix representing t center points
    :return:
    """
    min_dist = math.inf  # 初始设为无穷大
    k = centroids.shape[0]
    for i in range(k):
        dist = L2Dist(centroids[i, :], point)
        if dist < min_dist:
            min_dist = dist
    return min_dist ** 2


def kpp_centers(data_set):
    """
    choose k centers from data_set
    """
    n, k = data_set.shape
    first_row = np.random.choice(np.arange(n), size=1, replace=False)
    cluster_centers = data_set[first_row, :]
    d = np.zeros(n)
    for _ in range(1, k):
        total = 0.0
        for i in range(n):
            d[i] = get_closest_dist(data_set[i, :], cluster_centers)  # 与最近一个聚类中心的距离
            total += d[i]
        d = d / total
        # 选出下一个中心
        next_row = np.random.choice(np.arange(n), size=1, replace=False, p=d)
        cluster_centers = np.insert(cluster_centers, cluster_centers.shape[0], data_set[next_row, :], axis=0)
    return cluster_centers


def kmeans(A):
    """
    :param A: data set matrix (n * k), k is cluster number
    :return: k centers and a n * 2 matrix, the first row is the cluster num the point belongs to,
    the second row is the dist to the cluster center
    """
    n, k = A.shape
    clusterResult = np.zeros((n, 2))
    needUpdate = True

    # 初始化簇中心
    centers = kpp_centers(A)

    # 循环更新
    while needUpdate:
        needUpdate = False
        for i in range(n):
            minDist = 9999999.0
            minIdx = -1
            for j in range(k):
                dist = L2Dist(A[i, :], centers[j, :])
                if dist < minDist:
                    minDist = dist
                    minIdx = j
            if clusterResult[i, 0] != minIdx:
                needUpdate = True
                clusterResult[i, :] = minIdx, minDist ** 2
        # 更新簇中心
        for i in range(k):
            clusterPoints = A[np.nonzero(clusterResult[:, 0] == j)[0]]
            centers[j:] = np.mean(clusterPoints, axis=0)
    return centers, clusterResult


def spectral(S, k):
    """
    :param S: data set
    :param k: cluster number
    :return:
    """
    # 计算拉普拉斯矩阵
    L = calLapMat(S)

    # 奇异值分解
    H, lam, V = np.linalg.svd(L)

    # 取特征值最大的前k个特征向量
    H = H[:, np.argsort(lam)]
    H = H[:, :k]

    # 行单位化
    tmp = np.sum(np.abs(H) * np.abs(H), axis=1).reshape(-1, 1)
    H = H / (tmp ** 0.5)

    # kmeans聚类
    center, result = kmeans(H)

    # skLearn调包结果
    sklResult = KMeans(n_clusters=k).fit(H)

    return result[:, 0].reshape(1, -1)[0, :].astype(np.int), sklResult.labels_


def evaluatePurity(result, label, k):
    """
    :param result: the result of clustering 1*n
    :param label: the true label of data 1*n
    :param k: the number of clusters
    :return: correctness of clustering
    """
    correct = 0
    for i in range(k):
        vote = [0] * k
        idx = np.where(result == i)
        truth = label[idx]
        for j in truth:
            vote[j - 1] += 1
        correct += np.max(vote)
    return correct / len(result)


def evaluateAcc(result, label, k):
    """
    :param result: the result of clustering 1*n
    :param label: the true label of data 1*n
    :param k: the number of clusters
    :return: accuracy of clustering
    """
    cost_matrix = np.zeros((k, k))
    for i in range(0, len(result)):
        cost_matrix[result[i]][label[i]] += 1
    hungarian = Hungarian(cost_matrix, is_profit_matrix=True)
    hungarian.calculate()
    mapping = [0] * k
    hung_result = hungarian.get_results()
    for r in hung_result:
        mapping[r[0]] = r[1]

    correct = 0
    for i in range(0, len(result)):
        if label[i] == mapping[result[i]]:
            correct += 1
    return correct / len(result)


def NMI(result, label):
    """
    :param result: the result of clustering 1*n
    :param label: the true label of data 1*n
    :return: the NMI between two clusters
    """
    result = result + 1
    total = len(result)
    A_ids = set(result)
    B_ids = set(label)
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(result == idA)
            idBOccur = np.where(label == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps, 2)
    # Normalized Mutual information
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(result == idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps, 2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(label == idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps, 2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat


dataset = 'coil20'
# 加载数据
x, y, num_class = get_data(dataset, 0)

model = RGC(5, 0.0385, 0.1, 15)
model.graph_construct(x)

S = model.S
result, skResult = spectral(S, num_class)

# 评估聚类结果
print('Purity by myKMeans: ' + str(evaluatePurity(result, y, num_class)))
print('Accuracy by myKMeans: ' + str(evaluateAcc(result, y, num_class)))
print('NMI by myKMeans: ' + str(NMI(result, y)))

print('Purity by skLearn: ' + str(evaluatePurity(skResult, y, num_class)))
print('Accuracy by skLearn: ' + str(evaluateAcc(skResult, y, num_class)))
print('NMI by skLearn: ' + str(NMI(skResult, y)))
