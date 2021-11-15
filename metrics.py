import numpy as np
import math


def recall_k(targets, predicts, k):
    sum_recall = 0.0
    num_users = len(targets)
    for i in range(num_users):
        target = targets[i]
        predict = list(predicts[i][:k])
        if target in predict:
            sum_recall += 1

    return sum_recall / num_users


def ndcg_k(targets, predicts, k):
    res = 0
    num_users = len(targets)
    for i in range(num_users):
        idcg = idcg_k(k)
        dcg_k = sum([int(predicts[i][j] in set(predicts[i])) / math.log(j + 2, 2) for j in range(k)])
        res += dcg_k / idcg

    return res / float(len(targets))


def idcg_k(k):
    res = sum([1.0 / math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def mrr_k(targets, predicts, k):
    sum_mrr = 0.0
    num_user = len(targets)
    # print(num_user)
    for uid in range(num_user):
        predict = list(predicts[uid][:k])
        target = targets[uid]
        if target in predict:
            gnd_rate = predict.index(target) + 1
            sum_mrr += 1. / gnd_rate

    return sum_mrr / num_user


def precision_k(targets, predicts, k):
    sum_precision = 0.0
    num_user = len(targets)
    for i in range(num_user):
        target = targets[i]
        predict = predicts[i][:k]
        if target in predict:
            sum_precision += 1 / float(k)

    return sum_precision / num_user


def metric(targets, predicts, topk):
    recall = np.zeros(len(topk))
    mrr = np.zeros(len(topk))
    ndcg = np.zeros(len(topk))

    num_user = len(targets)
    for uid in range(num_user):
        predict = list(predicts[uid][:topk[-1]])
        target = targets[uid]
        if target in predict:
            rank = predict.index(target)
            for i, k in enumerate(topk):
                if rank < k:
                    recall[i] += 1
                    mrr[i] += 1. / (rank + 1)
                    ndcg[i] += 1. / np.log2(rank + 2)

    return recall / num_user, mrr / num_user, ndcg / num_user

