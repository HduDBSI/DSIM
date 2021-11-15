import torch
import pickle
import numpy as np


def open_pkl_file(path, description, time_span):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        num_user = data['num_user']
        num_item = data['num_item']
        data = data[description]
    time_interval = []
    type_seqs = []
    seq_lens = []
    type_duration = []
    users = []
    time_duration = []
    interacted = dict()
    for i in range(num_user):
        users.append(data[i][0])
        seq_lens.append(len(data[i][1]))
        type_seqs.append(torch.LongTensor([int(event) for event in data[i][1]]))
        time_interval.append(torch.FloatTensor([float(interval) for interval in data[i][2]]))
        time_duration.append([int(dtime) for dtime in data[i][5]])
        interacted[i] = list(set([int(event) for event in data[i][1]]))

    for i in range(num_user):
        min_duration = min(time_duration[i]) if min(time_duration[i]) >= 1 else 1
        type_duration.append(list(map(lambda x: int(x / min_duration), time_duration[i])))
        type_duration[i] = torch.LongTensor(list(map(lambda x: time_span if x > time_span else x, type_duration[i])))

    num_info = [num_user, num_item, time_span]
    data = [users, time_interval, type_seqs, seq_lens, type_duration, interacted]
    return data, num_info


def generate_dataset(data, ratio, window, max_num, data_type):
    users, time_intervals, type_seqs, seq_lens, type_durations, interactions = data
    seq_lens = np.array(seq_lens)
    num = len(seq_lens)

    div_line = int(num * ratio)
    train_interval,  train_type_duration, train_seq, user_train, seq_len_train, target_train \
        = [], [], [], [], [], []
    test_interval,  test_type_duration, test_seq, user_test, seq_len_test, target_test \
        = [], [], [], [], [], []

    for i in range(num):
        lens = seq_lens[i]
        if lens <= max_num:
            ends = np.arange(1, lens)
        else:
            ends = np.arange(lens-max_num, lens)
        if i < div_line:
            for end in ends:
                start = end - window if end - window >= 0 else 0
                train_interval.append(time_intervals[i][start:end])
                train_seq.append(type_seqs[i][start:end])
                seq_len_train.append(end - start)
                train_type_duration.append(type_durations[i][start:end])
                target_train.append(type_seqs[i][end])
                user_train.append(users[i])
        else:
            if data_type:
                ends = ends[-1:]
            for end in ends:
                start = end - window if end - window >= 0 else 0
                test_interval.append(time_intervals[i][start:end])
                test_seq.append(type_seqs[i][start:end])
                seq_len_test.append(end - start)
                test_type_duration.append(type_durations[i][start:end])
                target_test.append(type_seqs[i][end])
                user_test.append(users[i])

    train = [user_train, train_interval, train_seq, seq_len_train, train_type_duration, target_train]

    test = [user_test, test_interval, test_seq, seq_len_test, test_type_duration, target_test]

    return train, test


def padding_full(time_interval, type_train, type_duration, num_info, max_len):
    _, num_item, num_duration = num_info
    num = len(time_interval)
    time_interval_padded = torch.zeros(size=(num, max_len))
    type_train_padded = torch.zeros(size=(num, max_len), dtype=torch.long)
    type_duration_padded = torch.zeros(size=(num, max_len), dtype=torch.long)

    for idx in range(num):
        seq_len = len(time_interval[idx])
        if seq_len < max_len:
            num_padding = max_len - seq_len
            time_interval_padded[idx, num_padding:] = time_interval[idx]
            type_train_padded[idx, num_padding:] = type_train[idx]
            type_train_padded[idx, :num_padding] = num_item
            type_duration_padded[idx, num_padding:] = type_duration[idx]
            type_duration_padded[idx, :num_padding] = num_duration
        else:
            time_interval_padded[idx, :] = time_interval[idx]
            type_train_padded[idx, :] = type_train[idx]
            type_duration_padded[idx, :] = type_duration[idx]

    return time_interval_padded, type_train_padded, type_duration_padded


def gen_neg(interactions, num_item, neg_num):
    count = 0
    k = 0
    neg_item_set = []
    while count < neg_num:
        k += 1
        neg_item = np.random.randint(num_item)
        if neg_item not in interactions and neg_item not in neg_item_set:
            neg_item_set.append(neg_item)
            count += 1
    return neg_item_set


def gen_neg_batch_new2(interactions, users, num_item, neg_num=3):
    num_user = len(users)
    neg_batch = np.zeros((num_user, neg_num))
    for i, uid in enumerate(users):
        neg_batch[i] = gen_neg(interactions[int(uid)], num_item, neg_num)

    return neg_batch


class Data_Batch:
    def __init__(self, users, interval, events, duration, target):
        self.interval = interval
        self.events = events
        self.users_train = users
        self.duration = duration
        self.target = target

    def __len__(self):
        return self.events.shape[0]

    def __getitem__(self, index):
        sample = {
            'event_seq': self.events[index],
            'interval_seq': self.interval[index],
            'users': self.users_train[index],
            'duration_seq': self.duration[index],
            'target': self.target[index]
        }

        return sample
