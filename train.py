import argparse
import utils
import os
import datetime
import time
from torch.utils.data import DataLoader
import fusion
import torch
from metrics import *
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluation(model, test, topk=5):
    test_interval, type_test, seq_lens_test, test_duration, target_test = test
    n_test = test_interval.shape[0]
    batch_size = 64
    pred_list = None
    item_index = np.arange(model.num_item)

    logger.info("Total testing records:{}".format(n_test))

    num_batch = int(n_test / batch_size) + 1
    for batch_idx in range(num_batch):
        start = batch_idx * batch_size
        end = start + batch_size

        if batch_idx == num_batch - 1:
            if start < n_test:
                end = n_test
            else:
                break

        batch_interval = test_interval[start:end].to(device)
        batch_type = type_test[start:end].to(device)
        batch_duration = test_duration[start:end].to(device)

        batch = (batch_type, batch_interval, batch_duration)

        predict_items = torch.from_numpy(item_index).type(torch.LongTensor).to(device)

        predict_score = model.train_batch(batch, predict_items, config.interest_using)
        predict_score = predict_score.cpu().data.numpy().copy()

        ind = np.argpartition(predict_score, -topk)
        ind = ind[:, -topk:]
        arr_ind = predict_score[np.arange(len(predict_score))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(predict_score)), ::-1]
        batch_pred_list = ind[np.arange(len(predict_score))[:, None], arr_ind_argsort]

        if batch_idx == 0:
            pred_list = batch_pred_list
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
        # print(pred_list.shape)
    topk = [1, 3, 5, 10, 20]
    recall, mrr, ndcg = metric(target_test, pred_list, topk)

    return recall, mrr, ndcg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model..")

    parser.add_argument("--dataset", type=str, default='YFCC')
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=30, help="maximum epochs")
    parser.add_argument("--seq_len", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch_size for each train iteration")
    parser.add_argument("--used_past_model", default=False, type=bool, help="True to use a trained model named model.pt")
    parser.add_argument('--ratio', type=float, default=0.8, help="ratio for train data")
    parser.add_argument("--neg_num", type=int, default=3, help="the number of negative sampling")
    parser.add_argument("--hid_dim", type=int, default=128, help="the dimension of embedding + rnn unit")
    parser.add_argument('--beta', type=float, default=0.1, help='hyper parameter of softplus function')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of forward in self attention')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_false', help='whether to output attention in encoder')
    parser.add_argument('--d_fcn', type=int, default=512, help='dimension of fcn in the final step of fusion')
    parser.add_argument('--activation_out', type=str, default='relu', help='activation of fcn in the final '
                                                                           'step of fusion')
    parser.add_argument('--dl', type=int, default=30, help='sequence length for capturing dynamic interest')
    parser.add_argument('--use_duration', type=bool, default=True, help='whether to use duration in static interest')
    parser.add_argument('--mix', type=bool, default=False, help='whether to mix the item representation and side '
                                                                'information in static interest')
    parser.add_argument('--interest_using', type=str, default='both', help='interest using in the final step, '
                                                                           'options:[dynamic, static, both]')
    parser.add_argument('--window', type=int, default=200, help='window size')
    parser.add_argument('--max_num', type=int, default=10)
    parser.add_argument('--data_type', type=bool, default=1)
    parser.add_argument('--time_span', type=int, default=256)
    parser.add_argument('--loss', type=str, default='both')
    parser.add_argument('--alpha', type=float, default=0.1)

    config = parser.parse_args()
    print('use_duration: ', config.use_duration)

    if config.dataset[:4] == 'YFCC':
        config.batch_size = 128
        config.window = 20
        config.dl = 5
        config.n_heads = 2
        config.lr = 0.0001
        config.neg_num = 1
        config.max_num = 500
        config.num_epochs = 100
        config.data_type = 0

    if config.dataset[:7] == 'game':
        config.dl = 25
        config.window = 100

    now = str(datetime.datetime.today()).split()
    now = now[0] + "-" + now[1][:5]
    id_process = os.getpid()
    print("id: " + str(id_process))
    t1 = time.time()
    print("Processing data...")
    file_path = 'data/' + config.dataset + '/' + config.dataset + '.pkl'
    print(file_path)
    data, num_info = utils.open_pkl_file(file_path, 'data',  config.time_span)

    num_user, num_item,  num_duration = num_info

    log_file_name = "train_process" + '_' + config.dataset + '_' + config.interest_using + '_' \
                    + '_' + config.loss + '_' + str(config.alpha) + '_' + str(config.max_num) + '_' + \
                    '_' + str(config.hid_dim) + '_'+ str(config.window) + '_' + str(config.dl) + '_' + str(config.lr) \
                    + '_' + str(config.mix + 0) + '_' + str(config.use_duration + 0) + '_' + str(config.e_layers)\
                    + '_' + str(config.dropout) + '_' + str(num_duration) + '_' + str(config.neg_num) + '_' + \
                    str(id_process) + ".txt"

    log = open(log_file_name, 'w')
    log.write("Data when training: " + str(datetime.datetime.now()))
    log.write("\nTraining-id: " + str(id_process))
    log.write("\nTraining data: " + config.dataset)
    log.write("\nLearning rate: " + str(config.lr))
    log.write("\nMax epochs: " + str(config.num_epochs))
    log.write("\nseq lens: " + str(config.seq_len))
    log.write("\nbatch size for train: " + str(config.batch_size))
    log.write("\nuse previous model: " + str(config.used_past_model))
    log.write("\ndropout: " + str(config.dropout))
    log.write("\nlength for dynamic interest: " + str(config.dl))
    log.write("\nwindow_size:" + str(config.window))
    log.write("\nmix: " + str(config.mix))
    log.write("\ninterest_using: " + config.interest_using)
    log.write("\nmax_num:" + str(config.max_num))
    log.write("\nhid_dim:" + str(config.hid_dim))
    log.write("\nnum_head:" + str(config.n_heads))
    log.write("\nloss:" + config.loss)
    log.write("\nneg_num:" + str(config.neg_num))
    log.write("\n")

    train, test = utils.generate_dataset(data, config.ratio, config.window, config.max_num, config.data_type)

    users_train, train_interval, type_train, seq_lens_train, train_duration, target_train = train
    users_test, test_interval, type_test, seq_lens_test, test_duration, target_test = test

    train_interval, type_train, train_duration = utils.padding_full(train_interval, type_train, train_duration,
                                                                     num_info, config.window)
    test_interval, type_test, test_duration = utils.padding_full(test_interval, type_test, test_duration, num_info,
                                                                  config.window)

    test = (test_interval, type_test, seq_lens_test, test_duration, target_test)

    print('train:', len(train_interval))
    print('test:', len(test_interval))

    train_data = utils.Data_Batch(users_train, train_interval, type_train, train_duration, target_train)
    train_data = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    print("Data Processing Finished...")
    t2 = time.time()
    data_process_time = t2 - t1
    print("Getting data takes: " + str(data_process_time) + " seconds")
    log.write("\n\nGetting data takes: " + str(data_process_time) + " seconds")

    print("start training...")
    t3 = time.time()
    settings = {
        'num_item': num_item,
        'num_user': num_user,
        'num_duration': num_duration,
        'lr': config.lr,
        'device': device,
        'beta': config.beta,
        'hid_dim': config.hid_dim,
        'n_heads': config.n_heads,
        'e_layers': config.e_layers,
        'dropout': config.dropout,
        'activation': config.activation,
        'output_attention': config.output_attention,
        'd_ff': config.d_ff,
        'batch_size': config.batch_size,
        'd_fcn': config.d_fcn,
        'activation_out': config.activation_out,
        'dl': config.dl,
        'use_duration': config.use_duration,
        'mix': config.mix,
        'max_len': config.window
    }
    if config.used_past_model:
        model = torch.load("model.pt")
    else:
        model = fusion.Fusion(settings)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loss_value = []

    type_accuracy_list = []
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    epoch_loss = 0.0
    for i in range(config.num_epochs):
        t5 = time.time()
        max_len = len(train_data)
        for idx, a_batch in enumerate(train_data):
            intervals, type_items, durations, users, targets = a_batch['interval_seq'], a_batch['event_seq'], \
                                                               a_batch['duration_seq'], a_batch['users'], \
                                                               a_batch['target']
            optimizer.zero_grad()

            type_items = type_items.to(device)
            intervals = intervals.to(device)
            durations = durations.to(device)
            targets = torch.unsqueeze(torch.LongTensor(targets), dim=1)
            neg = utils.gen_neg_batch_new2(data[-1], users, num_item, config.neg_num)
            neg = torch.LongTensor(neg)
            predict_items = torch.cat((targets, neg), 1).to(device)
            batch = (type_items, intervals, durations)

            score = model.train_batch(batch,  predict_items, config.interest_using)
            (target_prediction, neg_prediction) = torch.split(score, [1, config.neg_num], dim=1)

            target_labels = torch.ones(target_prediction.shape, device=device)
            neg_labels = torch.zeros(neg_prediction.shape, device=device)

            if config.loss == 'bpr':
                # bpr loss
                loss = -torch.log(torch.sigmoid(target_prediction - neg_prediction) + 1e-8)
                loss = torch.mean(torch.sum(loss, dim=1))
            elif config.loss == 'bce':
                # bce loss
                loss = bce_criterion(target_prediction, target_labels)
                loss += bce_criterion(neg_prediction, neg_labels)
            else:
                config.alpha = 0 if i % 2 == 0 else 1
                loss1 = -torch.log(torch.sigmoid(target_prediction - neg_prediction) + 1e-8)
                loss1 = torch.mean(torch.sum(loss1, dim=1))
                loss2 = bce_criterion(target_prediction, target_labels)
                loss2 += bce_criterion(neg_prediction, neg_labels)
                loss = config.alpha * loss1 + (1 - config.alpha) * loss2

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            if (idx+1) % 500 == 0:
                print("In epochs {0}, process {1} over {2} is done".format(i, idx+1, max_len))
        epoch_loss /= max_len
        print("epoch_loss:{}".format(epoch_loss))
        log.write("\nepoch:{}".format(i))
        log.write("\nepoch training time:{}".format(time.time() - t5))
        log.write("\nepoch_loss:{}".format(epoch_loss))
        loss_value.append(epoch_loss)
        print("model saved..")
        torch.save(model, "model.pt")
        if (i + 1) % 1 == 0:
            start_time = time.time()
            logger.info('eval ...')
            log.write("\neval ...")
            model.eval()

            recall, mrr, ndcg = evaluation(model, test, topk=20)
            logger.info('recall:{}'.format(recall))
            logger.info('mrr:{}'.format(mrr))
            logger.info('ndcg:{}'.format(ndcg))
            logger.info("Evaluation time:{}".format(time.time() - start_time))
            log.write('\nrecall:{}'.format(recall))
            log.write('\nmrr:{}'.format(mrr))
            log.write('\nndcg:{}'.format(ndcg))
            log.write("\nEvaluation time:{}".format(time.time() - start_time))
            log.write("\n")

    print(loss_value)
    t4 = time.time()
    training_time = t4 - t3
    print("training done..")
    print("training takes {0} seconds".format(training_time))
    log.write("\ntraining takes {0} seconds".format(training_time))
    log.write('\nbpr_loss: ')
    log.writelines(str(loss) + " " for loss in loss_value)
    log.close()

    print("Every works are done!")
