# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
import random 
import pickle
import json
from imblearn.over_sampling import RandomOverSampler
from fb_ocsvm_class import FB_OCSVM

from sklearn import svm

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_auc_score

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(config_file, args_dict):
    with open(config_file) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict


def load_MNIST(test_task_idx, val_task_idx):

    """ load the required MT-MNIST dataset.

        Parameters
        ----------
        test_task_idx : int
            index of the meta-testing task.
        val_task_idx : int
            index of the meta-validation task.


        Returns
        -------
        mtl_train_tasks : dict
            meta-training tasks.
        val_task : dict
            meta-validation task.
        test_task_finetune_sets : dict
            list of different training sets sampled from the task considered.
        test_task_test_set : dict
            test set of the meta-testing task.

    """

    base_path = '/home/USER/Documents'
    if (not (os.path.exists(base_path))):
        base_path = '/home/ubuntu/Projects'
    data_path = base_path + '/MAML/input_data/FB_MNIST_val' + str(val_task_idx) + '_test' + str(test_task_idx)+'/'
    train_data_file = open(data_path + 'fb_train_data_val_' + str(val_task_idx) + '_test_' + str(test_task_idx) + '.txt', 'rb')
    train_data = pickle.load(train_data_file)

    data_path = base_path + '/MAML/input_data/MNIST_val' + str(val_task_idx) + '_test' + str(test_task_idx)+'/'


    val_task_file = open(data_path + 'val_task_val_'+ str(val_task_idx) + '_test_' + str(test_task_idx) + '.txt', 'rb')
    val_task = pickle.load(val_task_file)


    test_task_test_set_file = open(data_path + 'test_task_test_sets_val_'+ str(val_task_idx) + '_test_' + str(test_task_idx) + '.txt', 'rb')
    test_task_test_set = pickle.load(test_task_test_set_file)

    test_task_finetune_sets_file = open(data_path + 'test_task_finetune_sets_val_'+ str(val_task_idx) + '_test_' + str(test_task_idx) + '.txt', 'rb')
    test_task_finetune_sets = pickle.load(test_task_finetune_sets_file)

    return train_data, val_task, test_task_finetune_sets, test_task_test_set




def initialize_ocsvm(kernel, nu, gamma, **kwargs):

    if kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
        kernel = kernel
    else:
        kernel = 'precomputed'

    ocsvm = svm.OneClassSVM(kernel=kernel, nu=nu, gamma=gamma,**kwargs)
    return ocsvm

def train(ocsvm, X_train, X_test, Y_test, kernel, nu, GridSearch=True, **kwargs):

    if X_train.ndim > 2:
        X_train_shape = X_train.shape
        X_train = X_train.reshape(X_train_shape[0], np.prod(X_train_shape[1:]))
    else:
        X_train = X_train

    if kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
        # get_kernel_matrix(kernel=kernel, X_train=X_train, **kwargs)
        # svm.fit(K_train)
        print('unexpected behaviour')
    else:
        if GridSearch and kernel == 'rbf':

            # use grid search cross-validation to select gamma
           # print("Using GridSearchCV for hyperparameter selection...")

            # sample small hold-out set from test set for hyperparameter selection. Save as val set.
            
            n_test_set = len(X_test)
            n_val_set = int(0.1 * n_test_set)
            n_test_out = 0
            n_test_norm = 0
            n_val_out = 0
            n_val_norm = 0
            while (n_test_out == 0) | (n_test_norm == 0) | (n_val_out == 0) | (n_val_norm ==0):
                perm = np.random.permutation(n_test_set)
                X_val = X_test[perm[:n_val_set]]
                y_val = Y_test[perm[:n_val_set]]
                # only accept small test set if AUC can be computed on val and test set
                n_test_out = np.sum(Y_test[perm[:n_val_set]])
                n_test_norm = np.sum(Y_test[perm[:n_val_set]] == 0)
                n_val_out = np.sum(Y_test[perm[n_val_set:]])
                n_val_norm = np.sum(Y_test[perm[n_val_set:]] == 0)

            X_test = X_test[perm[n_val_set:]]
            Y_test = Y_test[perm[n_val_set:]]
            n_val = len(y_val)
            n_test_set = len(Y_test)

            val_scores = np.zeros((len(y_val), 1))
            test_scores = np.zeros((len(Y_test), 1))

            cv_auc = 0.0
            cv_acc = 0
            cv_f1 = 0

            g_best = 0.1
            for gamma in np.logspace(-10, -1, num=10, base=2):

                # train on selected gamma
                cv_svm = svm.OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
                cv_svm.fit(X_train)

                # predict on small hold-out set
                val_acc, _, _, _, val_f1_score, val_auc_roc = predict(cv_svm, X_val, y_val, kernel)

                # save model if AUC on hold-out set improved
                if val_f1_score > cv_f1:
 #                   print('gamma set to: ', g_best)
                    ocsvm = cv_svm
                    g_best = gamma
                    cv_auc = val_auc_roc
                    cv_f1 = val_f1_score

            # save results of best cv run
            # diag['val']['auc'] = cv_auc
            # diag['val']['acc'] = cv_acc

            oc_svm = svm.OneClassSVM(kernel='rbf', nu=nu, gamma=g_best)
 

            ocsvm.fit(X_train)


        else:
            # if rbf-kernel, re-initialize svm with gamma minimizing the
            # numerical error
            if kernel == 'rbf':
                gamma = 1 / (np.max(pairwise_distances(X_train)) ** 2)
                # ocsvm = svm.OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)

            ocsvm.fit(X_train)
            gamma = gamma

    return ocsvm



def predict(ocsvm, X, y, kernel, **kwargs):

    # reshape to 2D if input is tensor
    if X.ndim > 2:
        X_shape = X.shape
        X = X.reshape(X_shape[0], np.prod(X_shape[1:]))

    if kernel in ('DegreeKernel', 'WeightedDegreeKernel'):
        # get_kernel_matrix(kernel=kernel, which_set=which_set, **kwargs)
        # if which_set == 'train':
        #     scores = (-1.0) * ocsvm.decision_function(K_train)
        #     y_pred = (ocsvm.predict(K_train) == -1) * 1
        # if which_set == 'test':
        #     scores = (-1.0) * ocsvm.decision_function(K_test)
        #     y_pred = (ocsvm.predict(K_test) == -1) * 1
        print('unexpected behaviour')

    else:
        scores = (-1.0) * ocsvm.decision_function(X)
        y_pred = ocsvm.predict(X)

        y_pred[y_pred == 1.0] = 0.0
        y_pred[y_pred == -1.0] = 1.0

    scores_flattened = scores.flatten()
    acc = 100.0 * sum(y == y_pred) / len(y)

    TP = np.count_nonzero(y_pred * y)
    TN = np.count_nonzero((y_pred - 1) * (y - 1))
    FP = np.count_nonzero(y_pred* (y - 1))
    FN = np.count_nonzero((y_pred-1) *y)

    if(TP+FP == 0):
        prec = 0.0
    else:
        prec = TP/(TP + FP) 

    rec = TP / (TP + FN)
    spec = TN / (TN + FP)

    if(prec+rec == 0):
        f1_score = 0.0
    else:
        f1_score = 2*prec*rec/(prec + rec)

    # if sum(y) > 0:
    auc_roc = roc_auc_score(y, scores.flatten())
        
    # if which_set == 'test':
    #     rho = -svm.intercept_[0]

    return acc, prec, rec, spec, f1_score, auc_roc



def sample_random_val_finetune_data(val_task, K, cir, val_normal_indexes, val_anomalous_indexes):
    """samples K datapoints from the validation task.

    Parameters
    ----------
    val_task : dict
        contains the data of the validation task.
    K : int
        size of the finetuning set.
    cir : int
        class-imbalance rate (cir) of the target task (and therefore we sample the finetuning sets of the val tasks to have this same cir).
    val_normal_indexes : list
        indices of normal data samples of the validation task
    val_anomalous_indexes : list
        indices of anomalous data samples of the validation task

    Returns
    -------
    val_X_sampled : array
        features of the K datapoints sampled from the validation task 
        in the current multitask learning iteration.
    val_Y_sampled : array
        labels of the K datapoints sampled from the validation task 
        in the current multitask learning iteration.

    """

    n_needed_normal_val = int(K*cir)
    n_needed_anomalous_val = K - n_needed_normal_val
    val_normal_idxs = random.sample(val_normal_indexes, n_needed_normal_val)
    val_anomalous_idxs = random.sample(val_anomalous_indexes, n_needed_anomalous_val)
    val_idxs = val_normal_idxs
    val_idxs+=val_anomalous_idxs

    val_X_sampled, val_Y_sampled = val_task["finetune_X"][val_idxs], val_task["finetune_Y"][val_idxs]

    return val_X_sampled, val_Y_sampled


def main(args):

    seed = 123

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    cir_inner_loop_list = [float(i) for i in args.cir_inner_loop_list.split(' ')]
    K_list = [int(i) for i in args.K_list.split(' ')]

    train_data, val_task, test_task_finetune_sets, test_task_test_sets = load_MNIST(args.test_task_idx, args.val_task_idx)

    val_normal_indexes = list(np.nonzero(val_task['finetune_Y'] == 0)[0])
    val_anomalous_indexes = list(np.nonzero(val_task['finetune_Y'] == 1)[0])

    sess = tf.InteractiveSession()
    model = FB_OCSVM(sess, args, seed)

    summary = False
    if(args.summary_dir):
        summary = True


    if(summary):
        loddir_path = './summaries_FB'
        if (not (os.path.exists(loddir_path))):
            os.mkdir(loddir_path)
        if (not (os.path.exists(os.path.join(loddir_path, model.summary_dir)))):
            os.mkdir(os.path.join(loddir_path, model.summary_dir))
        train_writer = tf.summary.FileWriter(
            os.path.join(loddir_path, model.summary_dir) + '/train')

        val_task_writers = {}

        for K in K_list:
            for cir in cir_inner_loop_list:
                val_task_writers[str(K)+'_'+str(cir)] = tf.summary.FileWriter(
                    os.path.join(loddir_path, model.summary_dir) + '/val_task_K_' +str(K)+'_cir_'+str(cir))


    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())



    max_val_f1_score = {}
    max_val_f1_score_mtl_epoch = {}

    for K in K_list:
        for cir in cir_inner_loop_list:
            max_val_f1_score[str(K)+'_'+str(cir)] =-1
            max_val_f1_score_mtl_epoch[str(K)+'_'+str(cir)] = -1
            
    cir_val_OCSVM = 1.0
    print('cir_val_OCSVM set to ', cir_val_OCSVM)

    # FB-training
    for epoch in range(args.train_epochs+1):
        sampled_tr_idxs = random.sample(range(0, len(train_data['X'])), args.batch_size)
        X_train, Y_train = train_data['X'][sampled_tr_idxs], train_data['Y'][sampled_tr_idxs]
        
        train_loss, train_acc, tr_summaries = model.train_op(X_train, Y_train, epoch)
        if(summary and (epoch % model.summary_interval == 0)):
            print('Train epoch: ', epoch, ' train_acc: ', train_acc)
            train_writer.add_summary(tr_summaries, epoch)
            train_writer.flush()

        if(epoch % model.val_task_finetune_interval == 0):
            for K in K_list:
                for cir in cir_inner_loop_list:

                    if((K==10 and cir ==0.99) or (K<10 and cir not in [0.5, 1.0])):
                        pass
                    else:

                        X_val_finetune, Y_val_finetune = sample_random_val_finetune_data(val_task, K, cir_val_OCSVM, val_normal_indexes, val_anomalous_indexes)
                        
                        # if(cir > 0.5 and cir < 1.0):
                        #     ros = RandomOverSampler(random_state=seed)
                        #     X_val_finetune_reshaped = np.reshape(X_val_finetune, (X_val_finetune.shape[0], -1))
                        #     X_val_finetune_reshaped, Y_val_finetune_reshaped = ros.fit_resample(X_val_finetune_reshaped, np.squeeze(Y_val_finetune))
                        #     X_val_finetune = np.reshape(X_val_finetune_reshaped, (-1, 28, 28))
                        #     Y_val_finetune = np.expand_dims(Y_val_finetune_reshaped, -1)

                        val_acc, _, _, _, val_f1_score, val_auc_roc = model.val_op(X_val_finetune, Y_val_finetune, val_task['test_X'], val_task['test_Y'], K, cir, epoch)


                        if(val_f1_score > max_val_f1_score[str(K)+'_'+str(cir)]):
                            print('Epoch : ', epoch, ' model saved for K = ', K, ' val_f1_score = ', val_f1_score)
                            model.saver.save(
                                model.sess,
                                model.checkpoint_path +
                                model.summary_dir +
                                "_restore_val_task_test_f1_" + str(K) + '_' + str(cir_val_OCSVM) + "/model.ckpt")
                            max_val_f1_score[str(K)+'_'+str(cir)] = val_f1_score
                            max_val_f1_score_mtl_epoch[str(K)+'_'+str(cir)] = epoch

                        # if(summary):
                        #     val_task_writer = val_task_writers[str(K)+'_'+str(cir)]
                        #     val_task_writer.add_summary(val_task_summaries, epoch)
                        #     val_task_writer.flush()

    if(summary):
        train_writer.close()
        for K in K_list:
            for cir in cir_inner_loop_list:
                val_task_writers[str(K)+'_'+str(cir)].close()

    
    # test_task_finetune_writers, test_task_test_writers = {}, {}

    test_set = test_task_test_sets
    feed_dict_test_task_test = {model.X: test_set["test_X"], model.Y_oc: test_set["test_Y"]}
    
    kernel = 'rbf' 
    nu = 0.1
    GridSearch = True
    gamma = 'scale'

    for K in K_list:
        finetune_sets = test_task_finetune_sets[str(K)][str(cir_val_OCSVM)]
        acc_list, prec_list, rec_list, spec_list, f1_list, auc_roc_list = [], [], [], [], [], []
        
        model.saver.restore(
            model.sess,
            model.checkpoint_path +
            model.summary_dir +
            "_restore_val_task_test_f1_" + str(K) + '_' + str(cir_val_OCSVM) + "/model.ckpt")
        print('restored best model for K = ', K, ' based on val_f1')
        encoding_test = sess.run(model.extract_features_test, feed_dict=feed_dict_test_task_test)
        # print('encoded test set using this model - encoding_test.shape =', encoding_test.shape)

        for finetune_set_index, finetune_set in enumerate(finetune_sets):
            finetune_X, finetune_Y = finetune_set["finetune_X"], finetune_set["finetune_Y"]
            feed_dict_test_task_finetune = {model.X : finetune_X, model.Y_oc : finetune_Y}

            encoding_finetune = sess.run(model.extract_features_finetune, feed_dict=feed_dict_test_task_finetune)

            ocsvm = initialize_ocsvm(kernel, nu, gamma)
            ocsvm = train(ocsvm, encoding_finetune, encoding_test, np.squeeze(test_set["test_Y"]), kernel, nu, GridSearch)
            acc, prec, rec, spec, f1_score, auc_roc = predict(ocsvm, encoding_test, np.squeeze(test_set["test_Y"]), kernel)

            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)
            spec_list.append(spec)
            f1_list.append(f1_score)
            auc_roc_list.append(auc_roc)

        test_results_dict = {}
        test_results_dict['acc'] = acc_list
        test_results_dict['prec'] = prec_list
        test_results_dict['rec'] = rec_list
        test_results_dict['spec'] = spec_list
        test_results_dict['f1'] = f1_list
        test_results_dict['auc_roc'] = auc_roc_list
   

        results_dir_path = './results/'
        if (not (os.path.exists(results_dir_path))):
            os.mkdir(results_dir_path)
        filename = args.summary_dir + '_K_' + str(K) + '_cir_' + str(int(100*cir)) +'.txt'
        with open(results_dir_path+filename, 'wb') as file:
            pickle.dump(test_results_dict, file)



        print('K = ', K, ' average metrics')

        print(
            ' acc : ',
            np.mean(acc_list),
            ' prec : ',
            np.mean(prec_list),
            ' recall : ',
            np.mean(rec_list),
            ' specificity : ',
            np.mean(spec_list),
            ' f1_score : ',
            np.mean(f1_list),
            ' auc_pr : ',
            np.mean(auc_roc_list))

        print('min metrics')

        print(
            ' acc : ',
            np.amin(acc_list),
            ' prec : ',
            np.amin(prec_list),
            ' recall : ',
            np.amin(rec_list),
            ' specificity : ',
            np.amin(spec_list),
            ' f1_score : ',
            np.amin(f1_list),
            ' auc_pr : ',
            np.amin(auc_roc_list))

        print('max metrics')

        print(
            ' acc : ',
            np.amax(acc_list),
            ' prec : ',
            np.amax(prec_list),
            ' recall : ',
            np.amax(rec_list),
            ' specificity : ',
            np.amax(spec_list),
            ' f1_score : ',
            np.amax(f1_list),
            ' auc_pr : ',
            np.amax(auc_roc_list))

        n_test_tasks = len(acc_list)
        print('ci95 metrics - number of test tasks :', n_test_tasks)

        print(
            ' acc : ',
            1.96*np.std(acc_list)/np.sqrt(n_test_tasks),
            ' prec : ',
            1.96*np.std(prec_list)/np.sqrt(n_test_tasks),
            ' recall : ',
            1.96*np.std(rec_list)/np.sqrt(n_test_tasks),
            ' specificity : ',
            1.96*np.std(spec_list)/np.sqrt(n_test_tasks),
            ' f1_score : ',
            1.96*np.std(f1_list)/np.sqrt(n_test_tasks),
            ' auc_pr : ',
            1.96*np.std(auc_roc_list)/np.sqrt(n_test_tasks)
            )


    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-task training on multiple tasks then transfer learning (finetuning) on a test task/ purpose: benchmark with Model-Agnostic Meta-Learning (MAML)')
    parser.add_argument(
        '-filters',
        type=str,
        metavar='',
        help='number of filters for each convolutional layer e.g. "32 32 32 32"')
    parser.add_argument(
        '-kernel_sizes',
        type=str,
        metavar='',
        help='kernel sizes for each convolutional layer e.g. "3 3 3 3"')
    parser.add_argument(
        '-dense_layers',
        type=str,
        metavar='',
        help='size of each dense layer of the model e.g. "256 128 64 64"')
    parser.add_argument(
        '-lr',
        type=float,
        metavar='',
        help='learning rate (for pretraining and finetuning')
    parser.add_argument(
        '-train_epochs',
        type=int,
        metavar='',
        help='number of training epochs for the training tasks')
    parser.add_argument(
        '-finetune_epochs',
        type=int,
        metavar='',
        help='number of finetuning epochs (only for test task)')
    parser.add_argument(
        '-batch_size',
        type=int,
        metavar='',
        help='number of data points sampled for training')
    parser.add_argument(
        '-K_list',
        type=str,
        metavar='',
        help='number of finetuning examples in the test task')
    parser.add_argument(
        '-cir_inner_loop_list',
        type=str,
        metavar='',
        help='percentage of positive examples in the test task')
    parser.add_argument(
        '-test_task_idx',
        type=int,
        metavar='',
        help='index of the test task') 
    parser.add_argument(
        '-val_task_idx',
        type=int,
        metavar='',
        help='index of the val task') 
    parser.add_argument(
        '-summary_dir',
        type=str,
        metavar='',
        help=('name of the doirectory where the summaries should be saved. '
              'set to False, if summaries are not needed '))
    parser.add_argument('-config_file', 
        type=str, 
        default="None")


    args = parser.parse_args()

    args_dict = vars(args)
    if args.config_file is not "None":
        args_dict = extract_args_from_json(args.config_file, args_dict)

    for key in list(args_dict.keys()):

        if str(args_dict[key]).lower() == "true":
            args_dict[key] = True
        elif str(args_dict[key]).lower() == "false":
            args_dict[key] = False


    args = Bunch(args_dict)

    main(args)
