# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import os

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()



class MAML:
    """ This class defines the model trained with the MAML algorithm.

    """

    def __init__(self, sess, args, seed, n_train_tasks, input_shape):
        random.seed(seed)
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)

        self.lr = args.lr
        self.meta_lr = args.meta_lr
        self.meta_epochs = args.meta_epochs
        self.K = args.K
        self.num_updates = args.num_updates
        self.bn = args.bn

        self.anil = False
        if(hasattr(args, 'anil')):
            self.anil = args.anil

        self.dataset = None
        if(hasattr(args, 'dataset')):
            self.dataset = args.dataset

        self.sess = sess
        self.summary = False
        self.summary_dir = args.summary_dir

        if(self.summary_dir):
            self.summary = True
            self.summary_interval = 100
            summaries_list_metatrain = []
            summaries_list_val = []
            summaries_list_test_restore_val = []
        else:
            self.summary_dir = "no_summary"

        self.stop_grad = args.stop_grad

        self.n_queries = args.n_queries
        # dataset specific variables
        self.n_classes = 1
        self.dim_input = input_shape[0]
        self.input_shape = input_shape
        self.n_train_tasks = n_train_tasks

        # number of tasks to sample per meta-training iteration
        self.n_sample_tasks = 8
        self.flatten = tf.compat.v1.keras.layers.Flatten()

        # build model
        self.layers = []
        if(args.filters == ""):
            self.filter_sizes = []
        else:
            self.filter_sizes = [int(i) for i in args.filters.split(' ')]
            self.kernel_sizes = [int(i) for i in args.kernel_sizes.split(' ')]
            if(len(self.filter_sizes) == 1):
                self.layers.append(
                    tf.compat.v1.keras.layers.Conv1D(
                        filters=self.filter_sizes[0],
                        kernel_size=self.kernel_sizes[0],
                        input_shape=(None, self.dim_input, 1),
                        strides=2,
                        padding='same',
                        activation='relu',
                        name='conv_last'))

                if(self.bn):
                    self.layers.append(
                        tf.compat.v1.keras.layers.BatchNormalization(
                            name='bn_c_last'))

            else:
                self.layers.append(
                    tf.compat.v1.keras.layers.Conv1D(
                        filters=self.filter_sizes[0],
                        kernel_size=self.kernel_sizes[0],
                        input_shape=(None, self.dim_input, 1),
                        strides=2,
                        padding='same',
                        activation='relu',
                        name='conv0'))
                if(self.bn):
                    self.layers.append(
                        tf.compat.v1.keras.layers.BatchNormalization(
                            name='bn_c0'))

            for i in range(1, len(self.filter_sizes)):
                if(i != len(self.filter_sizes) - 1):
                    self.layers.append(
                        tf.compat.v1.keras.layers.Conv1D(
                            filters=self.filter_sizes[i],
                            kernel_size=self.kernel_sizes[i],
                            strides=2,
                            padding='same',
                            activation='relu',
                            name='conv' + str(i)))
                    if(self.bn):
                        self.layers.append(
                            tf.compat.v1.keras.layers.BatchNormalization(
                                name='bn_c' + str(i)))

                else:
                    self.layers.append(
                        tf.compat.v1.keras.layers.Conv1D(
                            filters=self.filter_sizes[i],
                            kernel_size=self.kernel_sizes[i],
                            strides=2,
                            padding='same',
                            activation='relu',
                            name='conv_last'))
                    if(self.bn):
                        self.layers.append(
                            tf.compat.v1.keras.layers.BatchNormalization(
                                name='bn_c_last'))

        if(args.dense_layers == ""):
            self.dense_sizes = []
        else:
            self.dense_sizes = [int(i) for i in args.dense_layers.split(' ')]

        for i in range(0, len(self.dense_sizes)):
            if(len(self.filter_sizes) == 0 and i == 0):
                self.layers.append(
                    tf.compat.v1.keras.layers.Dense(
                        units=self.dense_sizes[i],
                        activation='relu',
                        input_shape=(
                            None,
                        ) + input_shape,
                        name='dense' + str(i)))
            else:
                self.layers.append(
                    tf.compat.v1.keras.layers.Dense(
                        units=self.dense_sizes[i],
                        activation='relu',
                        name='dense' + str(i)))
            if(self.bn):
                self.layers.append(tf.compat.v1.keras.layers.BatchNormalization(
                    name='bn_d' + str(i + len(self.filter_sizes))))

        self.layers.append(
            tf.compat.v1.keras.layers.Dense(
                units=self.n_classes,
                name='dense_last'))

        # loss function
        self.loss_fct = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits

        self.X_train_a = tf.compat.v1.placeholder(
            tf.compat.v1.float32, (None, self.K, self.dim_input), name='X_train_a')
        self.Y_train_a = tf.compat.v1.placeholder(
            tf.compat.v1.float32, (None, self.K, self.n_classes), name='Y_train_a')
        self.X_train_b = tf.compat.v1.placeholder(
            tf.compat.v1.float32, (None, None, self.dim_input), name='X_train_b')
        self.Y_train_b = tf.compat.v1.placeholder(
            tf.compat.v1.float32, (None, None, self.n_classes), name='Y_train_b')
        self.X_finetune = tf.compat.v1.placeholder(
            tf.compat.v1.float32, (None, self.dim_input), name='X_finetune')
        self.Y_finetune = tf.compat.v1.placeholder(
            tf.compat.v1.float32, (None, self.n_classes), name='Y_finetune')

        self.construct_forward = tf.compat.v1.make_template(
            'construct_forward', self.feed_forward)

        self.finetune_output = self.construct_forward(
            self.X_finetune, training=True)

        self.finetune_loss = tf.compat.v1.reduce_mean(
            self.loss_fct(
                labels=self.Y_finetune,
                logits=self.finetune_output))

        self.m_vars = []

        for layer_idx in range(0, len(self.layers)):
            self.m_vars.append(self.layers[layer_idx].weights[0])
            self.m_vars.append(self.layers[layer_idx].weights[1])

        if(self.anil):
            if(self.bn):
                self.n_head_params = 2 * (len(self.dense_sizes) * 2 + 1)
            else:
                self.n_head_params = 2 * (len(self.dense_sizes) + 1)

            self.finetune_update_op = tf.compat.v1.train.GradientDescentOptimizer(self.lr).minimize(
                self.finetune_loss, var_list=self.m_vars[-self.n_head_params:])

        else:
            self.finetune_update_op = tf.compat.v1.train.GradientDescentOptimizer(
                self.lr).minimize(self.finetune_loss)

        self.test_output = self.construct_forward(
            self.X_finetune, training=False)
        self.test_loss = tf.compat.v1.reduce_mean(self.loss_fct(
            labels=self.Y_finetune,
            logits=self.test_output))

        self.my_acc, self.my_precision, self.my_recall, self.my_specificity, self.my_f1_score, self.my_auc_pr = self.compute_metrics(
            self.test_output, self.Y_finetune)

        if(self.bn):
            self.updated_bn_model = self.assign_stats(self.X_finetune)

        self.total_loss = self.compute_losses(training=True)

        meta_optimizer = tf.compat.v1.train.AdamOptimizer(self.meta_lr)

        self.meta_opt_compute_gradients = meta_optimizer.compute_gradients(
            self.total_loss)

        self.meta_update_op = meta_optimizer.apply_gradients(
            self.meta_opt_compute_gradients)

        if(self.summary):
            summaries_list_metatrain.append(
                tf.compat.v1.summary.scalar('total_train_loss', self.total_loss))
            self.merged_metatrain = tf.compat.v1.summary.merge(
                summaries_list_metatrain)
            summaries_list_val.append(
                tf.compat.v1.summary.scalar('val_test_loss', self.test_loss))
            summaries_list_val.append(
                tf.compat.v1.summary.scalar('val_accuracy', self.my_acc))
            summaries_list_val.append(
                tf.compat.v1.summary.scalar('val_precision', self.my_precision))
            summaries_list_val.append(
                tf.compat.v1.summary.scalar('val_recall', self.my_recall))
            summaries_list_val.append(
                tf.compat.v1.summary.scalar('val_specificity', self.my_specificity))
            summaries_list_val.append(
                tf.compat.v1.summary.scalar('val_f1_score', self.my_f1_score))
            summaries_list_val.append(
                tf.compat.v1.summary.scalar('val_auc_pr', self.my_auc_pr))
            summaries_list_test_restore_val.append(
                tf.compat.v1.summary.scalar('test_loss_1', self.test_loss))
            summaries_list_test_restore_val.append(
                tf.compat.v1.summary.scalar('accuracy_1', self.my_acc))
            summaries_list_test_restore_val.append(
                tf.compat.v1.summary.scalar('precision_1', self.my_precision))
            summaries_list_test_restore_val.append(
                tf.compat.v1.summary.scalar('recall_1', self.my_recall))
            summaries_list_test_restore_val.append(
                tf.compat.v1.summary.scalar('specificity_1', self.my_specificity))
            summaries_list_test_restore_val.append(
                tf.compat.v1.summary.scalar('f1_score_1', self.my_f1_score))
            summaries_list_test_restore_val.append(
                tf.compat.v1.summary.scalar('auc_pr_1', self.my_auc_pr))
            self.merged_test_restore_val = tf.compat.v1.summary.merge(
                summaries_list_test_restore_val)
            self.merged_val = tf.compat.v1.summary.merge(
                summaries_list_val)

        self.saver = tf.compat.v1.train.Saver()

        base_path = '/content/Few-Shot-One-Class-Classification-via-Meta-Learning'
        if (not (os.path.exists(base_path))):
            base_path = '/home/ubuntu/Projects'
        if (not (os.path.exists(base_path))):
            base_path = '/home/USER/Projects'
        if (not (os.path.exists(base_path))):
            base_path = '/home/ceesgniewyk/Projects'

        self.checkpoint_path = base_path + '/MAML/checkpoints_MAML'
        if (not (os.path.exists(self.checkpoint_path))):
            os.mkdir(self.checkpoint_path)
        if (not (os.path.exists(os.path.join(self.checkpoint_path, self.summary_dir)))):
            os.mkdir(os.path.join(self.checkpoint_path, self.summary_dir))

    def compute_metrics(self, logits, labels, logits_are_predictions=False):
        """compute non-running performance metrics.

        Parameters
        ----------
        logits : tensor
        labels : tensor


        Returns
        -------
        acc : tensor
            accuracy.
        precision : tensor
            precision.
        recall : tensor
            recall.
        specificity : tensor
            specificity.
        f1_score : tensor
            F1 score.
        auc_pr : tensor
            AUC-PR.

        """
        if(logits_are_predictions):
            predictions = logits
        else:
            predictions = tf.compat.v1.cast(
                tf.compat.v1.greater(
                    tf.compat.v1.nn.sigmoid(logits),
                    0.5),
                tf.compat.v1.float32)
        TP = tf.compat.v1.count_nonzero(predictions * labels, dtype=tf.compat.v1.float32)
        TN = tf.compat.v1.count_nonzero((predictions - 1) *
                              (labels - 1), dtype=tf.compat.v1.float32)
        FP = tf.compat.v1.count_nonzero(predictions * (labels - 1), dtype=tf.compat.v1.float32)
        FN = tf.compat.v1.count_nonzero((predictions - 1) * labels, dtype=tf.compat.v1.float32)
        acc = tf.compat.v1.reduce_mean(tf.compat.v1.to_float(tf.compat.v1.equal(predictions, labels)))

        precision = tf.compat.v1.cond(tf.compat.v1.math.equal((TP + FP), 0),
                            true_fn=lambda: 0.0, false_fn=lambda: TP / (TP + FP))
        recall = TP / (TP + FN)
        specificity = TN / (TN + FP)
        f1_score = tf.compat.v1.cond(
            tf.compat.v1.math.equal(
                (precision + recall),
                0),
            true_fn=lambda: 0.0,
            false_fn=lambda: 2 * precision * recall / (
                precision + recall))

        auc_pr = tf.compat.v1.metrics.auc(labels=labels, predictions=tf.compat.v1.nn.sigmoid(
            logits), curve='PR', summation_method='careful_interpolation')[1]

        return [acc, precision, recall, specificity, f1_score, auc_pr]

    def feed_forward(self, inp, training, no_head=False):
        """computes an output tensor by feeding the input through the network.

        Parameters
        ----------
        inp : tensor
            input tensor.
        training : bool
            argument for Batch normalization layers.

        Returns
        -------
        out : tensor
            output tensor.

        """

        h = tf.compat.v1.expand_dims(inp, -1)

        n_layers_no_head = len(self.layers) - len(self.dense_sizes) - 1
        if(self.bn):
            n_layers_no_head = len(self.layers) - len(self.dense_sizes) * 2 - 1

        for i in range(n_layers_no_head):
            if('conv' in self.layers[i].name):
                h = self.layers[i](h)
                # if(self.dataset == 'MIN'):
                h = tf.compat.v1.layers.max_pooling1d(
                    h, pool_size=2, strides=2, padding='same')
            elif('bn' in self.layers[i].name):
                h = self.layers[i](h, training=training)

            if(self.bn and 'bn_c_last' in self.layers[i].name):
                h = self.flatten(h)

            elif(not(self.bn) and 'conv_last' in self.layers[i].name):
                h = self.flatten(h)

        if(no_head):
            return h
        else:
            if(n_layers_no_head < 1):
                i = -1
            for j in range(i + 1, len(self.layers)):
                h = self.layers[j](h)
            return h

    def get_first_updated_weights(self):
        """computes the model parameters after the first adaptation/inner update.

        Returns
        -------
        new_weights : dict
            contains the parameters after applying the first adaptation
            update (the first theta prime).

        """

        if(self.anil):
            grads = tf.compat.v1.gradients(self.train_a_loss,
                                 self.m_vars[-self.n_head_params:])
        else:
            grads = tf.compat.v1.gradients(self.train_a_loss, self.m_vars)

        if(self.stop_grad):
            grads = [tf.compat.v1.stop_gradient(grad) for grad in grads]

        if(self.anil):
            new_weights = []
            for i in range(1, self.n_head_params + 1):
                new_weights.append(
                    self.m_vars[-self.n_head_params - 1 + i] - self.lr * grads[-self.n_head_params - 1 + i])
        else:
            if(self.bn):
                w_keys = []
                b_keys = []
                bn_gamma_keys = []
                bn_beta_keys = []

                for i in range(0, len(self.layers)):
                    w_keys.append('w' + str(i + 1))
                    b_keys.append('b' + str(i + 1))
                    bn_gamma_keys.append('bn_gamma' + str(i + 1))
                    bn_beta_keys.append('bn_beta' + str(i + 1))

                new_weights = dict(zip(w_keys, [
                                   self.m_vars[i] - self.lr * grads[i] for i in range(0, len(self.m_vars), 4)]))
                new_biases = dict(zip(b_keys, [
                                  self.m_vars[i] - self.lr * grads[i] for i in range(1, len(self.m_vars), 4)]))
                new_bn_gamma = dict(zip(bn_gamma_keys, [
                                    self.m_vars[i] - self.lr * grads[i] for i in range(2, len(self.m_vars), 4)]))
                new_bn_beta = dict(zip(bn_beta_keys, [
                                   self.m_vars[i] - self.lr * grads[i] for i in range(3, len(self.m_vars), 4)]))

                new_weights.update(new_biases)
                new_weights.update(new_bn_gamma)
                new_weights.update(new_bn_beta)

            else:
                w_keys = []
                b_keys = []

                for i in range(0, len(self.layers)):
                    w_keys.append('w' + str(i + 1))
                    b_keys.append('b' + str(i + 1))

                new_weights = dict(zip(w_keys, [
                                   self.m_vars[i] - self.lr * grads[i] for i in range(0, len(self.m_vars), 2)]))
                new_biases = dict(zip(b_keys, [
                                  self.m_vars[i] - self.lr * grads[i] for i in range(1, len(self.m_vars), 2)]))

                new_weights.update(new_biases)

        return new_weights

    def get_further_updated_weights(self, old_weights):
        """computes the model parameters after one inner update.

        Parameters
        ----------
        old_weights : dict
            contains the parameters before applying the current inner update

        Returns
        -------
        new_weights : dict
            contains the parameters after applying the current inner update

        """
        if(self.anil):
            grads = tf.compat.v1.gradients(self.train_a_loss, old_weights)
        else:
            old_weights_list = list(old_weights.values())
            grads = tf.compat.v1.gradients(self.train_a_loss, old_weights_list)

        if(self.stop_grad):
            grads = [tf.compat.v1.stop_gradient(grad) for grad in grads]

        if(self.anil):
            new_weights = [old_weights[i] - self.lr * grads[i]
                           for i in range(len(old_weights))]
        else:
            gradients = dict(zip(old_weights.keys(), grads))
            new_weights = dict(zip(old_weights.keys(), [
                old_weights[key] - self.lr * gradients[key] for key in old_weights.keys()]))

        return new_weights

    def new_weights_construct_forward(
            self, inp, weights, training, X_train_a=None):
        """computes an output tensor by feeding the input tensor through
        the model parametrized with given weights.

        Parameters
        ----------
        inp : tensor
            input tensor.
        weights : dict
            contains the parameters after applying inner updates
            (one of the theta primes).
        training : bool
            argument for Batch normalization layers.

        Returns
        -------
        out : tensor
            output tensor.

        """
        epsilon = 0.001

        h = tf.compat.v1.expand_dims(inp, -1)

        h = tf.compat.v1.cast(h, tf.compat.v1.float32)

        if(self.anil):

            features = self.construct_forward(h, training, no_head=True)
            f = features
            for i in range(1, self.n_head_params + 1, 2):
                f = tf.compat.v1.matmul(
                    f, weights[-self.n_head_params - 1 + i]) + weights[-self.n_head_params + i]
            return f

        else:
            if(X_train_a is not None and self.bn):
                mean_var_conv, mean_var_dense = [], []
                if(len(self.input_shape) < 3 and len(self.filter_sizes) > 0):
                    h_a = tf.compat.v1.expand_dims(X_train_a, -1)
                else:
                    h_a = X_train_a

                h_a = tf.compat.v1.cast(h_a, tf.compat.v1.float32)

                for i in range(0, len(self.filter_sizes)):
                    h_a = tf.compat.v1.nn.conv1d(h_a,
                                       filters=weights['w' + str(i + 1)],
                                       stride=2,
                                       padding="SAME") + weights['b' + str(i + 1)]
                    h_a = tf.compat.v1.keras.activations.relu(h_a)
                    h_a = tf.compat.v1.layers.max_pooling1d(
                        h_a, pool_size=2, strides=2, padding='same')
                    mean, var = tf.compat.v1.nn.moments(h_a, [0, 1])
                    mean_var_conv.append((mean, var))
                    h_a = ((h_a - mean) / tf.compat.v1.sqrt(var + epsilon)) * \
                        weights['bn_gamma' + str(i + 1)] + weights['bn_beta' + str(i + 1)]

                h_a = tf.compat.v1.layers.flatten(h_a)
                if(len(self.filter_sizes) == 0):
                    i = 0

                if(len(self.dense_sizes) > 0):
                    for j in range(i, len(self.filter_sizes) +
                                   len(self.dense_sizes)):
                        h_a = tf.compat.v1.matmul(
                            h_a, weights['w' + str(j + 1)]) + weights['b' + str(j + 1)]
                        h_a = tf.compat.v1.keras.activations.relu(h_a)
                        mean, var = tf.compat.v1.nn.moments(h_a, 0)
                        mean_var_dense.append((mean, var))
                        h_a = ((h_a - mean) / tf.compat.v1.sqrt(var + epsilon)) * \
                            weights['bn_gamma' + str(j + 1)] + weights['bn_beta' + str(j + 1)]

            for i in range(0, len(self.filter_sizes)):
                h = tf.compat.v1.nn.conv1d(h,
                                 filters=weights['w' + str(i + 1)],
                                 stride=2,
                                 padding="SAME") + weights['b' + str(i + 1)]
                h = tf.compat.v1.keras.activations.relu(h)
                h = tf.compat.v1.layers.max_pooling1d(
                    h, pool_size=2, strides=2, padding='same')
                if(self.bn):
                    if(X_train_a is not None):
                        mean, var = mean_var_conv[i]
                    else:
                        mean, var = tf.compat.v1.nn.moments(h, [0, 1])
                    h = ((h - mean) / tf.compat.v1.sqrt(var + epsilon)) * \
                        weights['bn_gamma' + str(i + 1)] + weights['bn_beta' + str(i + 1)]

            h = tf.compat.v1.layers.flatten(h)
            if(len(self.filter_sizes) == 0):
                i = 0

            if(len(self.dense_sizes) > 0):
                for j in range(i, len(self.filter_sizes) +
                               len(self.dense_sizes)):
                    h = tf.compat.v1.matmul(
                        h, weights['w' + str(j + 1)]) + weights['b' + str(j + 1)]
                    h = tf.compat.v1.keras.activations.relu(h)
                    if(self.bn):
                        if(X_train_a is not None):
                            mean, var = mean_var_dense[j]
                        else:
                            mean, var = tf.compat.v1.nn.moments(h, 0)
                        h = ((h - mean) / tf.compat.v1.sqrt(var + epsilon)) * \
                            weights['bn_gamma' + str(j + 1)] + weights['bn_beta' + str(j + 1)]

                out = tf.compat.v1.matmul(
                    h, weights['w' + str(j + 2)]) + weights['b' + str(j + 2)]
            else:
                out = tf.compat.v1.matmul(
                    h, weights['w' + str(i + 2)]) + weights['b' + str(i + 2)]
            return out

    def compute_losses(self, training):
        """computes the total loss over all tasks (loss for the meta-update).

        Parameters
        ----------
        training : bool
            argument for Batch normalization layers.

        Returns
        -------
        total_loss : tensor
            sum of the losses computed on the sampled outer loop batch of each sampled meta-training task.

        """

        loop_vars = (tf.compat.v1.constant(0), tf.compat.v1.constant(0.0, dtype=tf.compat.v1.float32))

        def cond(index, total_loss):
            return tf.compat.v1.less(index, self.n_sample_tasks)

        def body(index, total_loss):
            task_loss = self.metatrain_task(
                self.X_train_a[index],
                self.Y_train_a[index],
                self.X_train_b[index],
                self.Y_train_b[index],
                training)
            return tf.compat.v1.add(index, 1), tf.compat.v1.add(total_loss, task_loss)

        i, total_loss = tf.compat.v1.while_loop(cond, body, loop_vars, swap_memory=True)

        return total_loss

    def metatrain_op(self, epoch, X_train_a, Y_train_a, X_train_b, Y_train_b):
        """performs one meta-training iteration.

        Parameters
        ----------
        X_train_a : tensor
            contains features of the K datapoints sampled for the inner loop (adaptation) updates of each meta-training task.
        Y_train_a : tensor
            contains labels of the K datapoints sampled for the inner loop (adaptation) updates of each meta-training task.
        X_train_b : tensor
            contains features sampled for the outer loop updates of each meta-training task.
        Y_train_b : tensor
            contains labels sampled for the outer loop updates of each meta-training task.

        Returns
        -------
        metatrain_loss : float
            sum of the losses computed on the sampled outer loop batch of each sampled meta-training task.
        train_summaries : list
            training summaries.

        """

        feed_dict_train = {
            self.X_train_a: X_train_a,
            self.Y_train_a: Y_train_a,
            self.X_train_b: X_train_b,
            self.Y_train_b: Y_train_b,
        }

        metatrain_loss, _ = self.sess.run(
            [self.total_loss, self.meta_update_op], feed_dict_train)

        if(self.summary and (epoch % self.summary_interval == 0)):
            train_summaries = self.sess.run(
                self.merged_metatrain, feed_dict_train)
        else:
            train_summaries = None

        return metatrain_loss, train_summaries

    def metatrain_task(self, X_train_a, Y_train_a,
                       X_train_b, Y_train_b, training):
        """performs a meta-trainig iteration on a single meta-training task.

        Parameters
        ----------
        X_train_a : tensor
            contains features of the K datapoints sampled for the inner loop (adaptation) updates.
        Y_train_a : tensor
            contains labels of the K datapoints sampled for the inner loop (adaptation) updates.
        X_train_b : tensor
            contains features sampled for the outer loop updates.
        Y_train_b : tensor
            contains labels sampled for the outer loop updates.
        training : bool
            argument for Batch normalization layers.

        Returns
        -------
        train_b_loss : tensor
            loss computed on the outer loop data after adaptation (loss of the meta-update for one task).

        """

        train_a_output = self.construct_forward(X_train_a, training=training)
        self.train_a_loss = tf.compat.v1.reduce_mean(self.loss_fct(
            labels=Y_train_a,
            logits=train_a_output))
        new_weights = self.get_first_updated_weights()

        loop_vars = (tf.compat.v1.constant(0), new_weights)

        def cond(index, weights):
            return tf.compat.v1.less(index, self.num_updates - 1)

        def body(index, weights):
            train_a_output = self.new_weights_construct_forward(
                X_train_a, weights, training=training)
            self.train_a_loss = tf.compat.v1.reduce_mean(self.loss_fct(
                labels=Y_train_a,
                logits=train_a_output))
            new_weights_loop = self.get_further_updated_weights(weights)
            return tf.compat.v1.add(index, 1), new_weights_loop

        i, new_weights = tf.compat.v1.while_loop(cond, body, loop_vars, swap_memory=True)

        if(self.bn):
            self.train_b_output = self.new_weights_construct_forward(
                X_train_b, new_weights, training=training, X_train_a=X_train_a)
        else:
            self.train_b_output = self.new_weights_construct_forward(
                X_train_b, new_weights, training=training)
        self.train_b_loss = tf.compat.v1.reduce_mean(self.loss_fct(
            labels=Y_train_b,
            logits=self.train_b_output))

        return self.train_b_loss

    def val_op(self, K_X_val, K_Y_val, val_test_X, val_test_Y):
        """performs one validation episode.

        Parameters
        ----------
        K_X_val : array
            contains features of the K datapoints sampled for adaptation to the validation task(s).
        K_Y_val : array
            contains labels of the K datapoints sampled for adaptation to the validation task(s).
        val_test_X : array
            contains features of the test set(s) of the validation task(s).
        val_test_Y : array
            contains labels of the test set(s) of the validation task(s).

        Returns
        -------
        val_summaries : list
            validation summaries.
        val_test_loss : float
            loss computed on the test set after adaptation.
        acc : float
            accuracy computed on the test set after adaptation.
        precision : float
            precision computed on the test set after adaptation.
        recall : float
            recall computed on the test set after adaptation.
        specificity : float
            specificity computed on the test set after adaptation.
        f1_score : float
            F1 score computed on the test set after adaptation.
        auc_pr : float
            AUC-PR computed on the test set after adaptation.

        """

        # save current network parameters (including bn stats)
        old_vars = []
        for layer_idx in range(0, len(self.layers)):
            layer_weights = self.layers[layer_idx].get_weights()
            old_vars.append(layer_weights)

        val_test_feed_dict = {
            self.X_finetune: val_test_X, self.Y_finetune: val_test_Y
        }

        for i in range(self.num_updates):
            self.sess.run(self.finetune_update_op, {
                self.X_finetune: K_X_val, self.Y_finetune: K_Y_val})

        if(self.bn):
            # assign batch normalization stats using the available adaptation
            # set
            self.sess.run(self.updated_bn_model, {self.X_finetune: K_X_val})

        if(self.summary):
            self.sess.run(tf.compat.v1.local_variables_initializer())
            val_summaries, val_test_loss, acc, precision, recall, specificity, f1_score, auc_pr = self.sess.run(
                [self.merged_val, self.test_loss, self.my_acc, self.my_precision, self.my_recall, self.my_specificity, self.my_f1_score, self.my_auc_pr], feed_dict=val_test_feed_dict)
        else:
            self.sess.run(tf.compat.v1.local_variables_initializer())
            val_test_loss, acc, precision, recall, specificity, f1_score, auc_pr = self.sess.run(
                [self.test_loss, self.my_acc, self.my_precision, self.my_recall, self.my_specificity, self.my_f1_score, self.my_auc_pr], feed_dict=val_test_feed_dict)
            val_summaries = None

        # resetting old networks parameters (including bn stats)
        for layer_idx in range(0, len(self.layers)):
            self.layers[layer_idx].set_weights(old_vars[layer_idx])

        return val_summaries, val_test_loss, acc, precision, recall, specificity, f1_score, auc_pr

    def finetune_op(self, K_X_finetune, K_Y_finetune):
        """performs one adaptation/finetuning update.

        Parameters
        ----------
        K_X_finetune : tensor
            contains features of the K datapoints sampled for adaptation.
        K_Y_finetune : tensor
            contains labels of the K datapoints sampled for adaptation

        Returns
        -------
        finetune_loss : float
            adaptation/finetuning loss.

        """

        feed_dict_finetune = {
            self.X_finetune: K_X_finetune, self.Y_finetune: K_Y_finetune}
        finetune_loss, _ = self.sess.run(
            [self.finetune_loss, self.finetune_update_op], feed_dict_finetune)
        return finetune_loss

    def assign_stats(self, K_finetune_samples):
        """ compute BN stats (mean and variance) using the given adaptation set and assign them to the BN layers in the network.

        Parameters
        ----------
        K_finetune_samples : tensor
            conatins the features of the K datapoints sampled for adaptation.

        Returns
        -------
        out : tensor
            output of the last batch normalization layer (when it is computed using session.run, the BN stats are assigned)

        """

        h = tf.compat.v1.expand_dims(K_finetune_samples, -1)
        for i in range(len(self.layers)):
            if('dense_last' in self.layers[i].name):
                out = h
                return out
            elif('bn_c' in self.layers[i].name):
                mean, var = tf.compat.v1.nn.moments(h, [0, 1])
                assign_op_1 = self.layers[i].variables[-1].assign(var)
                assign_op_2 = self.layers[i].variables[-2].assign(mean)
                with tf.compat.v1.control_dependencies([assign_op_1, assign_op_2]):
                    h = self.layers[i](h, training=False)
            elif('bn_d' in self.layers[i].name):
                mean, var = tf.compat.v1.nn.moments(h, 0)
                assign_op_3 = self.layers[i].variables[-1].assign(var)
                assign_op_4 = self.layers[i].variables[-2].assign(mean)
                with tf.compat.v1.control_dependencies([assign_op_3, assign_op_4]):
                    h = self.layers[i](h, training=False)
            elif('conv' in self.layers[i].name):
                h = self.layers[i](h)
                h = tf.compat.v1.layers.max_pooling1d(
                    h, pool_size=2, strides=2, padding='same')
            if('bn_c_last' in self.layers[i].name):
                h = self.flatten(h)