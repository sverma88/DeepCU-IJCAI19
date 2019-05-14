import numpy as np
import scipy.io as sio
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

from ops import *
from utils import *
from functools import reduce



class Deep_ComUnq():

    def __init__(self, sess, data_dir, batch_size, hidden_v, hidden_a, hidden_t, LSTM_hid, text_out,
                 Filters_AVT, Filters_AT, Filters_VT, Filters_AV):
        """

        Args:
          sess: Tenlow session
          batch_size: The size of batch. Should be specified before training.
          data_dir: path to the director of the dataset
        """
        self.sess = sess
        self.y_dim = 1

        self.data_dir = data_dir
        self.batch_size = batch_size

        self.hv = hidden_v
        self.ha = hidden_a
        self.ht = hidden_t
        self.LSTM_hid = LSTM_hid
        self.t_out = text_out
        self.Filters_AVT = Filters_AVT
        self.Filters_AV = Filters_AV
        self.Filters_AT = Filters_AT
        self.Filters_VT = Filters_VT

        # batch normalization for Common Parts
        self.v_bn = batch_norm(name='video_subnet')
        self.v1_bn = batch_norm(name='video_subnet1')
        self.v2_bn = batch_norm(name='video_subnet2')
        self.a_bn = batch_norm(name='audio_subnet')
        self.a1_bn = batch_norm(name='audio_subnet1')
        self.a2_bn = batch_norm(name='audio_subnet2')
        self.t_bn = batch_norm(name='text_subnet')
        self.t1_bn = batch_norm(name='text_subnet1')
        self.t2_bn = batch_norm(name='text_subnet2')

        self.c_bn = batch_norm(name='conv_bn')
        self.c_bn01 = batch_norm(name='conv_bn01')
        self.c_bn02 = batch_norm(name='conv_bn02')
        self.c_bn1 = batch_norm(name='conv_bn1')
        self.c_bn11 = batch_norm(name='conv_bn11')
        self.c_bn2 = batch_norm(name='conv_bn2')
        self.c_bn21 = batch_norm(name='conv_bn21')
        self.c_bn3 = batch_norm(name='conv_bn3')
        self.c_bn31 = batch_norm(name='conv_bn31')

        # batch normalization for Unique Parts

        self.BN_V = batch_norm(name='FM_V')
        self.BN_A = batch_norm(name='FM_A')
        self.BN_T = batch_norm(name='FM_T')
        self.FM_A = batch_norm(name='FM_audio')
        self.FM_V = batch_norm(name='FM_visual')
        self.FM_T = batch_norm(name='FM_text')

        ## FM BN for LSTM
        self.LSTM_BN = batch_norm(name='LSTM')

        self.build_model()

    def build_model(self):

        audio_data, text_data, video_data, _ = self.load_train()

        self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')

        video_dim = video_data.shape[1]

        audio_dim = audio_data.shape[1]

        text_dim = text_data.shape

        self.video_inputs = tf.placeholder(tf.float32, [None, video_dim], name='video_data')

        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        self.audio_inputs = tf.placeholder(tf.float32, [None, audio_dim], name='audio_data')

        self.text_inputs = tf.placeholder(tf.float32, [None, text_dim[1], text_dim[2]], name='text_data')

        self.drop_ratio = tf.placeholder(tf.float32, [1], name='dratio')

        self.drop_LSTM = tf.placeholder(tf.float32, [1], name='drLSTM')


        #### Loss from 3D CNN
        self.D_logits_3D = self.TFConv_train(self.video_inputs, self.audio_inputs, self.text_inputs,
                                             self.hv, self.ha, self.ht, self.LSTM_hid, self.t_out,
                                             drop_LSTM=0.2 , dropl=self.drop_ratio, reuse=False)

        self.D_logits_3D_ = self.TFConv_test(self.video_inputs, self.audio_inputs, self.text_inputs,
                                             self.hv, self.ha,self.ht, self.LSTM_hid, self.t_out)

        #### Loss from AV CNN

        self.D_logits_AV = self.TFConv_train_AV(self.video_inputs, self.audio_inputs, self.hv, self.ha,
                                                dropl=self.drop_ratio, reuse=False)
        self.D_logits_AV_ = self.TFConv_test_AV(self.video_inputs, self.audio_inputs, self.hv, self.ha)

        #### Loss from VT CNN

        self.D_logits_VT = self.TFConv_train_VT(self.video_inputs, self.text_inputs, self.hv,self.ht,
                                                self.LSTM_hid, self.t_out, drop_LSTM=0.2,
                                                dropl=self.drop_ratio, reuse=False)

        self.D_logits_VT_ = self.TFConv_test_VT(self.video_inputs, self.text_inputs, self.hv, self.ht,
                                                self.LSTM_hid,  self.t_out)

        #### Loss for AT CNN

        self.D_logits_AT = self.TFConv_train_AT(self.audio_inputs, self.text_inputs, self.ha, self.ht,
                                                self.LSTM_hid, self.t_out,drop_LSTM=0.2,
                                                dropl=self.drop_ratio, reuse=False)

        self.D_logits_AT_ = self.TFConv_test_AT(self.audio_inputs, self.text_inputs, self.ha,
                                            self.ht, self.LSTM_hid, self.t_out)

        ##### Loss for Visual FM
        self.FMlogits_V = self.FMV_train(self.video_inputs, self.hv, dropl=self.drop_ratio, reuse=False)
        self.FMlogits_V_ = self.FMV_test(self.video_inputs, self.hv)

        ##### Loss for Audio FM
        self.FMlogits_A = self.FMA_train(self.audio_inputs, self.ha, dropl=self.drop_ratio, reuse=False)
        self.FMlogits_A_ = self.FMA_test(self.audio_inputs, self.ha)


        ##### Loss for text FM
        self.FMlogits_t = self.FMT_train(self.text_inputs, self.ht, self.LSTM_hid, self.t_out,
                                         drop_LSTM=0.2, dropl=self.drop_ratio, reuse=False)
        self.FMlogits_t_ = self.FMT_test(self.text_inputs, self.ht, self.LSTM_hid, self.t_out)


        #### Loss Function
        self.logits = self.D_logits_3D + self.D_logits_AT + self.D_logits_AV + self.D_logits_VT + \
                      self.FMlogits_A + self.FMlogits_t + self.FMlogits_V
        self.logits_ = self.D_logits_3D_ + self.D_logits_AT_ + self.D_logits_AV_ + self.D_logits_VT_+ \
                       self.FMlogits_A_ + self.FMlogits_t_ + self.FMlogits_V_

        #### Losss fucntion for training DeepCU
        self.cnn_loss = tf.reduce_sum(tf.losses.mean_squared_error(labels=self.y, predictions=self.logits))
        self.Accuracy = tf.reduce_sum(tf.abs(tf.subtract(self.y, self.logits_)))
        self.Pred = self.logits_
        self.diff = tf.abs(tf.subtract(self.y, self.logits_))

        self.saver = tf.train.Saver()


    #### Train the Network

    def train(self, config):

        if (config.Optimizer == "Adam"):
            cnn_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=config.beta1) \
                .minimize(self.cnn_loss)
        elif (config.Optimizer == "RMS"):
            cnn_optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cnn_loss)
        else:
            cnn_optim = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cnn_loss)

        checkpoint_path = '/flush4/ver100/TKDE/DeepCU-master/Trained_chk/'

        tf.global_variables_initializer().run()

        Au_trdat, Tx_trdat, Vi_trdat, Au_trlab = self.load_train()
        Au_tsdat, Tx_tsdat, Vi_tsdat, Au_tslab = self.load_test()
        Au_vdat, Tx_vdat, Vi_vdat, Au_vlab = self.load_val()

        train_batches = Au_trdat.shape[0] // self.batch_size
        test_batches = Au_tsdat.shape[0] // self.batch_size
        val_batches = Au_vdat.shape[0] // self.batch_size

        left_index_test = Au_tsdat.shape[0] - (test_batches * config.batch_size)
        left_index_train = Au_trdat.shape[0] - (train_batches * config.batch_size)
        left_index_val = Au_vdat.shape[0] - (val_batches * config.batch_size)

        dropout_list = np.arange(0.2, 0.95, 0.05)

        for drop1 in dropout_list:

            tf.global_variables_initializer().run()
            seed = 20

            print("dropout ratio --->", drop1)

            #### Start training the model
            lr = config.learning_rate

            for epoch in range(config.epoch):
                seed += 1

                if np.mod(epoch + 1, 40) == 0:
                    lr = lr - lr * 0.1

                random_index = np.random.RandomState(seed=seed).permutation(Au_trdat.shape[0])
                train_data_au = Au_trdat[random_index]
                train_data_vi = Vi_trdat[random_index]
                train_data_tx = Tx_trdat[random_index]
                train_lab_au = Au_trlab[random_index]

                for idx in range(train_batches):
                    batch_au = train_data_au[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_vi = train_data_vi[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_tx = train_data_tx[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = train_lab_au[idx * config.batch_size:(idx + 1) * config.batch_size]

                    _ = self.sess.run([cnn_optim],
                                      feed_dict={
                                          self.audio_inputs: batch_au,
                                          self.video_inputs: batch_vi,
                                          self.text_inputs: batch_tx,
                                          self.y: batch_labels,
                                          self.drop_ratio: [drop1],
                                          self.learning_rate: lr
                                      })

                ##### Printing Loss on each epoch to monitor convergence
                ##### Apply Early stoping procedure to report results

                print("epoch", epoch)

                Val_Loss = 0.0

                random_index = np.random.permutation(Au_vdat.shape[0])
                VAL_data_au = Au_vdat[random_index]
                VAL_data_vi = Vi_vdat[random_index]
                VAL_data_tx = Tx_vdat[random_index]
                VAL_lab_au = Au_vlab[random_index]

                for idx in range(val_batches):
                    batch_au = VAL_data_au[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_vi = VAL_data_vi[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_tx = VAL_data_tx[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = VAL_lab_au[idx * config.batch_size:(idx + 1) * config.batch_size]

                    Val_Loss += self.Accuracy.eval({
                        self.audio_inputs: batch_au,
                        self.video_inputs: batch_vi,
                        self.text_inputs: batch_tx,
                        self.y: batch_labels
                    })

                batch_au = train_data_au[-left_index_val:]
                batch_vi = train_data_vi[-left_index_val:]
                batch_tx = train_data_tx[-left_index_val:]
                batch_labels = train_lab_au[-left_index_val:]

                Val_Loss += self.Accuracy.eval({
                    self.audio_inputs: batch_au,
                    self.video_inputs: batch_vi,
                    self.text_inputs: batch_tx,
                    self.y: batch_labels
                })

                Val_MAE = Val_Loss / (Au_vdat.shape[0])

                ### Check the training loss
                Tr_Loss = 0.0

                random_index = np.random.permutation(Au_trdat.shape[0])
                train_data_au = Au_trdat[random_index]
                train_data_vi = Vi_trdat[random_index]
                train_data_tx = Tx_trdat[random_index]
                train_lab_au = Au_trlab[random_index]

                for idx in range(train_batches):
                    batch_au = train_data_au[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_vi = train_data_vi[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_tx = train_data_tx[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = train_lab_au[idx * config.batch_size:(idx + 1) * config.batch_size]

                    Tr_Loss += self.Accuracy.eval({
                        self.audio_inputs: batch_au,
                        self.video_inputs: batch_vi,
                        self.text_inputs: batch_tx,
                        self.y: batch_labels
                    })

                batch_au = train_data_au[-left_index_train:]
                batch_vi = train_data_vi[-left_index_train:]
                batch_tx = train_data_tx[-left_index_train:]
                batch_labels = train_lab_au[-left_index_train:]

                Tr_Loss += self.Accuracy.eval({
                    self.audio_inputs: batch_au,

                    self.video_inputs: batch_vi,
                    self.text_inputs: batch_tx,
                    self.y: batch_labels
                })

                Train_MAE = Tr_Loss / (Au_trdat.shape[0])

                Test_loss = 0.0

                for idx in range(test_batches):
                    batch_au = Au_tsdat[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_vi = Vi_tsdat[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_tx = Tx_tsdat[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = Au_tslab[idx * config.batch_size:(idx + 1) * config.batch_size]

                    Test_loss += self.Accuracy.eval({
                        self.audio_inputs: batch_au,
                        self.video_inputs: batch_vi,
                        self.text_inputs: batch_tx,
                        self.y: batch_labels

                    })

                ### Do it for the left exampels which does not account in batches
                batch_au = Au_tsdat[-left_index_test:]
                batch_vi = Vi_tsdat[-left_index_test:]
                batch_tx = Tx_tsdat[-left_index_test:]
                batch_labels = Au_tslab[-left_index_test:]

                Test_loss += self.Accuracy.eval({
                    self.audio_inputs: batch_au,
                    self.video_inputs: batch_vi,
                    self.text_inputs: batch_tx,
                    self.y: batch_labels
                })

                Test_MAE = Test_loss / Au_tsdat.shape[0]

                print(" ******* MOSI Results ************ ")
                print("Train MAE ---->", Train_MAE)
                print("VAl MAE ---->", Val_MAE)
                print("Test MAE ---->", Test_MAE)

        print('********** Iterations Terminated **********')

    #### Test the Network
    def test(self, config):

        tf.global_variables_initializer().run()

        Au_tsdat, Tx_tsdat, Vi_tsdat, Au_tslab = self.load_test()

        test_batches = Au_tsdat.shape[0] // self.batch_size

        left_index_test = Au_tsdat.shape[0] - (test_batches * config.batch_size)

        tf.global_variables_initializer().run()

        self.saver.restore(self.sess,'/flush4/ver100/TKDE/DeepCU-master/Trained_chk/0.65/DeepCU_0.65-150')

        Test_loss = 0.0
        for idx in range(test_batches):
            batch_au = Au_tsdat[idx * config.batch_size:(idx + 1) * config.batch_size]
            batch_vi = Vi_tsdat[idx * config.batch_size:(idx + 1) * config.batch_size]
            batch_tx = Tx_tsdat[idx * config.batch_size:(idx + 1) * config.batch_size]
            batch_labels = Au_tslab[idx * config.batch_size:(idx + 1) * config.batch_size]

            Test_loss += self.Accuracy.eval({
                self.audio_inputs: batch_au,
                self.video_inputs: batch_vi,
                self.text_inputs: batch_tx,
                self.y: batch_labels

            })

        ### Do it for the left exampels which does not account in batches
        batch_au = Au_tsdat[-left_index_test:]
        batch_vi = Vi_tsdat[-left_index_test:]
        batch_tx = Tx_tsdat[-left_index_test:]
        batch_labels = Au_tslab[-left_index_test:]

        Test_loss += self.Accuracy.eval({
            self.audio_inputs: batch_au,
            self.video_inputs: batch_vi,
            self.text_inputs: batch_tx,
            self.y: batch_labels
        })


        Test_MAE = Test_loss / Au_tsdat.shape[0]

        print(" ******* MOSI Results ************ ")
        print("Test MAE ---->", Test_MAE)


    #### Code for 3D convolution

    def TFConv_train(self, data_v, data_a, data_t, hidden_v, hidden_a, hidden_t, LSTM_hid, text_out,
                     drop_LSTM, dropl, reuse=False):
        with tf.variable_scope("TFConv_3D") as scope:
            if reuse:
                scope.reuse_variables()

            #### Video Subnet
            h0v1 = tf.nn.relu(self.v_bn(linear(data_v, hidden_v, 'h0_v1')))

            #### Audio Subnet
            h0a1 = tf.nn.relu(self.a_bn(linear(data_a, hidden_a, 'h0_a1')))

            #### Text Subent LSTM based
            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_hid)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0 - drop_LSTM)

            _, state = tf.nn.dynamic_rnn(LSTM_cell, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state.h, text_out, 'h0_t1'))

            h0t1 = tf.layers.dropout(h0t1, dropl)

            h0t2 = tf.nn.relu(self.t_bn(linear(h0t1, hidden_t, 'h0_t2')))

            ### now combien them in a tensor

            TF_tv = tf.einsum('ij,ik->ijk', h0t2, h0v1)

            TF_avt = tf.einsum('ijk,il->ijkl', TF_tv, h0a1)

            print(TF_avt)

            TF_avt = tf.expand_dims(TF_avt, [-1])

            conv0 = tf.nn.relu(self.c_bn(conv3d(TF_avt, TF_avt.shape[-1], 4, name='conv_0')))

            conv1 = tf.nn.relu(self.c_bn01(conv3d(conv0, conv0.shape[-1], 2, name='conv_1')))

            conv2 = tf.nn.relu(self.c_bn02(conv3d_3x3(conv1, conv1.shape[-1], 1, name='conv_2')))

            f_conv = tf.layers.flatten(conv2)

            print("conv3d output shape ---->", f_conv.shape)

            h1 = tf.layers.dropout(f_conv, dropl)

            h2 = tf.nn.relu(linear(h1, 5, 'h2_lin'))

            h3 = (linear(h2, 1, 'h3_lin'))

            return h3

    def TFConv_test(self, data_v, data_a, data_t, hidden_v, hidden_a, hidden_t, LSTM_hid, text_out):
        with tf.variable_scope("TFConv_3D") as scope:
            scope.reuse_variables()

            #### Video Subnet
            h0v1 = tf.nn.relu(self.v_bn(linear(data_v, hidden_v, 'h0_v1'),train=False))

            #### Audio Subnet
            h0a1 = tf.nn.relu(self.a_bn(linear(data_a, hidden_a, 'h0_a1'),train=False))

            #### Text Subent LSTM based
            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_hid)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0)

            # initial_state = LSTM_cell.zero_state(batch_size= self.batch_size, dtype=tf.float32)

            _, state = tf.nn.dynamic_rnn(LSTM_cell, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state.h, text_out, 'h0_t1'))

            h0t1 = tf.layers.dropout(h0t1, 1.0)

            h0t2 = tf.nn.relu(self.t_bn(linear(h0t1, hidden_t, 'h0_t2'),train=False))

            ### now combien them in a tensor

            TF_tv = tf.einsum('ij,ik->ijk', h0t2, h0v1)

            TF_avt = tf.einsum('ijk,il->ijkl', TF_tv, h0a1)

            TF_avt = tf.expand_dims(TF_avt, [-1])

            conv0 = tf.nn.relu(self.c_bn(conv3d(TF_avt, TF_avt.shape[-1], 4, name='conv_0'),train=False))

            conv1 = tf.nn.relu(self.c_bn01(conv3d(conv0, conv0.shape[-1], 2, name='conv_1'),train=False))

            conv2 = tf.nn.relu(self.c_bn02(conv3d_3x3(conv1, conv1.shape[-1], 1, name='conv_2'),
                                           train=False))

            f_conv = tf.layers.flatten(conv2)

            h1 = tf.layers.dropout(f_conv, 1.0)

            h2 = tf.nn.relu(linear(h1, 5, 'h2_lin'))

            h3 = (linear(h2, 1, 'h3_lin'))

            return h3


    #### Code for 2D-Conv for audio and video
    def TFConv_train_AV(self, data_v, data_a, hidden_a, hidden_v,  dropl, reuse=False):
        with tf.variable_scope("TConvAV") as scope:
            if reuse:
                scope.reuse_variables()

            #### Video Subnet
            h0v1 = tf.nn.relu(self.v1_bn(linear(data_v, hidden_v, 'h0_v1')))

            #### Audio Subnet
            h0a1 = tf.nn.relu(self.a1_bn(linear(data_a, hidden_a, 'h0_a1')))

            ### now combien them in a tensor

            TF_av = tf.einsum('ij,ik->ijk', h0a1, h0v1)

            TF_av = tf.expand_dims(TF_av, [-1])

            conv0 = tf.nn.relu(self.c_bn1(conv2d(TF_av, TF_av.shape[-1], 2, name='conv_0')))

            conv1 = tf.nn.relu(self.c_bn11(conv2d_3x3(conv0, conv0.shape[-1], 1, name='conv_1')))

            f_conv0 = tf.layers.flatten(conv1)

            print("ConvAV 0 size --->", conv0.shape)

            print("ConvAV 1 size --->", conv1.shape)

            print("conv output shape ---->", f_conv0.shape)

            h1 = tf.layers.dropout(f_conv0, dropl)

            h2 = tf.nn.relu(linear(h1, 10, 'h2_lin'))

            h3 = (linear(h2, 1, 'h3_lin'))

            return h3

    def TFConv_test_AV(self, data_v, data_a, hidden_a, hidden_v):
        with tf.variable_scope("TConvAV") as scope:
            scope.reuse_variables()

            #### Video Subnet
            h0v1 = tf.nn.relu(self.v1_bn(linear(data_v, hidden_v, 'h0_v1'),train=False))

            #### Audio Subnet
            h0a1 = tf.nn.relu(self.a1_bn(linear(data_a, hidden_a, 'h0_a1'), train=False))

            ### now combien them in a tensor

            TF_av = tf.einsum('ij,ik->ijk', h0a1, h0v1)

            TF_av = tf.expand_dims(TF_av, [-1])

            conv0 = tf.nn.relu(self.c_bn1(conv2d(TF_av, TF_av.shape[-1], 2, name='conv_0'),
                                          train=False))

            conv1 = tf.nn.relu(self.c_bn11(conv2d_3x3(conv0, conv0.shape[-1], 1, name='conv_1'),
                                           train=False))

            f_conv0 = tf.layers.flatten(conv1)

            h1 = tf.layers.dropout(f_conv0, 1.0)

            h2 = tf.nn.relu(linear(h1, 10, 'h2_lin'))

            h3 = (linear(h2, 1, 'h3_lin'))

            return h3


    #### Code for 2D-conv for video and text
    def TFConv_train_VT(self, data_v, data_t, hidden_v, hidden_t, LSTM_hid, text_out, drop_LSTM,
                         dropl, reuse=False):
        with tf.variable_scope("TConvVT") as scope:
            if reuse:
                scope.reuse_variables()

            #### Video Subnet
            h0v1 = tf.nn.relu(self.v2_bn(linear(data_v, hidden_v, 'h0_v1')))

            #### Text Subent LSTM based
            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_hid)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0 - drop_LSTM)

            _, state = tf.nn.dynamic_rnn(LSTM_cell, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state.h, text_out, 'h0_t1'))

            h0t1 = tf.layers.dropout(h0t1, dropl)

            h0t2 = tf.nn.relu(self.t1_bn(linear(h0t1, hidden_t, 'h0_t2')))

            ### now combien them in a tensor

            TF_vt = tf.einsum('ij,ik->ijk', h0v1, h0t2)

            TF_vt = tf.expand_dims(TF_vt, [-1])

            conv0 = tf.nn.relu(self.c_bn2(conv2d(TF_vt, TF_vt.shape[-1], 2, name='conv_0')))

            conv1 = tf.nn.relu(self.c_bn21(conv2d_3x3(conv0, conv0.shape[-1], 1, name='conv_1')))

            f_conv0 = tf.layers.flatten(conv1)

            print("ConvVT 0 size --->", conv0.shape)

            print("ConvVT 1 size --->", conv1.shape)

            print("conv output shape ---->", f_conv0.shape)

            h1 = tf.layers.dropout(f_conv0, dropl)

            h2 = tf.nn.relu(linear(h1, 10, 'h2_lin'))

            h3 = (linear(h2, 1, 'h3_lin'))

            return h3

    def TFConv_test_VT(self, data_v, data_t, hidden_v, hidden_t, LSTM_hid, text_out):
        with tf.variable_scope("TConvVT") as scope:
            scope.reuse_variables()

            #### Video Subnet
            h0v1 = tf.nn.relu(self.v2_bn(linear(data_v, hidden_v, 'h0_v1'),train=False))

            #### Text Subent LSTM based
            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_hid)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0)

            _, state = tf.nn.dynamic_rnn(LSTM_cell, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state.h, text_out, 'h0_t1'))

            h0t2 = tf.nn.relu(self.t1_bn(linear(h0t1, hidden_t, 'h0_t2'),train=False))

            ### now combien them in a tensor

            TF_vt = tf.einsum('ij,ik->ijk', h0v1, h0t2)

            TF_vt = tf.expand_dims(TF_vt, [-1])

            conv0 = tf.nn.relu(self.c_bn2(conv2d(TF_vt, TF_vt.shape[-1], 2, name='conv_0'),
                                          train=False))

            conv1 = tf.nn.relu(self.c_bn21(conv2d_3x3(conv0, conv0.shape[-1], 1, name='conv_1'),
                                           train=False))

            f_conv0 = tf.layers.flatten(conv1)

            h1 = tf.layers.dropout(f_conv0, 1.0)

            h2 = tf.nn.relu(linear(h1, 10, 'h2_lin'))

            h3 = (linear(h2, 1, 'h3_lin'))

            return h3


    #### Code for 2D-conv for audio and text
    def TFConv_train_AT(self, data_a, data_t, hidden_a, hidden_t, LSTM_hid, text_out,
                        drop_LSTM, dropl, reuse=False):
        with tf.variable_scope("TConvAT") as scope:
            if reuse:
                scope.reuse_variables()

            #### Audio Subnet
            h0a1 = tf.nn.relu(self.a2_bn(linear(data_a, hidden_a, 'h0_a1')))

            #### Text Subent LSTM based
            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_hid)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0 - drop_LSTM)

            _, state = tf.nn.dynamic_rnn(LSTM_cell, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state.h, text_out, 'h0_t1'))

            h0t1 = tf.layers.dropout(h0t1, dropl)

            h0t2 = tf.nn.relu(self.t2_bn(linear(h0t1, hidden_t, 'h0_t2')))

            ### now combien them in a tensor

            TF_at = tf.einsum('ij,ik->ijk', h0a1, h0t2)

            TF_at = tf.expand_dims(TF_at, [-1])

            conv0 = tf.nn.relu(self.c_bn3(conv2d(TF_at, TF_at.shape[-1], 2, name='conv_0')))

            conv1 = tf.nn.relu(self.c_bn31(conv2d_3x3(conv0, conv0.shape[-1], 1, name='conv_1')))

            f_conv0 = tf.layers.flatten(conv1)

            print("ConvAT 0 size --->", conv0.shape)

            print("ConvAT 1 size --->", conv1.shape)

            print("conv output shape ---->", f_conv0.shape)

            h1 = tf.layers.dropout(f_conv0, dropl)

            h2 = tf.nn.relu(linear(h1, 10, 'h2_lin'))

            h3 = (linear(h2, 1, 'h3_lin'))

            return h3

    def TFConv_test_AT(self, data_a, data_t, hidden_a, hidden_t, LSTM_hid, text_out):
        with tf.variable_scope("TConvAT") as scope:
            scope.reuse_variables()

            #### Audio Subnet
            h0a1 = tf.nn.relu(self.a2_bn(linear(data_a, hidden_a, 'h0_a1'),train=False))

            #### Text Subent LSTM based
            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_hid)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0)

            _, state = tf.nn.dynamic_rnn(LSTM_cell, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state.h, text_out, 'h0_t1'))

            h0t2 = tf.nn.relu(self.t2_bn(linear(h0t1, hidden_t, 'h0_t2'),train=False))

            ### now combien them in a tensor

            TF_at = tf.einsum('ij,ik->ijk', h0a1, h0t2)

            TF_at = tf.expand_dims(TF_at, [-1])

            conv0 = tf.nn.relu(self.c_bn3(conv2d(TF_at, TF_at.shape[-1], 2, name='conv_0'),
                                          train=False))

            conv1 = tf.nn.relu(self.c_bn31(conv2d_3x3(conv0, conv0.shape[-1], 1, name='conv_1'),
                                           train=False))

            f_conv0 = tf.layers.flatten(conv1)

            h1 = tf.layers.dropout(f_conv0, 1.0)

            h2 = tf.nn.relu(linear(h1, 10, 'h2_lin'))

            h3 = linear(h2, 1, 'h3_lin')

            return h3


    #### Code for FM over Audio
    def FMA_train(self, data_a, hidden_a, dropl, reuse=False):
        with tf.variable_scope("FMA") as scope:
            if reuse:
                scope.reuse_variables()

            h0a1 = tf.nn.relu(self.FM_A(linear(data_a, hidden_a, 'h0_a1')))

            h0a1 = tf.layers.dropout(h0a1, dropl)

            # #### Audio FM
            IntrVect_Audio = denseV(h0a1, hidden_a, 'Interaction_Vector_A')

            pair_interactions = self.BN_A(0.5*tf.subtract(
                                tf.pow(tf.matmul(h0a1, tf.transpose(IntrVect_Audio)), 2),
                                tf.matmul(tf.pow(h0a1, 2), tf.transpose(tf.pow(IntrVect_Audio, 2)))
                                ))

            Bilinear_A = tf.reduce_sum(pair_interactions, 1, keepdims=True)

            linear_terms = denseW(h0a1, 'linear_terms_a1')

            pred = linear_terms + Bilinear_A

            return pred

    def FMA_test(self, data_a, hidden_a):
        with tf.variable_scope("FMA") as scope:
            scope.reuse_variables()

            h0a1 = tf.nn.relu(self.FM_A(linear(data_a, hidden_a, 'h0_a1'), train=False))

            # #### Audio FM
            IntrVect_Audio = denseV(h0a1, hidden_a, 'Interaction_Vector_A')

            pair_interactions = self.BN_A(0.5 * tf.subtract(
                tf.pow(tf.matmul(h0a1, tf.transpose(IntrVect_Audio)), 2),
                tf.matmul(tf.pow(h0a1, 2), tf.transpose(tf.pow(IntrVect_Audio, 2)))
                ),train=False)


            Bilinear_A = tf.reduce_sum(pair_interactions, 1, keepdims=True)

            linear_terms = denseW(h0a1, 'linear_terms_a1')

            pred = linear_terms + Bilinear_A

            return pred


    #### Code for FM over Video
    def FMV_train(self, data_v, hidden_v, dropl, reuse=False):
        with tf.variable_scope("FMV") as scope:
            if reuse:
                scope.reuse_variables()

            h0v1 = tf.nn.relu(self.FM_V(linear(data_v, hidden_v, 'h0_v1')))

            h0v1 = tf.layers.dropout(h0v1, dropl)

            ##### Video FM
            IntrVect_Video = denseV(h0v1, hidden_v, 'Interaction_Vector_V')

            pair_interactions = self.BN_V(0.5 * tf.subtract(
                tf.pow(tf.matmul(h0v1, tf.transpose(IntrVect_Video)), 2),
                tf.matmul(tf.pow(h0v1, 2), tf.transpose(tf.pow(IntrVect_Video, 2)))
                ))

            Bilinear_V = tf.reduce_sum(pair_interactions, 1, keepdims=True)

            linear_terms_v1 = denseW(h0v1, 'linear_terms_v1')

            pred = linear_terms_v1 + Bilinear_V

            return pred

    def FMV_test(self, data_v, hidden_v):
        with tf.variable_scope("FMV") as scope:
            scope.reuse_variables()

            h0v1 = tf.nn.relu(self.FM_V(linear(data_v, hidden_v, 'h0_v1'), train=False))

            ##### Video FM
            IntrVect_Video = denseV(h0v1, hidden_v, 'Interaction_Vector_V')

            pair_interactions = self.BN_V(0.5 * tf.subtract(
                tf.pow(tf.matmul(h0v1, tf.transpose(IntrVect_Video)), 2),
                tf.matmul(tf.pow(h0v1, 2), tf.transpose(tf.pow(IntrVect_Video, 2)))
                ), train=False)

            Bilinear_V = tf.reduce_sum(pair_interactions, 1, keepdims=True)

            linear_terms_v1 = denseW(h0v1, 'linear_terms_v1')

            pred = linear_terms_v1 + Bilinear_V

            return pred


    #### Code for FM over Text
    def FMT_train(self, data_t, hidden_t, LSTM_hid, text_out, drop_LSTM, dropl, reuse=False):
        with tf.variable_scope("FMT") as scope:
            if reuse:
                scope.reuse_variables()

            #### Text Subent LSTM based
            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_hid)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0 - drop_LSTM)

            _, state = tf.nn.dynamic_rnn(LSTM_cell, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state.h, text_out, 'h0_t1'))

            h0t1 = tf.layers.dropout(h0t1, dropl)

            h0t2 = tf.nn.relu(self.LSTM_BN(linear(h0t1, hidden_t, 'h0_t2')))

            #### Text FM
            IntrVect_text = denseV(h0t2, hidden_t, 'Interaction_Vector_T')

            pair_interactions = self.BN_T(0.5*tf.subtract(
                                    tf.pow(tf.matmul(h0t2, tf.transpose(IntrVect_text)), 2),
                                    tf.matmul(tf.pow(h0t2, 2), tf.transpose(tf.pow(IntrVect_text, 2)))))

            Bilinear_T = tf.reduce_sum(pair_interactions, 1, keepdims=True)

            linear_terms_T = denseW(h0t2, 'linear_terms')

            pred = linear_terms_T + Bilinear_T

            return pred

    def FMT_test(self, data_t, hidden_t, LSTM_hid, text_out):
        with tf.variable_scope("FMT") as scope:
            scope.reuse_variables()

            #### Text Subent LSTM based
            LSTM_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_hid)
            LSTM_cell = tf.contrib.rnn.DropoutWrapper(LSTM_cell, output_keep_prob=1.0 - 0.0)

            _, state = tf.nn.dynamic_rnn(LSTM_cell, data_t, dtype=tf.float32)

            h0t1 = tf.nn.relu(linear(state.h, text_out, 'h0_t1'))

            h0t2 = tf.nn.relu(self.LSTM_BN(linear(h0t1, hidden_t, 'h0_t2'),train=False))

            #### Text FM
            IntrVect_text = denseV(h0t2, hidden_t, 'Interaction_Vector_T')

            pair_interactions = self.BN_T(0.5*tf.subtract(
                                    tf.pow(tf.matmul(h0t2, tf.transpose(IntrVect_text)), 2),
                                    tf.matmul(tf.pow(h0t2, 2), tf.transpose(tf.pow(IntrVect_text, 2))))
                                          ,train=False)

            Bilinear_T = tf.reduce_sum(pair_interactions, 1, keepdims=True)

            linear_terms_T = denseW(h0t2, 'linear_terms')

            pred = linear_terms_T + Bilinear_T

            return pred


    #### Data Loading part
    def load_train(self):

        train_dataset = sio.loadmat(self.data_dir + 'MOSI_train.mat')

        train_audio = train_dataset['audio']
        train_visual = train_dataset['video']
        train_text = train_dataset['text']
        train_label = train_dataset['labels']

        train_label = train_label + 3


        return train_audio, train_text, train_visual, train_label

    def load_test(self):

        test_dataset = sio.loadmat(self.data_dir + 'MOSI_test.mat')

        test_audio = test_dataset['audio']
        test_visual = test_dataset['video']
        test_text = test_dataset['text']
        test_label = test_dataset['labels']

        test_label = test_label + 3


        return test_audio, test_text, test_visual, test_label

    def load_val(self):

        val_dataset = sio.loadmat(self.data_dir + 'MOSI_val.mat')

        val_audio = val_dataset['audio']
        val_visual = val_dataset['video']
        val_text = val_dataset['text']
        val_label = val_dataset['labels']

        val_label = val_label + 3

        return val_audio, val_text, val_visual, val_label


