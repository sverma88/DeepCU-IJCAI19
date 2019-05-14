import os
import scipy.misc
import numpy as np
import tensorflow as tf


from DeepCU import Deep_ComUnq
from utils import pp, show_all_variables


flags = tf.app.flags
flags.DEFINE_integer("epoch", 300, "Epoch to train [25]")
flags.DEFINE_integer("hidden_a", 30, "Dimensions in audio")
flags.DEFINE_integer("hidden_v", 30, "Dimensions in video")
flags.DEFINE_integer("hidden_t", 30, "Dimensions in text")
flags.DEFINE_integer("Filters_AVT", 1, "Filters for AVT")
flags.DEFINE_integer("Filters_AV", 1, "Filters for AV")
flags.DEFINE_integer("Filters_AT", 1, "Filters for AT")
flags.DEFINE_integer("Filters_VT", 1, "Filters for VT")
flags.DEFINE_integer("LSTM_hid", 128, "Dimensions in text")
flags.DEFINE_integer("text_out", 64, "Dimensions in text_out")
flags.DEFINE_float("learning_rate", 0.006, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_float("momentum", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")
flags.DEFINE_string("data_dir", "/flush4/ver100/DeepCU-with-Attention-Networks-master/","directory of the data")
flags.DEFINE_string("Optimizer", "RMS","Adam, Grad, or Momentum")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        deep_cu = Deep_ComUnq(
            sess,
            data_dir = FLAGS.data_dir,
            batch_size = FLAGS.batch_size,
            hidden_a = FLAGS.hidden_a,
            hidden_t = FLAGS.hidden_t,
            LSTM_hid = FLAGS.LSTM_hid,
            hidden_v = FLAGS.hidden_v,
            text_out = FLAGS.text_out,
            Filters_AT = FLAGS.Filters_AT,
            Filters_VT = FLAGS.Filters_VT,
            Filters_AV = FLAGS.Filters_AV,
            Filters_AVT = FLAGS.Filters_AVT
        )

        show_all_variables()

        deep_cu.test(FLAGS)



if __name__ == '__main__':
    tf.app.run()
