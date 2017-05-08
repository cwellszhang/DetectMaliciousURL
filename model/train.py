# -*- coding: utf-8 -*-
import data_helper ,word2vec_helpers
import tensorflow as tf
import os ,time, datetime
import numpy as np
from sklearn.cross_validation import train_test_split
from URLCNN import *
import tempfile
# Parameters
# =======================================================

# Data loading parameters
tf.flags.DEFINE_string("data_file" ,"../data/data.csv", "Data source")
tf.flags.DEFINE_integer("num_labels", 2, "Number of labels for data. (default: 2)")
#
# # Model hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-spearated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("sequence_length", 100, "sequnce length of url embedding (default: 100)")
#
# # Training paramters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evalue model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (defult: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
#
# # Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# # Distribution
tf.flags.DEFINE_boolean("replicas",False,"Use the dirstribution mode")
tf.flags.DEFINE_boolean("is_sync",False,"Use the async or sync mode")
tf.flags.DEFINE_string("ps_hosts","192.168.0.107:2222","comma-separated lst of hostname:port pairs")
tf.flags.DEFINE_string("worker_hosts","10.211.55.14:2222,10.211.55.13:2222","comma-separated lst of hostname:port pairs")
tf.flags.DEFINE_string("job_name",None,"job name:worker or ps")
tf.flags.DEFINE_integer("task_index",0,"Worker task index,should be >=0, task=0 is "
                                       "the master worker task the performs the variable initialization")
tf.flags.DEFINE_string("log_dir","./runs/outputs/summary","parameter and log info")
# Parse parameters from commands
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
if FLAGS.replicas==False:
 timestamp = str(int(time.time()))
 out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
 print("Writing to {}\n".format(out_dir))
 if not os.path.exists(out_dir):
    os.makedirs(out_dir)
 # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
 checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
 checkpoint_prefix = os.path.join(checkpoint_dir, "model")
 if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
else:
 # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "replicas"))
 out_dir = os.path.abspath(os.path.join(os.path.curdir,"runs","outputs"))
 print("Writing to {}\n".format(out_dir))
 if not os.path.exists(out_dir):
    os.makedirs(out_dir)

 checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
 checkpoint_prefix = os.path.join(checkpoint_dir, "model")
 if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
 summary_dir= os.path.abspath(os.path.join(out_dir, "summary"))
 if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
 train_summary_dir = os.path.join(summary_dir,"train")
 dev_summary_dir= os.path.join(summary_dir,"dev")
 if not os.path.exists(train_summary_dir):
        os.makedirs(train_summary_dir)
 if not os.path.exists(dev_summary_dir):
        os.makedirs(dev_summary_dir)
def data_preprocess():
    # Data preprocess
    # =======================================================
    # Load data
    print("Loading data...")
    if not os.path.exists(os.path.join(out_dir,"data_x.npy")):
          x, y = data_helper.load_data_and_labels(FLAGS.data_file)
          # Get embedding vector
          x =x[:1000]
          y =y[:1000]
          sentences, max_document_length = data_helper.padding_sentences(x, '<PADDING>',padding_sentence_length=FLAGS.sequence_length)
          print(len(sentences[0]))
          if not os.path.exists(os.path.join(out_dir,"trained_word2vec.model")):
              x= np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size = FLAGS.embedding_dim, file_to_save = os.path.join(out_dir, 'trained_word2vec.model')))
          else:
              print('w2v model found...')
              x = np.array(word2vec_helpers.embedding_sentences(sentences, embedding_size = FLAGS.embedding_dim, file_to_save = os.path.join(out_dir, 'trained_word2vec.model'),file_to_load=os.path.join(out_dir, 'trained_word2vec.model')))
          y = np.array(y)
          # np.save(os.path.join(out_dir,"data_x.npy"),x)
          # np.save(os.path.join(out_dir,"data_y.npy"),y)
          del sentences
    else:
          print('data found...')
          x= np.load(os.path.join(out_dir,"data_x.npy"))
          y= np.load(os.path.join(out_dir,"data_y.npy"))
    print("x.shape = {}".format(x.shape))
    print("y.shape = {}".format(y.shape))

    # Save params
    if not os.path.exists(os.path.join(out_dir,"training_params.pickle")):
        training_params_file = os.path.join(out_dir, 'training_params.pickle')
        params = {'num_labels' : FLAGS.num_labels, 'max_document_length' : max_document_length}
        data_helper.saveDict(params, training_params_file)

    # Shuffle data randomly
    # np.random.seed(10)
    # shuffle_indices = np.random.permutation(np.arange(len(y)))
    # x_shuffled = x[shuffle_indices]
    # y_shuffled = y[shuffle_indices]
    # del x,y

    # x_train, x_test, y_train, y_test = train_test_split(x_shuffled, y_shuffled, test_size=0.2, random_state=42)  # split into training and testing set 80/20 ratio
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # split into training and testing set 80/20 ratio
    del x,y
    return x_train, x_test, y_train, y_test

if FLAGS.replicas==True:
    if FLAGS.job_name is None or FLAGS.job_name =="":
        raise ValueError("Must specify an explicit 'job_name")
    if FLAGS.task_index is None or FLAGS.task_index =="":
        raise ValueError("Must specify an explicit task_index")
    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    num_workers = len(worker_spec)
    cluster = tf.train.ClusterSpec({"ps": ps_spec,"worker":worker_spec})
    server = tf.train.Server(
                cluster, job_name=FLAGS.job_name,task_index=FLAGS.task_index
    )
    if FLAGS.job_name =="ps":
        server.join()

    elif FLAGS.job_name == "worker":
      x_train, x_test, y_train, y_test =data_preprocess()
      with tf.Graph().as_default():
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):
                 global_step = tf.Variable(0, name="global_step", trainable=False)
                 cnn = URLCNN(
                       sequence_length = FLAGS.sequence_length,
                       num_classes = FLAGS.num_labels,
                       embedding_size = FLAGS.embedding_dim,
                       filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                       num_filters = FLAGS.num_filters,
                       l2_reg_lambda = FLAGS.l2_reg_lambda)
                 # Define Training procedure
                 optimizer = tf.train.AdamOptimizer(1e-3)
                 grads_and_vars = optimizer.compute_gradients(cnn.loss)
                 if FLAGS.is_sync == 1:
                    #同步模式计算更新梯度
                    rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                            replicas_to_aggregate=len(
                                                              worker_spec),
                                                            total_num_replicas=len(
                                                              worker_spec),
                                                            use_locking=True)
                    train_op = rep_op.apply_gradients(grads_and_vars,
                                                   global_step=global_step)
                    init_token_op = rep_op.get_init_tokens_op()
                    chief_queue_runner = rep_op.get_chief_queue_runner()
                 else:
                     #异步模式计算更新梯度
                     train_op = optimizer.apply_gradients(grads_and_vars,
                                                   global_step=global_step)
                 init_op = tf.global_variables_initializer()
                 saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)
                 # Keep track of gradient values and sparsity (optional)
                 grad_summaries = []
                 for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                 grad_summaries_merged = tf.summary.merge(grad_summaries)
                 # Summaries for loss and accuracy
                 loss_summary = tf.summary.scalar("loss", cnn.loss)
                 acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
                 # Train Summaries
                 train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                 # Dev summaries
                 dev_summary_op = tf.summary.merge([loss_summary, acc_summary])

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters =["/job:ps",
                             "/job:worker/task:%d"%FLAGS.task_index]
        )
        # train_dir = tempfile.mkdtemp()
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            logdir=FLAGS.log_dir,
                            init_op=init_op,
                            summary_op=None,
                            saver=saver,
                            global_step=global_step

                            )

        with sv.prepare_or_wait_for_session(server.target,config=sess_config) as sess:
            # train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            if FLAGS.task_index == 0 and FLAGS.is_sync == 1:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
                       # Initialize all variables
            sess.run(init_op)

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                if sv.is_chief:
                  sv.summary_computed(sess,sess.run(train_summary_op,feed_dict))
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 50 ==0:
                   print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                if sv.is_chief:
                  sv.summary_computed(sess,sess.run(dev_summary_op,feed_dict))
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            # Generate batches
            batches = data_helper.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if sv.is_chief and current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_test,y_test)
                    print("")
                if sv.is_chief and current_step % FLAGS.checkpoint_every == 0:
                    sv.saver.save(sess,checkpoint_prefix,global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(checkpoint_prefix))
        sv.stop()

else:
    x_train, x_test, y_train, y_test =data_preprocess()
    print "Training..."
    # Training
    # =======================================================
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement = FLAGS.allow_soft_placement,
            log_device_placement = FLAGS.log_device_placement)

        sess = tf.Session(config = session_conf)
        with sess.as_default():
            cnn = URLCNN(
            sequence_length = x_train.shape[1],
            num_classes = y_train.shape[1],
            embedding_size = FLAGS.embedding_dim,
            filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters = FLAGS.num_filters,
            l2_reg_lambda = FLAGS.l2_reg_lambda)

            # Define Training procedure
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 50 ==0:
                   print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helper.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_test, y_test, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))






