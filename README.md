
Using Word2Vec+CNN to detect the Malicious url



Requirements
----------------------------------- 
    Python 2.12
    Tensorflow > 0.12
    Numpy


Training
----------------------------------- 

    python train.py --help
    usage: train.py [-h] [--data_file DATA_FILE] [--num_labels NUM_LABELS]
                [--embedding_dim EMBEDDING_DIM] [--filter_sizes FILTER_SIZES]
                [--num_filters NUM_FILTERS]
                [--dropout_keep_prob DROPOUT_KEEP_PROB]
                [--l2_reg_lambda L2_REG_LAMBDA] [--batch_size BATCH_SIZE]
                [--num_epochs NUM_EPOCHS] [--evaluate_every EVALUATE_EVERY]
                [--checkpoint_every CHECKPOINT_EVERY]
                [--num_checkpoints NUM_CHECKPOINTS]
                [--allow_soft_placement [ALLOW_SOFT_PLACEMENT]]
                [--noallow_soft_placement]
                [--log_device_placement [LOG_DEVICE_PLACEMENT]]
                [--nolog_device_placement]


        optional arguments:
      -h, --help            show this help message and exit
      --data_file DATA_FILE
                        Data source
      --num_labels NUM_LABELS
                        Number of labels for data. (default: 2)
      --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
      --filter_sizes FILTER_SIZES
                        Comma-spearated filter sizes (default: '3,4,5')
      --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
      --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
      --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularization lambda (default: 0.0)
      --batch_size BATCH_SIZE
                        Batch Size (default: 64)
      --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 200)
      --evaluate_every EVALUATE_EVERY
                        Evalue model on dev set after this many steps
                        (default: 100)
      --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (defult: 100)
      --num_checkpoints NUM_CHECKPOINTS
                        Number of checkpoints to store (default: 5)
      --allow_soft_placement [ALLOW_SOFT_PLACEMENT]
                        Allow device soft device placement
      --noallow_soft_placement
      --log_device_placement [LOG_DEVICE_PLACEMENT]
                        Log placement of ops on devices
      --nolog_device_placement


References
----------------------------------- 
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

[Using Word2Vec+ CNN to Detect Malicious URL](http://blog.csdn.net/u011987514/article/details/71189491)



