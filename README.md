[![Travis](https://img.shields.io/travis/rust-lang/rust.svg)]()
[![PyPI](https://img.shields.io/pypi/pyversions/Django.svg)]()

Characterstic
----------------------------------- 
 * Using Word2Vec+CNN to detect the Malicious URL and it's a really exquisite structure!
 
 * Finially result about 96.2% precision
 
 * High scalability supporting for Distributed System
 
 * Supporting for Online Learning  





Requirements
----------------------------------- 
 * Python 2.12
 * Tensorflow  1.1.0
 * Numpy
 * Gensim 2.0.0


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
                [--noreplicas] [--is_sync [IS_SYNC]] [--nois_sync]
                [--ps_hosts PS_HOSTS] [--worker_hosts WORKER_HOSTS]
                [--job_name JOB_NAME] [--task_index TASK_INDEX]
                [--log_dir LOG_DIR]

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
      --replicas [REPLICAS]
                        Use the dirstribution mode
      --noreplicas
      --is_sync [IS_SYNC]   Use the async or sync mode
      --nois_sync
      --ps_hosts PS_HOSTS   comma-separated lst of hostname:port pairs
      --worker_hosts WORKER_HOSTS
                        comma-separated lst of hostname:port pairs
      --job_name JOB_NAME   job name:worker or ps
      --task_index TASK_INDEX
                        Worker task index,should be >=0, task=0 is the master
                        worker task the performs the variable initialization
      --log_dir LOG_DIR     parameter and log info      
      
Distribution      
-----------------------------------    
   
    Let's take 192.168.0.107 as ps server , 10.211.55.13 and 10.211.55.14 as training server.
    Make every machine has a copy of the code.
   
### Async-parallelism mode:

![image](https://github.com/paradise6/DetectMaliciousURL/blob/master/data/screenshot/async.png)

          On 192.168.0.107:
          python train.py --replicas=True --job_name=ps --task_index=0 --ps_hosts=192.168.0.107:2222\
                           --worker_hosts=10.211.55.13:2222,10.211.55.14:2222
          On 10.211.55.13:
          python train.py --replicas=True --job_name=worker --task_index=0 --ps_hosts=192.168.0.107:2222\
                           --worker_hosts=10.211.55.13:2222,10.211.55.14:2222       
          On 10.211.55.14:
          python train.py --replicas=True --job_name=worker --task_index=1 --ps_hosts=192.168.0.107:2222\
                           --worker_hosts=10.211.55.13:2222,10.211.55.14:2222                 
     
     
     
 ### Sync-parallelism mode:
 ![image](https://github.com/paradise6/DetectMaliciousURL/blob/master/data/screenshot/sync.png)
       
          On 192.168.0.107:
          python train.py --replicas=True --is_sync=True --job_name=ps --task_index=0 --ps_hosts=192.168.0.107:2222\
                           --worker_hosts=10.211.55.13:2222,10.211.55.14:2222
          On 10.211.55.13:
          python train.py --replicas=True --is_sync=True --job_name=worker --task_index=0 --ps_hosts=192.168.0.107:2222\
                           --worker_hosts=10.211.55.13:2222,10.211.55.14:2222       
          On 10.211.55.14:
          python train.py --replicas=True --is_sync=True --job_name=worker --task_index=1 --ps_hosts=192.168.0.107:2222\
                           --worker_hosts=10.211.55.13:2222,10.211.55.14:2222  

Evaluation
----------------------------------- 

     python eval.py --help 
     usage: eval.py [-h] [--input_text_file INPUT_TEXT_FILE][--single_url SINGLE_URL]
               [--input_label_file INPUT_LABEL_FILE] [--batch_size BATCH_SIZE]
               [--checkpoint_dir CHECKPOINT_DIR] [--eval_train [EVAL_TRAIN]]
               [--noeval_train]
               [--allow_soft_placement [ALLOW_SOFT_PLACEMENT]]
               [--noallow_soft_placement]
               [--log_device_placement [LOG_DEVICE_PLACEMENT]]
               [--nolog_device_placement]


    python eval.py --checkpoint_dir ./runs/{TIME_DIR}/checkpoints}
  
### Single URL Detection
    
    python eval.py --checkpoint_dir ./runs/{TIME_DIR}/checkpoints} --single_url=hottraveljobs.com/forum/docs/info.php
    
Here I use the defualt checkpoint_dir to detection single_url    
    
    python eval.py --single_url=hottraveljobs.com/forum/docs/info.php

###  Panel Testing
    
    python eval.py --checkpoint_dir ./runs/{TIME_DIR}/checkpoints} --input_text_file="../data/data2.csv"






HTTP Server API
----------------------------------
This is the HTTP service to load TensorFlow model and inference to predict malicious url.

### Usage
Run HTTP server with [Django] and use HTTP client under /server

     ./manage.py runserver 0.0.0.0:8000
### Inference to predict url
Use url as your GET parameter
     
     127.0.0.1:8000/detection/predict/?url=appst0re.net/upload.aspx
And you will get
    
    Success to predict appst0re.net/upload.aspx, result: bad

### Implementation
    django-admin startproject server
  
    python manage.py startapp detection
  
    #Add customized urls and views.



References
----------------------------------- 
[[1]Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

[[2]Using Word2Vec+ CNN to Detect Malicious URL](http://blog.csdn.net/u011987514/article/details/71189491)

[[3]deep_recommend_system](https://github.com/tobegit3hub/deep_recommend_system/tree/master/http_service#implementation)

[[4]using-machine-learning-detect-malicious-urls](http://fsecurify.com/using-machine-learning-detect-malicious-urls/)

[[5]Malware URLs](http://malwareurls.joxeankoret.com)

[[6]Malicious URL Detection using Machine Learning](https://arxiv.org/abs/1701.07179)
