from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import os
import data_helper,word2vec_helpers
import tensorflow as tf
import numpy as np
import json
class PredictService(object):
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.sess = None
        self.graph = None
        self.input_x = None
        self.predictions=None
        self.dropout_keep_prob=None
        self.trained_word2vec_model_file=None
        self.training_params_file=None
        self.init_session_handler()


    def init_session_handler(self):
      self.graph=tf.Graph()
      with self.graph.as_default():
       self.sess = tf.Session()
       with self.sess.as_default():
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            # print("Use the model {}".format(ckpt.model_checkpoint_path))
            print "{}.meta".format(checkpoint_file)
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(self.sess, checkpoint_file)
            self.input_x = self.graph.get_operation_by_name("input_x").outputs[0]
            self.predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]
            self.dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        else:
            print("No model found, exit now")
            exit()
        # validate word2vec model file
        self.trained_word2vec_model_file = os.path.join(self.checkpoint_path, "..", "trained_word2vec.model")
        if not os.path.exists(self.trained_word2vec_model_file):
             print("Word2vec model file \'{}\' doesn't exist!".format(self.trained_word2vec_model_file))
             exit()
        print("Using word2vec model file : {}".format(self.trained_word2vec_model_file))

        # validate training params file
        self.training_params_file = os.path.join(self.checkpoint_path, "..", "training_params.pickle")
        if not os.path.exists(self.training_params_file):
            print("Training params file \'{}\' is missing!".format(self.training_params_file))
            exit()
        print("Using training params file : {}".format(self.training_params_file))


    def predict(self,url):
        params = data_helper.loadDict(self.training_params_file)
        num_labels = int(params['num_labels'])
        max_document_length = int(params['max_document_length'])
        x_raw = [url]
        sentences, max_document_length = data_helper.padding_sentences(x_raw, '<PADDING>', padding_sentence_length = max_document_length)
        x_test = np.array(word2vec_helpers.embedding_sentences(sentences, file_to_load = self.trained_word2vec_model_file))
        # print(x_test)
        # with self.graph.as_default():
        # with self.sess.as_default():

        result = self.sess.run(self.predictions, {self.input_x: x_test,self.dropout_keep_prob: 1.0})
        result = 'good' if result else 'bad'
        print("Request examples: {}, inference result: {}".format(url,result))
        return result

checkpoint_path = "../model/runs/1493912734/checkpoints/"
predict_service = PredictService(checkpoint_path)

# Create your views here.
def index(request):
    return HttpResponse(
        "You should GET /detection/predict/ .")


# Disable CSRF, refer to https://docs.djangoproject.com/en/dev/ref/csrf/#edge-cases
@csrf_exempt
def predict(request):
    if request.method == 'GET':
        url = request.GET.get('url',0)
        if url != 0:
          print url
          result = predict_service.predict(url)
          return HttpResponse("Success to predict {}, result: {}".format(url,
            result))
        else:
          return HttpResponse("Please input URL parameter!")
    else:
        return HttpResponse("Please use POST to request with data")