This is a simple practice code to use Estimator API and non-depreciated features of tensorflow (V 1.8) to create and build charecter
 LM using GRU units.
I have used following features:
1- Dataset
2- Estimator API

This code is based on Tensorflow DNN example (https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py)
 but adapted for RNNs.

Update:
I added RnnLmS2S.py  that uses seq2seq model for LM modeling. Basically no encoder and simple decoder and does not have the isses mentioned for
 the RNN model with estomator API.However, it seems to me it is converging much slower. For example, for a charecter LM the simple RNN model
can generate okish sequences even after 500iterations, but  RNN using seq2seq model has even probelm generating <EOS> symble after few 1000 iterations
 and tends to generate reptative patterns.
 It works better with words though. It seems long sequences cause issue for this model while not as much for simple RNN.


Issues:
1- For generation, I am using  Estimator.predict to predict every single charecter. However,
 everytime I call the predict method it loads the whole graph which makes genration very slow.
  I have not figured out to do it in a more efficientway yet.
2- I use estimator input and outputs for sending back and forth the state between the model method and the genrator method.
 As a result I had to use state_is_tuple=False 
which is prob. not a good approach.


Data:
    Anna Karenina text.

Files:
    DataPreppy.py : Class for converting data to tfrecord and also method to read dataset from these records. 
    data_preprocessing.py: scripts to clean the input data.
    create_tfrecords.py: create tfrecords for training/eval.
    RnnLm.py: main model and test


Steps:
    1- Data Prepration:
        1- Run data_preprocessing.py to clean the data_preprocessing    
        2- Run create_tfrecords.py to conver the data to tfrecords.
    2- Trianing:
        1- Run Python RnnLm.py
    3- Generation:
        1 - After training run: python RnnLm.py
