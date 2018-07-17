# File: RnnLmS2S.py
# Amir Harati, July 2018
"""
    Main model class that wrap the custom estimator.
    This model use decoder from seq2seq library instead of normal RNN.
    This allows to not initite the graph for every sample (issue with the other model.)
    Suggestion from: https://stackoverflow.com/questions/51277181/text-generation-using-estimator-api?noredirect=1#comment89536499_51277181
    also this code helped: https://github.com/ilblackdragon/tf_examples/tree/master/seq2seq
"""

import DataPreppy as DP
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense
import logging

# Fix for chnaging len error:
# from: https://github.com/tensorflow/nmt/issues/117
class FixedHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):
    def sample(self, *args, **kwargs):
        batch_size = 1
        result = super().sample(*args, **kwargs)
        result.set_shape([batch_size])
        return result

class RnnLmS2S:
    def __init__(self, model_size, embedding_size, num_layers,
         keep_prob, vocabs, reverse_vocabs, batch_size, num_itr, train_tfrecords, eval_tfrecords,
         model_dir):
        self.params = dict()
        self.params["model_size"] = model_size
        self.params["embedding_size"] = embedding_size
        self.params["num_layers"] = num_layers
        self.params["keep_prob"] = keep_prob
        self.vocab_size = len(vocabs)
        self.params["vocab_size"] = self.vocab_size

        self.batch_size = batch_size
        self.num_itr = num_itr
        self.train_tfrecords = train_tfrecords
        self.eval_tfrecords = eval_tfrecords
        self.model_dir = model_dir
        self.vocabs = vocabs
        self.reverse_vocabs = reverse_vocabs

    def train(self):
        """
            train the custom estimator.
        """
        est = tf.estimator.Estimator(
            model_fn=self._model,
            model_dir=self.model_dir,
            params=self.params)
        est.train(self._train_input_fn, steps=self.num_itr)

    def generate(self):
        """
            generate new sequences.
        """
        model_size = self.params["model_size"]
        
        num_layers = self.params["num_layers"]
        est = tf.estimator.Estimator(
            model_fn=self._model,
            model_dir=self.model_dir,
            params=self.params,
            warm_start_from=self.model_dir)
        
        current_seq_ind = []
        X = np.zeros((1, 1), dtype=np.int32)
        X[0, 0] = 0
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"seq": X},
            num_epochs=1,
            shuffle=False)
            
        result = est.predict(input_fn=predict_input_fn)
        print(next(result))

    def _train_input_fn(self):
        return DP.DataPreppy.make_dataset(self.train_tfrecords, "train", self.batch_size)

    def _eval_input_fn(self):
        return DP.DataPreppy.make_dataset(self.eval_tfrecords, "eval", self.batch_size)


    def _model(self, features, labels, mode, params):
        """
            main model.
        """
        # labels is None in this case since we use features as labels in LM
        # How to update the params such as keep_prob during training? It seems we need to divide the  training into chunks
        # and check for conditons in between chunks.
        
        #if mode == tf.estimator.ModeKeys.PREDICT:
        #    lengths = 0 

        #else:
        if mode == tf.estimator.ModeKeys.TRAIN:
            lengths = features['length']
            sequence = features['seq']
            batch_size = tf.shape(sequence)[0]
            start_token = tf.ones([self.vocabs["<START>"]], tf.int32) 
        elif mode == tf.estimator.ModeKeys.PREDICT:
            start_token = tf.ones([self.vocabs["<START>"]], tf.int32) # update this
            batch_size = 1
            #sequence = self.vocabs["<START>"]
            sequence = features['seq']
            lengths = [1]

        model_size = params["model_size"]
        num_layers = params["num_layers"]
        keep_prob = params["keep_prob"]
        vocab_size = params["vocab_size"]
        embedding_size = params["embedding_size"]

        with tf.variable_scope("main", initializer=tf.contrib.layers.xavier_initializer()):
            embedding = tf.get_variable("embedding",
            [vocab_size, embedding_size], dtype=tf.float32)
            #if mode == tf.estimator.ModeKeys.TRAIN:
            embed_seq = tf.nn.embedding_lookup(embedding, sequence)
            output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(sequence, 0)), 1)  #update this
            train_helper = tf.contrib.seq2seq.TrainingHelper(embed_seq, output_lengths)
            #if mode == tf.estimator.ModeKeys.PREDICT:
            pred_helper = FixedHelper(embedding, 
                    start_tokens=tf.to_int32(start_token), end_token=2) # update this

            def decode(helper, scope, reuse=None):
                with tf.variable_scope(scope, reuse=reuse):
                    cells = []
                    for i in range(num_layers):
                        c = tf.nn.rnn_cell.GRUCell(model_size)
                        c = tf.nn.rnn_cell.DropoutWrapper(c, input_keep_prob=keep_prob,
                                                        output_keep_prob=keep_prob)
                        cells.append(c)   
                    cell = tf.nn.rnn_cell.MultiRNNCell(cells) 
                    #out_cell = tf.contrib.rnn.OutputProjectionWrapper(cell, vocab_size, reuse=reuse)
                    projection_layer = Dense(units=vocab_size, use_bias=True, name="logits")
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=cell, helper=helper,
                        initial_state=cell.zero_state(dtype=tf.float32, batch_size=batch_size),
                        output_layer=projection_layer)
                    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder, output_time_major=False,
                        impute_finished=True)
                    return outputs
            #if mode == tf.estimator.ModeKeys.TRAIN:
            train_outputs = decode(train_helper, 'decode')
            #elif mode == tf.estimator.ModeKeys.PREDICT:
            pred_outputs = decode(pred_helper, 'decode', reuse=True)
            if mode == tf.estimator.ModeKeys.TRAIN:
                outputs = train_outputs.rnn_output
            elif mode == tf.estimator.ModeKeys.PREDICT:
                outputs = pred_outputs.rnn_output

            # run the rnn
            #if mode == tf.estimator.ModeKeys.PREDICT:            
            #    outputs, state = tf.nn.dynamic_rnn(cell, embed_seq, dtype=tf.float32, initial_state=init_state, scope="DRNN")
            #else:
            #    outputs, state = tf.nn.dynamic_rnn(cell, embed_seq, dtype=tf.float32, sequence_length=lengths, scope="DRNN")
    
            if mode == tf.estimator.ModeKeys.TRAIN:
                targets = sequence[:, 1:]
                # for predict we use the output for train exclude the last timestep
                outputs = outputs[:, :-1, :]

            with tf.variable_scope("softmax"):
                logits = outputs
                # alternatively we can remove projection layer above and change the logits to below.
                #logits = tf.layers.dense(outputs, vocab_size, None, name="logits")
                
                # probablities.
                probs = tf.nn.softmax(logits, name="probs")
                # in case in prediction mode return
                if mode == tf.estimator.ModeKeys.PREDICT:
                        return tf.estimator.EstimatorSpec(
                            mode=mode,
                            predictions={"probs": probs})

                mask = tf.sign(tf.abs(tf.cast(targets, dtype=tf.float32)))
                loss = tf.contrib.seq2seq.sequence_loss(logits, targets,
                                                        mask,
                                                        average_across_timesteps=False,
                                                        average_across_batch=True)
                # alternative loss
                #loss = tf.losses.sparse_softmax_cross_entropy(targets,
                #                                       logits,
                #                                       weights=mask)


            # compute the loss and also predictions
            loss = tf.reduce_mean(loss)
            tf.summary.scalar("loss", loss)
            with tf.variable_scope("train_op"):
                learning_rate = tf.Variable(0.0, trainable=False)
                initial_learning_rate = tf.constant(0.001)
                learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                        tf.train.get_global_step(), 100, 0.99)
                tf.summary.scalar("learning_rate", learning_rate)
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                # Visualise gradients
                vis_grads = [0 if i is None else i for i in grads]
                for g in vis_grads:
                    tf.summary.histogram("gradients_" + str(g), g)
                train_op = optimizer.apply_gradients(zip(grads, tvars),
                 global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=None,
                loss=loss,
                train_op=train_op
            )

def test():
    tf.logging.set_verbosity(logging.DEBUG)
    dpp = DP.DataPreppy("char", "./data/annakarenina_chars2id.txt", "", "")
    m = RnnLmS2S(model_size=128, embedding_size=100, num_layers=1,
         keep_prob=1.0, batch_size=64, num_itr=200, vocabs=dpp.vocabs, reverse_vocabs=dpp.reverse_vocabs,
         train_tfrecords='./data/annakarenina_char-train.tfrecord',
         eval_tfrecords='./data/annakarenina_char-eval.tfrecord',
         model_dir="./checkpoints")
    #m.train()
    m.generate()

if __name__ == "__main__":
    test()