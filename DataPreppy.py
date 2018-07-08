'''
  A class to convert data to dataset format for tensoflow.
  Partially adapted from: https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6
'''
import tensorflow as tf
from random import shuffle


def expand(x):
    '''
    Hack. Because padded_batch doesn't play nice with scalres, so we expand the scalar  to a vector of length 1
    :param x:
    :return:
    From: https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6
    '''
    x['length'] = tf.expand_dims(tf.convert_to_tensor(x['length']), 0)
    return x


def deflate(x):
    '''
        Undo Hack. We undo the expansion we did in expand
        From: https://medium.com/@TalPerry/getting-text-into-tensorflow-with-the-dataset-api-ffb832c8bec6
    '''
    x['length'] = tf.squeeze(x['length'])
    return x


class DataPreppy():
  def __init__(self, prefix, input_voc_dict, input_data_file, outdir):
    self.vocabs = {}
    self.reverse_vocabs = {}
    self.prefix = prefix
    self.input_data = input_data_file
    self.outdir = outdir
    lines = [line for line in open(input_voc_dict)]
    for line in lines:
      parts = line.split()
      # if voc is space
      if (len(parts) == 1):
        print(line)
        self.vocabs[" "] = int(parts[0])
        self.reverse_vocabs[int(parts[0])] = ""
      else:
        self.vocabs[parts[0]] = int(parts[1])
        self.reverse_vocabs[int(parts[1])] = parts[0]

  def sequence_to_tf_example(self, sequence):
    """convert a sequence (already in final token form) to tf.SequenceExample

    Arguments:
      sequence {Int} -- Input List of tokens

    Returns:
      tf.SequenceExample
    """
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    # +2 For start and end
    ex.context.feature["length"].int64_list.value.append(sequence_length + 2)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_tokens.feature.add().int64_list.value.append(self.vocabs["<START>"])
    for token in sequence:
      fl_tokens.feature.add().int64_list.value.append(token)
    fl_tokens.feature.add().int64_list.value.append(self.vocabs["<EOS>"])
    return ex

  @staticmethod
  def parse(ex):
    """
      convert a serilized example to a tensor.
    """
    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }

    # Parse the example (returns a dictionary of tensors)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=ex,
        context_features=context_features,
        sequence_features=sequence_features
    )
    return {"seq": sequence_parsed["tokens"], "length": context_parsed["length"]}

  def save_to_tfrecord(self):
    # since our data is small we read it at once into memory.
    lines = [line for line in open(self.input_data)]
    all_data = []
    for line in lines:
      all_data.append(list(map(int, line.split())))

    # suffle
    shuffle(all_data)

    val = all_data[:1000]
    test = all_data[1000:2000]
    train = all_data[2000:]
    print("trian size:", len(train))
    for (data, path) in [(val, './data/' + self.prefix + '-val.tfrecord'), (test, './data/' + self.prefix + '-test.tfrecord'), (train, './data/' + self.prefix + '-train.tfrecord')]:
      with open(path, 'w') as f:
        writer = tf.python_io.TFRecordWriter(f.name)
      for example in data:
        # print(example)
        record = self.sequence_to_tf_example(sequence=example)
        writer.write(record.SerializeToString())
    pass

  @staticmethod
  def make_dataset(path, mode, batch_size=128):
    '''
      Makes  a Tensorflow dataset that is shuffled, batched and parsed according to BibPreppy.
      You can chain all the lines here, I split them into seperate calls so I could comment easily
      :param path: The path to a tf record file
      :param path: The size of our batch
      :return: a Dataset that shuffles and is padded
    '''
    # Read a tf record file. This makes a dataset of raw TFRecords
    dataset = tf.data.TFRecordDataset([path])
    # Apply/map the parse function to every record. Now the dataset is a bunch of dictionaries of Tensors
    dataset = dataset.map(DataPreppy.parse, num_parallel_calls=8)
    if mode == "train":
      # Shuffle the dataset
      dataset = dataset.shuffle(buffer_size=1000)

    #In order the pad the dataset, I had to use this hack to expand scalars to vectors.
    dataset = dataset.map(expand)
    # Batch the dataset so that we get batch_size examples in each batch.
    # Remember each item in the dataset is a dict of tensors, we need to specify padding for each tensor seperatly
    dataset = dataset.padded_batch(batch_size, padded_shapes={
        "length": 1, # always 1
        "seq": tf.TensorShape([None]) # seqeunce is variable length, we pass that information to TF
    })

    if mode == "train":  
      # unlimited repeats
      dataset = dataset.repeat()

    #Finnaly, we need to undo that hack from the expand function
    dataset = dataset.map(deflate)

    """
      here we can return the dataset; however in https://www.tensorflow.org/programmers_guide/datasets
      it is recommended to return iterator.
      ###return dataset
    """  
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features 
