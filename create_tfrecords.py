"""
  script for creating tfrecord.
"""
import DataPreppy as DP

DP1 = DP.DataPreppy("annakarenina_char", "./data/annakarenina_chars2id.txt", "./data/annakarenina_charid_data.txt", "./data")
DP1.save_to_tfrecord()

DP2 = DP.DataPreppy("annakarenina_word", "./data/annakarenina_word2id.txt", "./data/annakarenina_wordid_data.txt", "./data")
DP2.save_to_tfrecord()
