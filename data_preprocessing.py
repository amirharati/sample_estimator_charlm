"""
extract and normalize from raw data and save the resutls.

Amir Harati, April 2018
issues:
  1- chunking is not used (e.g. new york would be new + york)
"""
import json
import textacy as tc
import glob, os
import spacy as sp
import  re

regex = re.compile('[^0-9a-zA-Z\s+\.\?]')

outdir = "./data"

data = []
lines = [line.strip() for line in open("data/anna_karenina.txt")]
for line in lines:
  if len(line) > 0:
    #line = line.replace('?', '.')
    #line = line.replace('!', '.')
    line = line.replace('?', ' ? ')
    #line = line.replace('\'ve',' have ')
    #line = line.replace('\'re',' are ')
    line = line.replace('...', ' ')
    line = line.replace('..', ' ')
    line = line.replace('.', ' . ')
    line = line.lower()
    nline = regex.sub('', line)
    #print("*** ", nline)
    data.append(nline.split())

data = [item for sublist in data for item in sublist]

#print(data[0:1000])

line = ""
pdata = []
words = set()
chars = set()
for w in data:
  if w != ".":
    words.add(w)
    for c in w:
      chars.add(c)
    line += w + " "
  else:
    pdata.append(line[0:-1])
    line = ""

chars = list(chars)
words = list(words)

print("#chars: ", len(chars))
print("#words:", len(words))

words = sorted(words)
chars = sorted(chars)
# for words add start, end and pad symboles
words = ["<PAD>", "<START>", "<EOS>"] + words
# for chars in addition to above add space
chars = ["<PAD>", "<START>", "<EOS>", " "] + chars

words_to_ids = {w: id for id, w in enumerate(words)}
ids_to_words = {words_to_ids[x]: x for x in words_to_ids}
chars_to_ids = {w: id for id, w in enumerate(chars)}
ids_to_chars = {chars_to_ids[x]: x for x in chars_to_ids}


# save data
with open(outdir + "/annakarenina_word2id.txt", "w") as wif:
  for key, val in words_to_ids.items():
    wif.write(key + "\t" + str(val) + "\n")

with open(outdir + "/annakarenina_chars2id.txt", "w") as wif:
  for key, val in chars_to_ids.items():
    wif.write(key + "\t" + str(val) + "\n")

with open(outdir + "/annakarenina_text_data.txt", "w") as f:
  for sen in pdata:
    f.write(sen + "\n")

with open(outdir + "/annakarenina_charid_data.txt", "w") as f:
  for sen in pdata:
    ostr = ""
    #le = [x for x in tweet]
    for c in sen:
      ostr = ostr + str(chars_to_ids[c]) + " "
    f.write(ostr + "\n")


with open(outdir + "/annakarenina_wordid_data.txt", "w") as f:
  for sen in pdata:
    ostr = ""
    for word in sen.split():
      #print(word)
      ostr = ostr + str(words_to_ids[word]) + " "
    f.write(ostr + "\n")
