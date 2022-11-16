import os, random
import torch

import const
from load import convert_raw_data
from voc import read_vocs, trim_rare_words
from model import Model
from train import run, batch2train_data
from evaluate import eval_model

CORPUS_NAME = 'cornell movie-dialogs corpus'
CORPUS_DIR = os.path.join('data', CORPUS_NAME)
FORMATTED_DATAFILE = 'formatted_movie_lines.txt'
MIN_COUNT = 3

#convert_raw_data(CORPUS_DIR, FORMATTED_DATAFILE)

voc, pairs = read_vocs(os.path.join(CORPUS_DIR, FORMATTED_DATAFILE), CORPUS_NAME)

# Print some pairs to validate
print('\nPairs:')
for pair in pairs[:10]:
    print(pair)

pairs = trim_rare_words(voc, pairs, MIN_COUNT)

"""
batch_size = 5
batches = batch2train_data(voc, [random.choice(pairs) for _ in range(batch_size)])
ivar, lengths, ovar, mask, max_target_len = batches

print('\ninput variable:', ivar)
print('\nlenghts:', lengths)
print('\noutput variable:', ovar)
print('\nmask:', mask)
print('\nmax target len:', max_target_len)
"""

model_name = 'cb_model'
model = Model(model_name, voc)

save_dir = os.path.join('data', 'save', model_name, CORPUS_NAME,
        '{}-{}_{}'.format(model.encoder_n_layers, model.decoder_n_layers, model.hidden_size))
#run(model, pairs, save_dir)
load_filename = os.path.join(save_dir, '{}_{}.tar'.format(4000, 'checkpoint'))
model.load(load_filename)
eval_model(model)

