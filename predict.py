import pickle, gzip
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import random
import math
import argparse
import os, sys

import model
import data_utils

parser = argparse.ArgumentParser(description='Sequence Models')
parser.add_argument('--output_path', type=str, default='output/')
parser.add_argument('--data_path', type=str, default='data/examples/')
parser.add_argument('--template_path', type=str, default='data/templates/')
parser.add_argument('--embedding_path', type=str, default='inference/')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--slot_vocab_name', type=str, default='inference/approved.5.txt')

parser.add_argument('--generate', action='store_true', default=False)
parser.add_argument('--score_surprisal', action='store_true', default=False)
parser.add_argument('--score', action='store_true', default=False)
parser.add_argument('--val_viterbi', action='store_true', default=False)

parser.add_argument('--model_type', type=str, default='alignment') 
   # alignment | seq2seq | seq2seq_attention | unconditional
parser.add_argument('--bidirectional', action='store_true', default=False)

parser.add_argument('--param_transitions', dest='param_transitions', action='store_true')
parser.add_argument('--no_param_transitions', dest='param_transitions', action='store_false')
parser.set_defaults(param_transitions=False)

parser.add_argument('--hard_em_train', dest='hard_em_train', action='store_true')
parser.add_argument('--no_hard_em_train', dest='hard_em_train', action='store_false')
parser.set_defaults(hard_em_train=True)

parser.add_argument('--batch_size', type=int, default=32, metavar='N')
parser.add_argument('--drop_rate', type=float, default=0.5) 
parser.add_argument('--learning_rate', type=float, default=1e-3) 

parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--enc_max_length', type=int, default=50)
parser.add_argument('--dec_max_length', type=int, default=50)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--max_train_iter', type=int, default=50)
parser.add_argument('--seed', type=int, default=1931)

args = parser.parse_args()

if args.seed > 0:
  torch.manual_seed(args.seed)

torch.set_printoptions(threshold=1000, edgeitems=10, linewidth=300)
np.set_printoptions(linewidth=200)

tdata = pickle.load(gzip.open(args.data_path + '/train.freq.pkl.gz','rb'))
vdata = pickle.load(gzip.open(args.data_path + '/valid.freq.pkl.gz','rb'))
voc2i = pickle.load(gzip.open(args.data_path + '/v2i.freq.pkl.gz','rb'))
i2voc = pickle.load(gzip.open(args.data_path + '/i2v.freq.pkl.gz','rb'))

if not os.path.exists(args.output_path):
  os.makedirs(args.output_path)
log_path = args.output_path + "/"

# Construct log filename path
log_path += "drop" + str(int(args.drop_rate*100)) 
log_path += ".{}-{}".format(args.enc_max_length, args.dec_max_length)
log_path += "." + args.model_type + ".h{}".format(args.hidden) 

if args.model_type != "alignment":
  args.hard_em_train = False
if args.param_transitions:
  log_path += ".paramT"
if args.hard_em_train:
  log_path += ".EM"

model_name = log_path + ".final.model.pt" if args.model_name == "" else args.model_name
print("Model name: %s" % (model_name))

# Load Saved Model
net = torch.load(model_name)      
net.cuda()
net.eval()

if args.val_viterbi: 
  inps, outs = data_utils.populate_data(tdata, args.enc_max_length,
          args.dec_max_length, voc2i["<S>"])
  v_inps, v_outs = data_utils.populate_data(vdata, args.enc_max_length, args.dec_max_length, voc2i["<S>"])
  net.print_viterbi(v_inps, v_outs, ".final.val.txt")


if args.generate: 
  slot_vocab = set()
  slot_fname = args.slot_vocab_name
  with open(slot_fname, 'r') as slot_f:
    for line in slot_f:
      if line.strip() in voc2i:
        slot_vocab.add(voc2i[line.strip()])

  generate_sets = ["valid"]

  print("Generating from abstract.")
  for sset in generate_sets:
    tfile = args.template_path + '/' + sset + ".template.tsv"
    tokens, abstract, tags, _, _ = data_utils.read_template(tfile, voc2i,
            args.enc_max_length, args.dec_max_length, True)
    net.run_generate(log_path, sset, tokens, abstract, tags,
            slot_vocab=slot_vocab) # TODO check that both oracle and greedy decoding are included
 
if args.score_surprisal:
  slot_vocab = set()
  slot_fname = args.slot_vocab_name
  with open(slot_fname, 'r') as slot_f:
    for line in slot_f:
      if line.strip() in voc2i:
        slot_vocab.add(voc2i[line.strip()])

  generate_sets = ["valid"]

  print("Scoring gold cloze for surprisal and mean rank.")
  for sset in generate_sets:
    tfile = args.template_path + '/' + sset + ".template.tsv"
    tokens, abstract, tags, _, _ = data_utils.read_template(tfile, voc2i,
            args.enc_max_length, args.dec_max_length, True)
    net.run_surprisal(sset, tokens, abstract, tags, slot_vocab=slot_vocab)
    

if args.score: 
  print("Scoring cloze for exact match.")
  pre = "lm"
  abstract = data_utils.read_file(open("cloze_alignment/{}.valid.abs.txt".format(pre)))
  concrete = data_utils.read_file(open("cloze_alignment/{}.valid.conc.txt".format(pre)))
  counts = [[int(v) for v in line.strip().split()] for line in open("cloze_alignment/{}.valid.count".format(pre))]
  vals = net.run_evaluation(abstract, concrete)

  correct = 0
  total = 0
  previous = [0,0]
  previous_val = -1e100
  for i in range(len(vals)):
    val = vals[i]
    count = counts[i]

    if val > previous_val:
      previous_val = val
      previous = count

    if count[0] == 0:     # End of a block
      correct += previous[0]
      total += previous[1]

      previous = [0,0]
      previous_val = -1e100

  print("Done scoring cloze.")
  print(correct, total, 100*correct/total)


