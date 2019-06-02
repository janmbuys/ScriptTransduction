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
parser.add_argument('--embedding_path', type=str, default='inference/')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--slot_vocab_name', type=str, default='')

parser.add_argument('--load_model', action='store_true', default=False)
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

## prediction procedures ##

if args.load_model:
  print("Load existing model")
  net = torch.load(model_name)      # Load Saved Model
else:
  net = model.Seq2SeqWithAlignment(args, voc2i, i2voc)
net.cuda()

inps, outs = data_utils.populate_data(tdata, args.enc_max_length,
        args.dec_max_length, net.START)
v_inps, v_outs = data_utils.populate_data(vdata, args.enc_max_length,
        args.dec_max_length, net.START)

print("Train  {:5}".format(len(inps)))
print("Val    {:5}".format(len(v_inps)))

optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate) 

print("Starting Training")
patience_count = 0
prev_val = np.inf
batch_size = args.batch_size

if args.hard_em_train:
  max_sgd_train_iter = min(10, args.max_train_iter)
  max_em_iter = args.max_train_iter - max_sgd_train_iter
else:
  max_sgd_train_iter = args.max_train_iter
  max_em_iter = 0

for epoch in range(max_sgd_train_iter):
  print("Epoch", epoch)

  # Evaluation
  net.eval()
  v_inds = range(int(len(v_inps)/batch_size))
  v_tot = 0
  batches = 0
  for i in tqdm.tqdm(v_inds, ncols=80, disable=True):
    padded = data_utils.pad_data(v_inps[i*batch_size:(i+1)*batch_size],
            v_outs[i*batch_size:(i+1)*batch_size], batch_size)
    # step
    v_src = Variable(torch.from_numpy(padded[0]).cuda())
    v_tgt = Variable(torch.from_numpy(padded[1]).cuda())
    optimizer.zero_grad()
    net.eval()
    loss = torch.mean(net((v_src, v_tgt)))

    v_tot += torch.sum(loss).data
    batches += 1
  v_tot /= batches
  print("Validation Loss {:8.4f}".format(v_tot))
  if prev_val < v_tot + 0.01:
    patience_count += 1
    if patience_count == args.patience:
      break
  else:
    patience_count = 0

  prev_val = v_tot

  # Viterbi decoding
  if epoch % 10 == 0 and epoch > 0:
    viterbi_fname = log_path + ".alignments.{}.val.txt".format(epoch)
    net.print_viterbi(v_inps, v_outs, viterbi_fname)

  net.train()
  # Training
  data = np.copy(inps) 
  dout = np.copy(outs)
  inds = list(range(int(len(data)/batch_size)))
  random.shuffle(inds)
  iterate = tqdm.tqdm(inds, ncols=80, disable=True)
  for i in iterate:
    padded = data_utils.pad_data(data[i*batch_size:(i+1)*batch_size],
            dout[i*batch_size:(i+1)*batch_size], batch_size)
    # step
    v_src = Variable(torch.from_numpy(padded[0]).cuda())
    v_tgt = Variable(torch.from_numpy(padded[1]).cuda())
    optimizer.zero_grad()
    net.train()
    loss = torch.mean(net((v_src, v_tgt)))
    loss.backward()
    optimizer.step()

    iterate.set_description("Training Batch Loss {:8.4f}".format(loss.data))
  print("Completed training epoch.")

  net.eval()

  # Training Viterbi decoding
  if epoch % 10 == 0 and epoch > 0:
    net.print_viterbi(inps[:len(data)], outs[:len(data)], ".{}.train.txt".format(epoch))
    torch.save(net, log_path + ".{}.model.pt".format(epoch))

patience_count = 0

for epoch in range(max_em_iter):
  print("Hard EM Epoch", epoch)

  # Evaluation
  net.eval()
  v_inds = range(int(len(v_inps)/batch_size))
  v_tot = 0
  batches = 0
  for i in tqdm.tqdm(v_inds, ncols=80, disable=True):
    padded = data_utils.pad_data(v_inps[i*batch_size:(i+1)*batch_size],
            v_outs[i*batch_size:(i+1)*batch_size], batch_size)
    # step
    v_src = Variable(torch.from_numpy(padded[0]).cuda())
    v_tgt = Variable(torch.from_numpy(padded[1]).cuda())
    optimizer.zero_grad()
    net.eval()
    loss = torch.mean(net((v_src, v_tgt)))

    v_tot += torch.sum(loss).data
    batches += 1
  v_tot /= batches
  print("Validation Loss {:8.4f}".format(v_tot))
  if v_tot > prev_val - 0.01:
    patience_count += 1
    if patience_count == args.patience:
      break

  prev_val = v_tot
  
  net.train()
  # Training
  data = np.copy(inps)
  dout = np.copy(outs)
  inds = list(range(int(len(data)/batch_size)))
  random.shuffle(inds)
  iterate = tqdm.tqdm(inds, ncols=80, disable=True)
  for i in iterate:
    align = net.run_viterbi(data[i*batch_size:(i+1)*batch_size], 
                        dout[i*batch_size:(i+1)*batch_size])
    padded = data_utils.pad_data(data[i*batch_size:(i+1)*batch_size], 
                      dout[i*batch_size:(i+1)*batch_size],
                      batch_size,
                      alignments=align)
    # em_step
    v_src = Variable(torch.from_numpy(padded[0]).cuda())
    v_tgt = Variable(torch.from_numpy(padded[1]).cuda())
    v_algn = Variable(torch.from_numpy(padded[2]).cuda())
    optimizer.zero_grad()
    net.train()
    loss = torch.mean(net.hard_forward((v_src, v_tgt, v_algn)))
    loss.backward()
    optimizer.step()

    iterate.set_description("Training Batch Loss {:8.4f}".format(loss.data))
  print("Completed training epoch.")

  # Training Viterbi decoding
  if epoch % 5 == 0 and epoch > 0:
    net.print_viterbi(inps[:len(data)], outs[:len(data)], ".{}.train.txt".format(epoch))

  torch.save(net, log_path + ".{}.model.pt".format(epoch))

torch.save(net, log_path + ".final.model.pt")

