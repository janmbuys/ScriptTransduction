import numpy as np


def space(vals):
  """ Remove spaces """
  return [v for v in vals if v != 7]


def sort_data(inputs, outputs):
  """ 
    Sorted by input length and then output length
  """
  v = []
  for i, o in zip(inputs, outputs):
    v.append((len(i), len(o), i, o))
  v.sort(key=lambda x: (x[0], x[1]))

  sorted_inputs = []
  sorted_outputs = []
  for len_i, len_o, i, o in v:
    if len_i > 0 and len_o > 0:
      sorted_inputs.append(i)
      sorted_outputs.append(o)
  return sorted_inputs, sorted_outputs
  

def read_template(path, voc2i, enc_max_length, dec_max_length, set_slot_vocab=True):
    tokens = []
    tags = []
    absts = []
    slot_restrictions = []
    size = 0

    NONE = voc2i["<PAD>"]
    UNK = voc2i["#UNK#"]
    START = voc2i["<S>"]

    if set_slot_vocab:
      slot_vocab = set()
    else:
      slot_vocab = None

    slot_restriction = set([START,NONE,UNK,voc2i['.']])
    prev_abs = ''
    abs_ind = 0
    ind = 0
    with open(path, 'r') as f:
      for line in f:
        entry = line.split('\t')
        abstract, concrete, commonsense_tags = entry[0], entry[1], entry[2]
        abst = abstract.strip()
        if abst != prev_abs:
          for j in range(abs_ind, ind):
            slot_restrictions.append(slot_restriction)
          slot_restriction = set([START,NONE,UNK,voc2i['.']])
          abs_ind = ind
        prev_abs = abst

        tgs = ['1'] + commonsense_tags.strip().split()
        tags.append(list(map(int, tgs))[:dec_max_length])

        line = np.array([voc2i[w] if w in voc2i else UNK for w in concrete.strip().split()])
        toks = np.insert(line, 0, START)[:dec_max_length]
        tokens.append(toks)

        abs_line = np.array([voc2i[w] if w in voc2i else UNK for w in abstract.strip().split()])
        abs_toks = abs_line[:enc_max_length]
        absts.append(abs_toks)

        if set_slot_vocab:
          for i, w in enumerate(toks):
            if tgs[i] == '0':
              slot_vocab.add(w)
              slot_restriction.add(w)
        ind += 1
      if abs_ind < ind:
        for j in range(abs_ind, ind):
          slot_restrictions.append(slot_restriction)

    return tokens, absts, tags, slot_restrictions, slot_vocab


def pad_data(inputs, outputs, batch_size, alignments=None):
  max_i = max([len(i) for i in inputs])
  max_o = max([len(o) for o in outputs])
  if alignments is not None:
    max_a = max([len(a) for a in alignments])
  
  padded_i = np.zeros((batch_size, max_i), dtype=np.int64)
  padded_o = np.zeros((batch_size, max_o), dtype=np.int64)
  if alignments is not None:
    padded_a = np.zeros((batch_size, max_a), dtype=np.int64)
  else:
    padded_a = None

  for i in range(batch_size):
    padded_i[i, :len(inputs[i])] = np.copy(inputs[i])
    padded_o[i, :len(outputs[i])] = np.copy(outputs[i])
    if alignments is not None:
      padded_a[i, :len(alignments[i])] = np.copy(alignments[i])

  return padded_i, padded_o, padded_a


def populate_data(data_set, enc_max_length, dec_max_length, START):
  inps = []
  outs = []
  for c, t1, t2, t3, t4, t5 in data_set:
    out = t1
    for t in [t2, t3, t4, t5]:
      if t != [7, 6]:
        out += t
    outs.append(np.insert(np.array(space(out))[:dec_max_length-1], 0, START))
    inps.append(np.array(space(c))[:enc_max_length])
  inps, outs = sort_data(inps, outs)
  return inps, outs

