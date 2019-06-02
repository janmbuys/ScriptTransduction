import os,sys,math
from collections import defaultdict

import numpy as np

from spacy.lang.en.stop_words import STOP_WORDS
STOP_WORDS.update([',','.','\'','"',';',':','(',')',''])

def STOP(vals):
  oked = []
  for left_count, left in vals:
    if left not in STOP_WORDS:
      oked.append((left_count, left))
  return oked

raw = open("raw.tsv",'w')

F = [line.strip().split() for line in open(sys.argv[1],'r')]

E = [F[i] for i in range(0, len(F), 4)]
D = [F[i+1] for i in range(0, len(F), 4)]
A = [[int(a) for a in F[i+2]] for i in range(0, len(F), 4)]

j_totals = defaultdict(int)
j_total = 0
e_totals = defaultdict(int)
e_total = 0
d_totals = defaultdict(int)
d_total = 0
lengths = []
for e,d,a in zip(E,D,A):
  mapping = [[] for _ in range(len(e))]
  for a_idx, d_idx in zip(a, range(len(d))):
    mapping[a_idx].append(d[d_idx])

  for i in range(len(mapping)):
    enc = e[i]
    dec = " ".join(mapping[i])

    #if len(lengths) > 7000:
    #  print(a, e, i, mapping, dec)
    #  sys.exit()
    if len(dec.replace("<S> ","")) > 0:
      lengths.append(len(mapping[i]))
      raw.write("{}\t{}\n".format(enc, dec.replace("<S> ","")))
    j_totals[(enc, dec)] += 1
    j_total += 1
    e_totals[enc] += 1
    e_total += 1
    d_totals[dec] += 1
    d_total += 1
raw.close()

e_probs = defaultdict(float)
for v in e_totals:
  e_probs[v] = e_totals[v]/e_total
d_probs = defaultdict(float)
for v in d_totals:
  d_probs[v] = d_totals[v]/d_total
j_probs = defaultdict(float)
for v in j_totals:
  j_probs[v] = j_totals[v]/j_total


PMI = [(math.log(e_totals[v[0]]*j_probs[v]/(e_probs[v[0]] * d_probs[v[1]])), v) for v in j_probs]
PMI.sort()
PMI.reverse()

o = open("pmi.txt",'w')
for p, v in PMI:
  if len(v[1]) > 0:
    o.write("{:8.6f}  {:20s} {:20s}\n".format(p, v[0], v[1]))
o.close()

lengths.sort()
lengths = np.array(lengths)
print(np.min(lengths), np.mean(lengths), lengths[int(len(lengths)/2)], np.max(lengths))
