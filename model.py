import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import tqdm


class Candidate():
  def __init__(self):
    template_str = ""
    output_seq = []
    slot_strs = []
    slot_nll = 0
    template_score = 0
    hidden_state = None 

class Seq2SeqWithAlignment(nn.Module):
  def __init__(self, args, voc2i, i2voc):
    super(Seq2SeqWithAlignment, self).__init__()
    self.hidden = args.hidden 
    self.embed_dim = 100 # size of glove embeddings
    self.vocab_size = len(voc2i)
    self.align_last = True
    self.update_embeddings = False
    self.share_embeddings = False
    self.enc2dec = False # if True, initialize decoder with encoder final hidden state in alignment-based model
    self.enc_max_length = args.enc_max_length
    self.dec_max_length = args.dec_max_length
    self.i2voc = i2voc

    self.NONE = voc2i["<PAD>"]
    self.UNK = voc2i["#UNK#"]
    self.START = voc2i["<S>"]

    self.model_type = args.model_type 
    self.use_param_transitions = args.param_transitions

    self.embeddings = nn.Embedding(self.vocab_size, self.embed_dim).cuda()
    # Load glove embeddings
    self.embeddings.weight.data.copy_(
        torch.from_numpy(np.load("inference/infer_glove.npy")))
    self.embeddings.requires_grad = self.update_embeddings
    
    self.bidirectional = args.bidirectional
    self.enc = nn.LSTM(self.embed_dim, self.hidden, batch_first=True, bidirectional=args.bidirectional)
    self.dec = nn.LSTM(self.embed_dim, self.hidden, batch_first=True)

    self.drop = nn.Dropout(p=args.drop_rate)
    
    self.collapse_bider = nn.Linear(2*self.hidden, self.hidden)

    if self.model_type == "alignment" or self.model_type == "seq2seq_attention":
      self.emit_hidden = nn.Linear(2*self.hidden, self.embed_dim)
      self.emit_hidden_LM = nn.Linear(self.hidden, self.embed_dim)
    else:
      self.emit_hidden = nn.Linear(self.hidden, self.embed_dim)

    self.emit = nn.Linear(self.embed_dim, self.vocab_size)

    if self.share_embeddings:
      assert self.update_embeddings
      self.emit.weight = self.embeddings.weight

    self.shift_hidden = nn.Linear(2*self.hidden, self.hidden)
    self.shift = nn.Linear(self.hidden, 1, bias=False) # parameterized transition function
     
    if self.model_type == "seq2seq_attention":
      self.attention_bilinear = nn.Linear(self.hidden, self.hidden)

    # Transition Matrix
    self.fixed_shift_p = math.log(self.dec_max_length / (self.dec_max_length + self.enc_max_length))
    self.fixed_emit_p = math.log(self.enc_max_length / (self.dec_max_length + self.enc_max_length))

    p_shift = []
    for i in range(self.enc_max_length):
      p_shift.append([])
      for j in range(self.enc_max_length):
        if i+j < self.enc_max_length:
          p_shift[i].append((self.enc_max_length - (i+j)-1)*self.fixed_shift_p + self.fixed_emit_p)
        else:
          p_shift[i].append(0)
    p_shift = np.array(p_shift, dtype=np.float32)
    self.tran = Variable(torch.from_numpy(p_shift), requires_grad=False).cuda()
    
    # Mask out bottom right of matrix
    mask_br = np.tril(np.zeros((self.enc_max_length, self.enc_max_length), dtype=np.float32) - np.inf)
    mask_br = np.fliplr(mask_br)
    for i in range(self.enc_max_length): 
      mask_br[i, self.enc_max_length - i - 1] = 0
    self.mask = Variable(torch.from_numpy(mask_br.copy()), requires_grad=False).cuda()

  def init_hidden(self, batch_size):
    weight = next(self.parameters()).data
    return (Variable(weight.new(batch_size, 1, self.hidden).zero_()),
            Variable(weight.new(batch_size, 1, self.hidden).zero_()))

  def flip(self, mat):
    # https://discuss.pytorch.org/t/how-to-reverse-a-torch-tensor/382
    idx = torch.LongTensor([i for i in range(mat.size(1)-1, -1, -1)]).cuda()
    inv_mat = mat.index_select(1, idx)
    return inv_mat

  def viterbi(self, vals, a_last=True):
    x, y = vals
    x = self.embeddings(x)
    dec_words = y.clone()
    y = self.embeddings(y)

    E, enc_final_hidden = self.enc(x)
    if self.bidirectional:
      E = self.collapse_bider(E)
      enc_final_hidden = (torch.mean(enc_final_hidden[0], 0, keepdim=True), 
                          torch.mean(enc_final_hidden[1], 0, keepdim=True))
    if (self.model_type == "seq2seq" or self.model_type == "seq2seq_attention"
            or (self.model_type == "alignment" and self.enc2dec)):
      Et = E.transpose(2, 1)
      D, _ = self.dec(y, enc_final_hidden) 
    else:
      D, _ = self.dec(y)

    # Forward Pass
    N = x.size()[0] # batch size
    assert N == 1 # For decoding
    T = y.size()[1] # dec_length
    K = x.size()[1] # enc_length
    V = self.vocab_size

    if self.model_type == "unconditional" or self.model_type == "seq2seq":
      alpha = Variable(torch.zeros(N)).cuda()
      for t in range(1, T):                        
        Dt = D[:,t-1,:].view(N, self.hidden)
        emissions = F.log_softmax(self.emit(self.drop(F.relu(self.emit_hidden(Dt)))), 1)
        word_idx = dec_words[:, t].unsqueeze(1)
        emission_alpha = emissions.gather(1, word_idx).view(-1)
        alpha += emission_alpha

      if self.model_type == "seq2seq": 
        alignments = np.ones(T, dtype=np.int)*(K-1)
      else:
        alignments = np.zeros(T, dtype=np.int)

      return alignments, alpha

    elif self.model_type == "seq2seq_attention":
      alpha = Variable(torch.zeros(N)).cuda()
      for t in range(1, T):                        
        Dt = D[:,t-1,:].view(N, self.hidden)
        query = self.attention_bilinear(Dt).view(N, 1, self.hidden)
        align = torch.bmm(query, Et).view(N, K)
        align = F.softmax(align, 1).view(N, 1, K)
        cur_alpha = torch.bmm(align, E).view(N, self.hidden)

        joint = torch.cat((Dt, cur_alpha), 1)
        emissions = F.log_softmax(self.emit(self.drop(F.relu(self.emit_hidden(joint)))), 1)
        word_idx = dec_words[:, t].unsqueeze(1)
        emission_alpha = emissions.gather(1, word_idx).view(-1)
        alpha += emission_alpha
      alignments = np.ones(T, dtype=np.int)*(K-1)

      return alignments, alpha

    # Target Functionality
    # for j in range(1, J):
    #   for i in range(0, I):
    #     alpha(i,j) = p(d_j | h_e_i, h_d_j) x
    #                  max_k_0_i alpha(k, j-1) * p(a_j = i | a_j-1 = k)

    pre_alpha = Variable(torch.zeros(N, K)).cuda()
    cur_alpha = Variable(torch.zeros(N, K)).cuda()

    fixed_transitions = self.tran[self.enc_max_length-K:,:K].unsqueeze(0) # submatrix of right diagonal matrix
    mask = self.mask[self.enc_max_length-K:,:K].unsqueeze(0)
    indices = torch.zeros(T, K, dtype=torch.int)

    for t in range(1, T):                        
      cur_alpha = pre_alpha.unsqueeze(2).expand(N, K, K) 
      Dt = D[:,t-1,:].view(N, 1, -1).expand(N, K, self.hidden)
      joint = torch.cat((E,Dt), 2)
      
      # Transition
      if self.use_param_transitions:
        shift_logit = self.shift(self.drop(F.relu(self.shift_hidden(joint))))
        shift_p = F.logsigmoid(shift_logit).squeeze(2)
        emit_p = F.logsigmoid(-shift_logit).expand(N, K, K)

        # Magic to make more efficient 
        transitions = Variable(torch.zeros(N, K, K, dtype=torch.float)).cuda()
        for k in range(K-1):
          transitions[:,:K-k-1,:k+1] += shift_p[:,k].view(N, 1, 1).expand(N, K-k-1, k+1)

        # Unvectorized
        #for i in range(K):
        #  for j in range(K-i):
        #    # Shifting from j to (K-i-1)
        #    transitions[:,i,j] = torch.sum(shift_p[:,j:K-i-1].view(N, -1), 1)
   
        transitions += emit_p
        cur_alpha += transitions
      else:
        cur_alpha += fixed_transitions

      cur_alpha = cur_alpha + mask             # Zero-out bottom-right
      cur_alpha, ind = torch.max(cur_alpha, 1)

      indices[t] = self.flip(ind)
      cur_alpha = self.flip(cur_alpha)         # reorder alphas
       
      # Emission
      emissions = F.log_softmax(self.emit(self.drop(F.relu(self.emit_hidden(joint)))), -1)
      word_idx = dec_words[:, t].unsqueeze(1).expand(N,K).unsqueeze(2)
      cur_alpha = cur_alpha + emissions.gather(2, word_idx).squeeze()

      # Update
      pre_alpha = cur_alpha.clone()

    # Follow back-pointers:
    alignments = np.zeros(T, dtype=np.int)
    if self.align_last and a_last:
      val = pre_alpha[0, K - 1]     # Batch_size 1
      ind = K - 1
    else:
      val, ind = torch.max(pre_alpha, 1)
    ind = int(ind)
    for t in range(T-1, -1, -1):
      ind = int(indices[t][ind])
      alignments[t] = ind

    return alignments, val

  def run_phrase_generate(self, output_path, sset, seq_abstracts, seq_concrete):
    # Sample greedily for each word in an abstract
    temperature = 0.5
    max_phrase_len = 10

    outf = output_path + "/lm." + sset + ".phrase.generate.txt"
    assert self.model_type == "alignment"
    with open(outf, 'w') as outf:
      for i, abs_s in enumerate(seq_abstracts):
        abs_seq = torch.from_numpy(abs_s).cuda().unsqueeze(0)
        abs_seq_embed = self.embeddings(abs_seq)
        abs_str = ' '.join(self.i2voc[int(s)] for s in abs_s)
        outf.write(abs_str + '\n')
        conc_str = ' '.join(self.i2voc[int(s)] for s in seq_concrete[i])
        outf.write(conc_str + '\n')

        E, enc_final_hidden = self.enc(abs_seq_embed)
        K = abs_seq.size()[1]

        # Start by feeding start symbol:  
        output_seq = [self.START]
        input = torch.from_numpy(np.array([self.START])).view(1, 1).cuda()
        hidden = self.init_hidden(1)

        emb = self.embeddings(input)
        output, hidden = self.dec(emb, hidden)
        output = output.squeeze(1)

        for k in range(K):
          cur_alpha = E[:,k].view(1, self.hidden)
          align_out_str = ""

          emit_state = torch.cat((cur_alpha, output), 1)
          shift_logit = self.shift(self.drop(F.relu(self.shift_hidden(emit_state))))

          phrase_len = 0
          while shift_logit < 0 and len(output_seq) < self.dec_max_length and phrase_len < max_phrase_len:
            logits = self.emit(self.drop(F.relu(self.emit_hidden(emit_state))))
            word_weights = logits.squeeze().data.div(temperature) 
            word_weights[self.UNK] = -np.inf # disallow UNK
            word_weights = word_weights.exp().cpu()
               
            word_idx = int(torch.multinomial(word_weights, 1)[0])
            input.data.fill_(word_idx)
            output_seq.append(word_idx)
            word = self.i2voc[word_idx]
            align_out_str += word + ' '
            phrase_len += 1

            # next state
            emb = self.embeddings(input)
            output, hidden = self.dec(emb, hidden)
            output = output.squeeze(1)

            emit_state = torch.cat((cur_alpha, output), 1)
            shift_logit = self.shift(self.drop(F.relu(self.shift_hidden(emit_state))))
          outf.write(self.i2voc[int(abs_s[k])] + '\t' + align_out_str.strip() + '\n')
        outf.write('\n')

  def run_surprisal(self, sset, seq_tokens, seq_abstracts, seq_tags,
          seq_slot_restrictions=None, slot_vocab=None):
    if slot_vocab is not None:
      mask = np.ones(self.vocab_size)*-np.inf
      mask[0] = 0 # Have to allow UNK
      for w in slot_vocab:
        mask[w] = 0
    else:
      mask = np.zeros(self.vocab_size)
    seq_mask = torch.Tensor(mask).cuda()

    slot_count = 0
    slot_nll = 0
    slot_token_count = 0
    r_rank = 0
    top_k_count = 0

    print(len(seq_tokens))
    for i, seq in tqdm.tqdm(enumerate(seq_tokens), ncols=80, disable=True):
      abs_seq = torch.from_numpy(seq_abstracts[i]).cuda().unsqueeze(0)
      abs_seq_embed = self.embeddings(abs_seq)

      seq_t = torch.from_numpy(seq).cuda()
      seq_embed = self.embeddings(seq_t)

      E, enc_final_hidden = self.enc(abs_seq_embed)
      if self.bidirectional:
        E = self.collapse_bider(E)
        enc_final_hidden = (torch.mean(enc_final_hidden[0], 0, keepdim=True), 
                            torch.mean(enc_final_hidden[1], 0, keepdim=True))
      K = abs_seq.size()[1]
      if (self.model_type == "seq2seq" or self.model_type == "seq2seq_attention"
              or (self.model_type == "alignment" and self.enc2dec)):
        Et = E.transpose(2, 1)
        hidden = enc_final_hidden
      else:
        hidden = self.init_hidden(1)

      seed_start_indexes = []
      slot_start_indexes = []
      slot_state = False
      tags = seq_tags[i]
      assert tags[0] == 1

      for j, is_kept in enumerate(tags):
        if is_kept == 0 and not slot_state:
          slot_start_indexes.append(j)  
          slot_state = True
        elif j == 0 or is_kept == 1 and slot_state:
          seed_start_indexes.append(j)  
          slot_state = False

      for k, slot_start in enumerate(slot_start_indexes):
        seed_start = seed_start_indexes[k]
        slot_end = seed_start_indexes[k+1] if k < len(seed_start_indexes) -1 else len(seq)

        # To seed hidden state
        emb = seq_embed[seed_start:slot_start].view(1, slot_start-seed_start, -1)
        output, hidden = self.dec(emb, hidden) 
        output = output[:,-1].unsqueeze(1)

        if self.model_type == "alignment":
          alignment, viterbi_score = self.viterbi(
              (abs_seq, seq_t.view(1, -1)), a_last=False) 
          align_ind = alignment[-1]
          # Predict next alignment
          Dt = output.view(1, 1, self.hidden).expand(1, K, self.hidden)
          joint = torch.cat((E,Dt), 2)

          # All shift probabilities: 
          if self.use_param_transitions:
            shift_logit = self.shift(self.drop(F.relu(self.shift_hidden(joint))))
            prev_align_ind = align_ind
            while align_ind < (K - 1) and shift_logit[0, align_ind] > 0:
              if len(shift_logit[0, align_ind]) > 0:
                slot_nll += -F.logsigmoid(shift_logit[0, align_ind]).cpu().data
              align_ind += 1

            # Emit probability for the slot
            if len(shift_logit[0, align_ind]) > 0:
              slot_nll += -F.logsigmoid(-shift_logit[0, align_ind]).cpu().data
          else:
            slot_nll += -self.fixed_emit_p # don't shift

        # Generate gold number of words
        for j in range(slot_end - slot_start):
            if j > 0:
              emb = seq_embed[slot_start+j-1].view(1, 1, -1)
              output, hidden = self.dec(emb, hidden)
            output = output.squeeze(1)
            if self.model_type == "alignment":
              cur_alpha = E[:,align_ind].view(1, self.hidden)
              emit_state = torch.cat((cur_alpha, output), 1)
              if self.use_param_transitions:
                shift_logit = self.shift(self.drop(F.relu(self.shift_hidden(joint))))
                slot_nll += -F.logsigmoid(-shift_logit[0, align_ind]).cpu().data
              else:
                slot_nll += -self.fixed_emit_p

            elif self.model_type == "seq2seq_attention":
              query = self.attention_bilinear(output).view(1, 1, self.hidden)
              align = torch.bmm(query, Et).view(1, K)
              align = F.softmax(align, 1).view(1, 1, K)
              cur_alpha = torch.bmm(align, E).view(1, self.hidden)
              emit_state = torch.cat((output, cur_alpha), 1)
            else:
              emit_state = output

            logits = self.emit(self.drop(F.relu(self.emit_hidden(emit_state))))
            word_weights = logits.squeeze().data
            word_weights += seq_mask
            word_dist = F.log_softmax(word_weights, 0)
            word_ll = word_dist[seq[slot_start+j]].cpu().data
            slot_nll += -word_ll

            dist_list = list(-word_dist.cpu().data.numpy())
            dist_list.sort()
            rank = dist_list.index(-word_ll)
            if rank < 5:
              top_k_count += 1
            r_rank += (rank + 1)

        slot_count += 1
        slot_token_count += slot_end - slot_start

    surprisal = slot_nll / slot_token_count
    print("Surprisal: ", surprisal)
    mean_rank = r_rank / slot_token_count
    print("Mean Rank: ", mean_rank)
    top_k_prec = top_k_count / slot_token_count
    print("Top 5 precision: ", top_k_prec)

  
  def next_slot_prediction(self, candidate, seq, abs_seq, seq_embed, seed_start, 
                           slot_start, slot_end, E, Et):
    # First advance the candidate to the start of the next slot prediction
    for j in range(seed_start, slot_start):
      word_idx = int(seq[j])
      candidate.output_seq.append(word_idx)
      word = self.i2voc[word_idx]
      candidate.template_str += word + ' '
      candidate.template_str += '##' + str(k+1) + '## '

    orig_slot_str = ''
    for j in range(slot_start, slot_end):
      word_idx = int(seq[j])
      word = self.i2voc[word_idx]
      orig_slot_str += word + ' '

    dec_seq = torch.LongTensor(candidate.output_seq).cuda().view(1, -1)
    # To seed hidden state
    emb = seq_embed[seed_start:slot_start].view(1, slot_start-seed_start, -1)
    output, hidden = self.dec(emb, hidden) 
    output = output[:,-1].unsqueeze(1)

    if self.model_type == "alignment":
      alignment, viterbi_score = self.viterbi(
          (abs_seq, dec_seq), a_last=False) 
      align_ind = alignment[-1]

      # Predict next alignment
      if self.use_param_transitions: 
        # All shift emit probabilities: 
        Dt = output.view(1, 1, self.hidden).expand(1, K, self.hidden)
        joint = torch.cat((E,Dt), 2)
        shift_logit = self.shift(self.drop(F.relu(self.shift_hidden(joint))))
        prev_align_ind = align_ind
        while align_ind < (K - 1) and shift_logit[0, align_ind] > 0:
          if len(shift_logit[0, align_ind]) > 0:
            self.slot_nll += -F.logsigmoid(shift_logit[0, align_ind]).cpu().data
          align_ind += 1
        # Emit probability for the slot
        if len(shift_logit[0, align_ind]) > 0:
          self.slot_nll += -F.logsigmoid(-shift_logit[0, align_ind]).cpu().data
      else:
        slot_nll += -self.fixed_emit_p # don't shift

      slot_str = ''
      input = torch.from_numpy(np.array([0])).view(1, 1).cuda()
      # Generate (up to) gold number of words
      for j in range(slot_end - slot_start):
          if j > 0:
              emb = self.embeddings(input)
              output, hidden = self.dec(emb, hidden)
          output = output.squeeze(1)
          if self.model_type == "alignment":
            cur_alpha = E[:,align_ind].view(1, self.hidden)
            emit_state = torch.cat((cur_alpha, output), 1)
            if self.use_param_transitions:
              shift_logit = self.shift(self.drop(F.relu(self.shift_hidden(joint))))
              slot_nll += -F.logsigmoid(-shift_logit[0, align_ind]).cpu().data
            else:
              slot_nll += -self.fixed_emit_p

          elif self.model_type == "seq2seq_attention":
            query = self.attention_bilinear(output).view(1, 1, self.hidden)
            align = torch.bmm(query, Et).view(1, K)
            align = F.softmax(align, 1).view(1, 1, K)
            cur_alpha = torch.bmm(align, E).view(1, self.hidden)
            emit_state = torch.cat((output, cur_alpha), 1)
          else:
            emit_state = output

          logits = self.emit(self.drop(F.relu(self.emit_hidden(emit_state))))
          temperature = 0.5
          word_weights = logits.squeeze().data.div(temperature) 
          word_weights += seq_mask
          word_ll = F.log_softmax(word_weights, 0)
          word_weights = word_weights.exp().cpu()

          if sum(word_weights) > 0:
            word_idx = int(torch.multinomial(word_weights, 1)[0])
          else:
            word_idx = 0 # UNK
          input.data.fill_(word_idx)
          output_seq.append(word_idx)
          word = self.i2voc[word_idx]
          candidate.slot_nll += -word_ll[word_idx].cpu().data
          slot_str += word + ' '

      slot_count += 1
      slot_token_count += slot_end - slot_start
      if orig_slot_str.strip() == self.slot_str.strip():
        c_slot_count += 1
      candidate.slot_strs.append('##' + str(k+1) + '##\t' + orig_slot_str.strip()  
                       + '\t' + slot_str.strip())

    return candidate 


  def run_generate_beam_search(self, log_path, sset, seq_tokens, seq_abstracts, seq_tags,
          seq_slot_restrictions=None, slot_vocab=None):
    outf = log_path + ".generate." + sset + ".txt"
    write_all_samples = False

    if slot_vocab is not None:
      mask = np.ones(self.vocab_size)*-np.inf
      for w in slot_vocab:
        mask[w] = 0
    else:
      mask = np.zeros(self.vocab_size)
    num_samples = 5

    slot_count = 0
    slot_total_nll = 0
    slot_token_count = 0
    c_slot_count = 0

    # removed fixed_shift/emit_p

    with open(outf, 'w') as outf:
      for i, seq in tqdm.tqdm(enumerate(seq_tokens), ncols=80, disable=True):
        abs_seq = torch.from_numpy(seq_abstracts[i]).cuda().unsqueeze(0)
        abs_seq_embed = self.embeddings(abs_seq)

        seq_t = torch.from_numpy(seq).cuda()
        seq_embed = self.embeddings(seq_t)

        E, enc_final_hidden = self.enc(abs_seq_embed)
        if self.bidirectional:
          E = self.collapse_bider(E)
          enc_final_hidden = (torch.mean(enc_final_hidden[0], 0, keepdim=True), 
                              torch.mean(enc_final_hidden[1], 0, keepdim=True))
        K = abs_seq.size()[1]
        if (self.model_type == "seq2seq" or self.model_type == "seq2seq_attention"
              or (self.model_type == "alignment" and self.enc2dec)):
          Et = E.transpose(2, 1)
          hidden = enc_final_hidden
        else:
          hidden = self.init_hidden(1)
          Et = None 

        seed_start_indexes = []
        slot_start_indexes = []
        slot_state = False
        tags = seq_tags[i]
        assert tags[0] == 1

        abs_str = ' '.join(self.i2voc[int(s)] for s in seq_abstracts[i])
        if not write_all_samples:
          outf.write(abs_str + '\n')

        seq_mask = torch.Tensor(mask)
        if seq_slot_restrictions is not None:
          for w in seq_slot_restrictions[i]:
            seq_mask[w] = -np.inf
        seq_mask = seq_mask.cuda()

        for j, is_kept in enumerate(tags):
          if is_kept == 0 and not slot_state:
            slot_start_indexes.append(j)  
            slot_state = True
          elif j == 0 or is_kept == 1 and slot_state:
            seed_start_indexes.append(j)  
            slot_state = False

        candidates = []

        template_strs = []
        slots_strs = []
        template_scores = []
        output_seqs = []
        slot_nlls = []

        for l in range(num_samples):
            candidate = Candidate()
            for k, slot_start in enumerate(slot_start_indexes):
              seed_start = seed_start_indexes[k]
              slot_end = seed_start_indexes[k+1] if k < len(seed_start_indexes) -1 else len(seq)
              assert len(output_seq) == slot_start
              candidate = self.next_slot_prediction(candidate, seq, abs_seq, seq_embed,
                  seed_start, slot_end, E, Et)

            if len(seed_start_indexes) == len(slot_start_indexes) + 1:
              for j in range(seed_start_indexes[-1], len(seq)):
                word_idx = int(seq[j])
                candidate.output_seq.append(word_idx)
                word = self.i2voc[word_idx]
                candidate.template_str += word + ' '
            
            candidate.slot_nll = float(candidate.slot_nll)

            # Score sequence:
            dec_seq = torch.LongTensor(candidate.output_seq).cuda().view(1, -1)
            if self.model_type == "alignment":
              alignment, viterbi_score = self.viterbi((abs_seq, dec_seq)) 
              candidate.template_score = float(viterbi_score.cpu().data)
            else:
              D, _ = self.dec(self.embeddings(dec_seq))
              if self.model_type == "seq2seq" or self.model_type == "seq2seq_attention":
                hidden = enc_final_hidden
              else:
                hidden = self.init_hidden(1)

              alpha = torch.zeros(1).cuda()
              for t in range(1, len(seq)):                        
                Dt = D[:,t-1,:].view(1, self.hidden)
                if self.model_type == "seq2seq_attention":
                  query = self.attention_bilinear(Dt).view(1, 1, self.hidden)
                  align = torch.bmm(query, Et).view(1, K)
                  align = F.softmax(align, 1).view(1, 1, K)
                  cur_alpha = torch.bmm(align, E).view(1, self.hidden)
                  emit_state = torch.cat((Dt, cur_alpha), 1)
                else:
                  emit_state = Dt
                
                emissions = F.log_softmax(self.emit(self.drop(F.relu(self.emit_hidden(emit_state)))), 1)
                alpha += emissions[0, seq[t]]
              candidate.template_scores = float(alpha.cpu().data)
            cnadidates.append(candidate)

        ind = np.argmax([cand.template_score for cand in candidates])
        slot_total_nll += candidates[ind].slot_nll

        if write_all_samples:
          for cand in candidates:
            outf.write(abs_str + '\n')
            outf.write(cand.template_str.strip() + '\n')
            for slot in cand.slots_strs:
              outf.write(slot + '\n')
            outf.write('\n')
        else:
          outf.write(cand.template_str.strip() + '\n')
          for slot in cand.slots_strs:
            outf.write(slot + '\n')
          outf.write('\n')
    surprisal = slot_total_nll / slot_token_count
    print("Accuracy: ", c_slot_count/slot_count*100)
    print("Prediction surprisal: ", surprisal)


  def run_generate(self, log_path, sset, seq_tokens, seq_abstracts, seq_tags,
          seq_slot_restrictions=None, slot_vocab=None):
    outf = log_path + ".generate." + sset + ".txt"
    write_all_samples = False

    if slot_vocab is not None:
      mask = np.ones(self.vocab_size)*-np.inf
      for w in slot_vocab:
        mask[w] = 0
    else:
      mask = np.zeros(self.vocab_size)
    num_samples = 5

    slot_count = 0
    slot_total_nll = 0
    slot_token_count = 0
    c_slot_count = 0

    # removed fixed shift/emit_p

    with open(outf, 'w') as outf:
      print(len(seq_tokens))
      for i, seq in tqdm.tqdm(enumerate(seq_tokens), ncols=80, disable=True):
        abs_seq = torch.from_numpy(seq_abstracts[i]).cuda().unsqueeze(0)
        abs_seq_embed = self.embeddings(abs_seq)

        seq_t = torch.from_numpy(seq).cuda() # Variable
        seq_embed = self.embeddings(seq_t)

        E, enc_final_hidden = self.enc(abs_seq_embed)
        if self.bidirectional:
          E = self.collapse_bider(E)
          enc_final_hidden = (torch.mean(enc_final_hidden[0], 0, keepdim=True), 
                              torch.mean(enc_final_hidden[1], 0, keepdim=True))
        K = abs_seq.size()[1]
        if (self.model_type == "seq2seq" or self.model_type ==
            "seq2seq_attention"
              or (self.model_type == "alignment" and self.enc2dec)):
          Et = E.transpose(2, 1)
          hidden = enc_final_hidden
        else:
          hidden = self.init_hidden(1)

        seed_start_indexes = []
        slot_start_indexes = []
        slot_state = False
        tags = seq_tags[i]
        assert tags[0] == 1

        abs_str = ' '.join(self.i2voc[int(s)] for s in seq_abstracts[i])
        if not write_all_samples:
          outf.write(abs_str + '\n')

        seq_mask = torch.Tensor(mask)
        if seq_slot_restrictions is not None:
          for w in seq_slot_restrictions[i]:
            seq_mask[w] = -np.inf
        seq_mask = seq_mask.cuda()

        for j, is_kept in enumerate(tags):
          if is_kept == 0 and not slot_state:
            slot_start_indexes.append(j)  
            slot_state = True
          elif j == 0 or is_kept == 1 and slot_state:
            seed_start_indexes.append(j)  
            slot_state = False

        template_strs = []
        slots_strs = []
        template_scores = []
        output_seqs = []
        slot_nlls = []

        for l in range(num_samples):
            template_str = ''
            slot_strs = []
            output_seq = []
            slot_nll = 0
            for k, slot_start in enumerate(slot_start_indexes):
              seed_start = seed_start_indexes[k]
              for j in range(seed_start, slot_start):
                  word_idx = int(seq[j])
                  output_seq.append(word_idx)
                  word = self.i2voc[word_idx]
                  template_str += word + ' '
              template_str += '##' + str(k+1) + '## '
                  
              slot_end = seed_start_indexes[k+1] if k < len(seed_start_indexes) -1 else len(seq)
              orig_slot_str = ''
              for j in range(slot_start, slot_end):
                  word_idx = int(seq[j])
                  word = self.i2voc[word_idx]
                  orig_slot_str += word + ' '

              # Encode current generated decoder output
              assert len(output_seq) == slot_start
              dec_seq = torch.LongTensor(output_seq).cuda().view(1, -1)

              # To seed hidden state
              emb = seq_embed[seed_start:slot_start].view(1, slot_start-seed_start, -1)
              output, hidden = self.dec(emb, hidden) 
              output = output[:,-1].unsqueeze(1)

              if self.model_type == "alignment":
                alignment, viterbi_score = self.viterbi(
                    (abs_seq, dec_seq), a_last=False) 
                align_ind = alignment[-1]
                # Predict next alignment
                
                if self.use_param_transitions: 
                  # All shift emit probabilities: 
                  Dt = output.view(1, 1, self.hidden).expand(1, K, self.hidden)
                  joint = torch.cat((E,Dt), 2)
                  shift_logit = self.shift(self.drop(F.relu(self.shift_hidden(joint))))
                  prev_align_ind = align_ind
                  while align_ind < (K - 1) and shift_logit[0, align_ind] > 0:
                    if len(shift_logit[0, align_ind]) > 0:
                      slot_nll += -F.logsigmoid(shift_logit[0, align_ind]).cpu().data
                    align_ind += 1
                  # Emit probability for the slot
                  if len(shift_logit[0, align_ind]) > 0:
                    slot_nll += -F.logsigmoid(-shift_logit[0, align_ind]).cpu().data
                else:
                  slot_nll += -self.fixed_emit_p

              slot_str = ''
              input = torch.from_numpy(np.array([0])).view(1, 1).cuda() # Variable
              # Generate (up to) gold number of words
              for j in range(slot_end - slot_start):
                  if j > 0:
                      emb = self.embeddings(input)
                      output, hidden = self.dec(emb, hidden)
                  output = output.squeeze(1)
                  if self.model_type == "alignment":
                    cur_alpha = E[:,align_ind].view(1, self.hidden)
                    emit_state = torch.cat((cur_alpha, output), 1)
                    if self.use_param_transitions:
                      shift_logit = self.shift(self.drop(F.relu(self.shift_hidden(joint))))
                      slot_nll += -F.logsigmoid(-shift_logit[0, align_ind]).cpu().data
                    else:
                      slot_nll += -self.fixed_emit_p

                  elif self.model_type == "seq2seq_attention":
                    query = self.attention_bilinear(output).view(1, 1, self.hidden)
                    align = torch.bmm(query, Et).view(1, K)
                    align = F.softmax(align, 1).view(1, 1, K)
                    cur_alpha = torch.bmm(align, E).view(1, self.hidden)
                    emit_state = torch.cat((output, cur_alpha), 1)
                  else:
                    emit_state = output

                  logits = self.emit(self.drop(F.relu(self.emit_hidden(emit_state))))
                  temperature = 0.5
                  word_weights = logits.squeeze().data.div(temperature) 
                  word_weights += seq_mask
                  word_ll = F.log_softmax(word_weights, 0)
                  word_weights = word_weights.exp().cpu()

                  if sum(word_weights) > 0:
                    word_idx = int(torch.multinomial(word_weights, 1)[0])
                  else:
                    word_idx = 0 # UNK
                  input.data.fill_(word_idx)
                  output_seq.append(word_idx)
                  word = self.i2voc[word_idx]
                  slot_nll += -word_ll[word_idx].cpu().data
                  slot_str += word + ' '

              slot_count += 1
              slot_token_count += slot_end - slot_start
              if orig_slot_str.strip() == slot_str.strip():
                c_slot_count += 1
              slot_strs.append('##' + str(k+1) + '##\t' + orig_slot_str.strip()  
                               + '\t' + slot_str.strip())

            if len(seed_start_indexes) == len(slot_start_indexes) + 1:
              for j in range(seed_start_indexes[-1], len(seq)):
                word_idx = int(seq[j])
                output_seq.append(word_idx)
                word = self.i2voc[word_idx]
                template_str += word + ' '
            
            template_strs.append(template_str)
            slots_strs.append(slot_strs)
            slot_nlls.append(float(slot_nll))
            # Score sequence
            dec_seq = torch.LongTensor(output_seq).cuda().view(1, -1)
            if self.model_type == "alignment":
              alignment, viterbi_score = self.viterbi(
                    (abs_seq, dec_seq)) 
              template_scores.append(float(viterbi_score.cpu().data))
            else:
              D, _ = self.dec(self.embeddings(dec_seq))
              if self.model_type == "seq2seq" or self.model_type == "seq2seq_attention":
                hidden = enc_final_hidden
              else:
                hidden = self.init_hidden(1)

              alpha = torch.zeros(1).cuda()
              for t in range(1, len(seq)):                        
                Dt = D[:,t-1,:].view(1, self.hidden)
                if self.model_type == "seq2seq_attention":
                  query = self.attention_bilinear(Dt).view(1, 1, self.hidden)
                  align = torch.bmm(query, Et).view(1, K)
                  align = F.softmax(align, 1).view(1, 1, K)
                  cur_alpha = torch.bmm(align, E).view(1, self.hidden)
                  emit_state = torch.cat((Dt, cur_alpha), 1)
                else:
                  emit_state = Dt
                
                emissions = F.log_softmax(self.emit(self.drop(F.relu(self.emit_hidden(emit_state)))), 1)
                alpha += emissions[0, seq[t]]
              template_scores.append(float(alpha.cpu().data))

        ind = np.argmax(template_scores)
        slot_total_nll += slot_nlls[ind]

        if write_all_samples:
          for ind, score in enumerate(template_scores):
            outf.write(abs_str + '\n')
            outf.write(template_strs[ind].strip() + '\n')
            for slot in slots_strs[ind]:
              outf.write(slot + '\n')
            outf.write('\n')
        else:
          outf.write(template_strs[ind].strip() + '\n')
          for slot in slots_strs[ind]:
            outf.write(slot + '\n')
          outf.write('\n')
    surprisal = slot_total_nll / slot_token_count
    print("Accuracy: ", c_slot_count/slot_count*100)
    print("Prediction surprisal: ", surprisal)


  def run_evaluation(self, inps, outs):
    vals = []
    for i in tqdm.tqdm(range(0, len(inps)), ncols=80, disable=True):
      src = torch.from_numpy(inps[i]).cuda().unsqueeze(0) # Variable
      tgt = torch.from_numpy(outs[i]).cuda().unsqueeze(0)
      _, val = self.viterbi((src, tgt))
      vals.append(float(val.cpu().data))
    return vals

  def run_viterbi(self, d_ins, d_out):
    alignments = []
    for i in range(0, len(d_ins)):
      src = torch.from_numpy(d_ins[i]).cuda().unsqueeze(0) # Variable
      tgt = torch.from_numpy(d_out[i]).cuda().unsqueeze(0)
      alignment, _ = self.viterbi((src, tgt))
      alignments.append(alignment)
    return alignments
 

  def print_viterbi(self, d_ins, d_out, fname):
    o = open(fname, 'w')
    alignments = []
    def to_string(seq):
      """  Map ints to AA sequence  """
      return " ".join([self.i2voc[s] for s in seq]).replace("<PAD>","_")

    for i in tqdm.tqdm(range(0, len(d_ins)), ncols=80, disable=True):
      src = torch.from_numpy(d_ins[i]).cuda().unsqueeze(0) # Variable
      tgt = torch.from_numpy(d_out[i]).cuda().unsqueeze(0)
      alignment, _ = self.viterbi((src, tgt))
      alignments.append(alignment)
      o.write(to_string(d_ins[i]).strip() + "\n")
      o.write(to_string(d_out[i]).strip() + "\n")
      o.write(" ".join(["{}".format(v) for v in alignment]) + "\n\n")
    print("Completed training Viterbi decoding.")
    return alignments
 

  def hard_forward(self, vals):
    assert len(vals) == 3
    x, y, align = vals
    x = self.embeddings(x)
    dec_words = y.clone()
    y = self.embeddings(y)

    assert self.model_type == "alignment"

    E, enc_final_hidden = self.enc(x)
    if self.bidirectional:
      E = self.collapse_bider(E)
      enc_final_hidden = (torch.mean(enc_final_hidden[0], 0, keepdim=True), 
                          torch.mean(enc_final_hidden[1], 0, keepdim=True))
    if self.enc2dec:
      D, _ = self.dec(y, enc_final_hidden) 
    else:
      D, _ = self.dec(y)

    # Forward Pass
    N = x.size()[0] # batch size
    T = y.size()[1] # dec_max_length
    K = x.size()[1] # enc_max_length
    V = self.vocab_size

    # Target Functionality
    # for j in range(1, J):
    #   for i in range(0, I):
    #     alpha(i,j) = p(d_j | h_e_i, h_d_j) x
    #                  sum_k_0_i alpha(k, j-1) * p(a_j = i | a_j-1 = k)

    fixed_transitions = self.tran[self.enc_max_length-K:,:K].view(K, K) # submatrix of right diagonal matrix

    pre_alpha = Variable(torch.zeros(N)).cuda()
    cur_alpha = Variable(torch.zeros(N)).cuda()

    for t in range(1, T):                        
      Dt = D[:,t-1,:].unsqueeze(1).expand(N, K, self.hidden)
      joint = torch.cat((E, Dt), 2)

      # only compute for the current alignment value
      cur_alpha = pre_alpha #.unsqueeze(2).expand(N, K, K) #TODO check
      
      # Transition
      transitions = Variable(torch.zeros(N, dtype=torch.float)).cuda()
      if self.use_param_transitions:
        shift_logit = self.shift(self.drop(F.relu(self.shift_hidden(joint))))
        shift_p = F.logsigmoid(shift_logit).squeeze(2)
        emit_p = F.logsigmoid(-shift_logit).view(N, K) #.expand(N, K, K)

        # Shift from align[:,t-1] to align[:,t] 
        for n in range(N): 
          transitions[n] = torch.sum(shift_p[n,align[n,t-1].data:align[n,t].data]) 
        transitions += emit_p.gather(1, align[:,t].view(-1, 1)).view(-1)
      else:
        for n in range(N): 
          # Shift from align[n, t-1] to align[n,t] -> j to (K-i-1)
          transitions[n] = fixed_transitions[K-align[n,t]-1,align[n, t-1]] 
              
      cur_alpha += transitions

      # Emission
      emissions = F.log_softmax(self.emit(self.drop(F.relu(self.emit_hidden(joint)))), -1)
      word_idx = dec_words[:, t].unsqueeze(1).expand(N,K).unsqueeze(2)
      emission_alpha = emissions.gather(2, word_idx).view(N, K)
      cur_alpha += emission_alpha.gather(1, align[:,t].view(-1, 1)).view(-1)
      
      # Update
      pre_alpha = cur_alpha.clone()

    alpha = cur_alpha
    loss = -1 * alpha
    return loss 


  def forward(self, vals):
    assert len(vals) == 2
    x, y = vals
    x = self.embeddings(x)
    dec_words = y.clone()
    y = self.embeddings(y)

    E, enc_final_hidden = self.enc(x)
    if self.bidirectional:
      E = self.collapse_bider(E)
      enc_final_hidden = (torch.mean(enc_final_hidden[0], 0, keepdim=True), 
                          torch.mean(enc_final_hidden[1], 0, keepdim=True))
    if (self.model_type == "seq2seq" or self.model_type == "seq2seq_attention"
            or (self.model_type == "alignment" and self.enc2dec)):
      Et = E.transpose(2, 1)
      D, _ = self.dec(y, enc_final_hidden) 
    else:
      D, _ = self.dec(y)

    # Forward Pass
    N = x.size()[0] # batch size
    T = y.size()[1] # dec_max_length
    K = x.size()[1] # enc_max_length
    V = self.vocab_size

    if self.model_type == "alignment":
      # Target Functionality
      # for j in range(1, J):
      #   for i in range(0, I):
      #     alpha(i,j) = p(d_j | h_e_i, h_d_j) x
      #                  sum_k_0_i alpha(k, j-1) * p(a_j = i | a_j-1 = k)

      pre_alpha = Variable(torch.zeros(N, K)).cuda()
      cur_alpha = Variable(torch.zeros(N, K)).cuda()

      fixed_transitions = self.tran[self.enc_max_length-K:,:K].unsqueeze(0) # submatrix of right diagonal matrix
      mask = self.mask[self.enc_max_length-K:,:K].unsqueeze(0)

      for t in range(1, T):                        
        cur_alpha = pre_alpha.unsqueeze(2).expand(N, K, K) 
        Dt = D[:,t-1,:].unsqueeze(1).expand(N, K, self.hidden)
        joint = torch.cat((E, Dt), 2)
        
        # Transition
        if self.use_param_transitions:
          shift_logit = self.shift(self.drop(F.relu(self.shift_hidden(joint))))
          shift_p = F.logsigmoid(shift_logit).squeeze(2)
          emit_p = F.logsigmoid(-shift_logit).expand(N, K, K)

          # Unvectorized
          #for i in range(K):
          #  for j in range(K-i):
          #    # Shifting from j to (K-i-1)
          #    transitions[:,i,j] = torch.sum(shift_p[:,j:K-i-1].view(N, -1), 1)
   
          # Magic to make more efficient 
          transitions = Variable(torch.zeros(N, K, K, dtype=torch.float)).cuda()
          for k in range(K-1):
            transitions[:,:K-k-1,:k+1] += shift_p[:,k].view(N, 1, 1).expand(N, K-k-1, k+1)
    
          transitions += emit_p
          cur_alpha += transitions
        else:
          cur_alpha += fixed_transitions

        cur_alpha = cur_alpha + mask             # Zero-out bottom-right
        cur_alpha = torch.logsumexp(cur_alpha, 1)         # Sum columns

        # This overcounts by # of zeros, so we subtract them
        cur_alpha = self.flip(cur_alpha)   # reorder alphas

        # Emission
        emissions = F.log_softmax(self.emit(self.drop(F.relu(self.emit_hidden(joint)))), -1)
        word_idx = dec_words[:, t].unsqueeze(1).expand(N,K).unsqueeze(2)
        emission_alpha = emissions.gather(2, word_idx).squeeze()

        cur_alpha = cur_alpha + emission_alpha

        # Update
        pre_alpha = cur_alpha.clone()

      if self.align_last:
        alpha = cur_alpha[:,K-1] 
      else:
        alpha = torch.logsumexp(cur_alpha, dim=1)

    elif self.model_type == "unconditional" or self.model_type == "seq2seq":
      alpha = Variable(torch.zeros(N)).cuda()
      for t in range(1, T):                        
        Dt = D[:,t-1,:]
        emissions = F.log_softmax(self.emit(self.drop(F.relu(self.emit_hidden(Dt)))), 1)
        word_idx = dec_words[:, t].unsqueeze(1)
        emission_alpha = emissions.gather(1, word_idx).view(-1)
        alpha += emission_alpha

    elif self.model_type == "seq2seq_attention":
      # Attention is not fed.
      alpha = Variable(torch.zeros(N)).cuda()
      for t in range(1, T): 
        Dt = D[:,t-1,:].view(N, self.hidden)
        query = self.attention_bilinear(Dt).view(N, 1, self.hidden)
        # dim [batch_size, 1, hidden] x [batch_size, hidden, enc_len] -> [batch_size, 1, enc_len]
        align = torch.bmm(query, Et).view(N, K)
        align = F.softmax(align, 1).view(N, 1, K)
        # dim [batch_size, 1, enc_len] x [batch_size, enc_len, hidden] -> [batch_size, 1, hidden]
        cur_alpha = torch.bmm(align, E).view(N, self.hidden)

        joint = torch.cat((Dt, cur_alpha), 1)
        emissions = F.log_softmax(self.emit(self.drop(F.relu(self.emit_hidden(joint)))), 1)
        word_idx = dec_words[:, t].unsqueeze(1)
        emission_alpha = emissions.gather(1, word_idx).view(-1)
        alpha += emission_alpha

    loss = -1 * alpha
    return loss 


