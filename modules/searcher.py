# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 17:00:50 2021

@author: 14048
Beam searcher
"""

import torch
import numpy as np
import math
import collections


class CTC_decoder(object):
    #beam searcher
    def __init__(self, vocab, blank_id, beam_width=10, top_n=50, lm=None, lm_weight=0.0, length_pen=1.0):
        self.vocab = vocab
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.top_n = top_n
        self.lm = lm
        self.lm_weight = lm_weight  #0.3 is good
        self.length_pen = length_pen
    
    def make_new_beam(self):
        fn = lambda: ((-np.inf, -np.inf), tuple())
        return collections.defaultdict(fn)

    def logsumexp(self, *args):
        if all(a == -np.inf for a in args):
            return -np.inf
        a_max = max(args)
        lsp = math.log(sum(math.exp(a - a_max)
                           for a in args))
        return a_max + lsp
    
    def decode(self, probs, len_T):
        T, S = probs.shape

        # 求概率的对数
        probs = np.log(probs)
    
        # Elements in the beam are (prefix, (p_blank, p_no_blank))
        # Initialize the beam with the empty sequence, a probability of
        # 1 for ending in blank and zero for ending in non-blank
        # (in log space).
        # Always keep beam_eidth paths
        beam = [(tuple(), ((0.0, -np.inf), tuple()))]
        
        for t in range(len_T):  # Loop over time
            # A default dictionary to store the next step candidates.
            next_beam = self.make_new_beam()
    
            for s in np.argsort(probs[t], axis=-1)[-self.top_n:]:  # Loop over vocab
                # print(s)
                p = probs[t, s]
    
                # The variables p_b and p_nb are respectively the
                # probabilities for the prefix given that it ends in a
                # blank and does not end in a blank at this time step.
                for prefix, ((p_b, p_nb), prefix_p) in beam:  # Loop over beam
                    # If we propose a blank the prefix doesn't change.
                    # Only the probability of ending in blank gets updated.
    
                    if s == self.blank_id:
                        # 增加的字母是blank
                        # 先取出对应prefix的两个概率，然后更后缀为blank的概率n_p_b
                        (n_p_b, n_p_nb), _ = next_beam[prefix]  # -inf, -inf
                        n_p_b = self.logsumexp(n_p_b, p_b + p, p_nb + p)  # 更新后缀为blank的概率
                        next_beam[prefix] = ((n_p_b, n_p_nb), prefix_p)  # s=blank， prefix不更新，因为blank要去掉的。
                        # print(next_beam[prefix])
                        continue
    
                    # Extend the prefix by the new character s and add it to
                    # the beam. Only the probability of not ending in blank
                    # gets updated.
                    end_t = prefix[-1] if prefix else None
                    n_prefix = prefix + (s,)  # 更新 prefix, 它是一个tuple
                    n_prefix_p = prefix_p + (p,)
                    # 先取出对应 n_prefix 的两个概率, 这个是更新了blank概率之后的 new 概率
                    (n_p_b, n_p_nb), _ = next_beam[n_prefix]  # -inf, -inf
                    
                    #lm log probability
                    if self.lm is not None:
                        sen= ['<s>']
                        for x in n_prefix:
                            sen.append(self.vocab[x])
                        lm_lp = self.lm_weight * np.log(self.lm.p(sen))
                    else:
                        lm_lp = -np.inf
                    
                    if s != end_t:
                        # 如果s不和上一个重复，则更新非空格的概率
                        n_p_nb = self.logsumexp(n_p_nb, p_b + p + lm_lp, p_nb + p + lm_lp)
                    else:
                        # 如果s和上一个重复，也要更新非空格的概率
                        # We don't include the previous probability of not ending
                        # in blank (p_nb) if s is repeated at the end. The CTC
                        # algorithm merges characters not separated by a blank.
                        n_p_nb = self.logsumexp(n_p_nb, p_b + p + lm_lp)
                        next_beam[n_prefix] = ((n_p_b, n_p_nb), n_prefix_p)
    
                    # If s is repeated at the end we also update the unchanged
                    # prefix. This is the merging case.
                    if s == end_t:
                        (n_p_b, n_p_nb), n_prefix_p = next_beam[prefix]
                        n_p_nb = self.logsumexp(n_p_nb, p_nb + p)
                        # 如果是s=end_t，则prefix不更新
                        next_beam[prefix] = ((n_p_b, n_p_nb), n_prefix_p)
                    else:
                        # *NB* this would be a good place to include an LM score.
                        next_beam[n_prefix] = ((n_p_b, n_p_nb), n_prefix_p)
            # print(t, next_beam.keys())
            # Sort and trim the beam before moving on to the
            # next time-step.
            # 根据概率进行排序，每次保留概率最高的beam_size条路径
            beam = sorted(next_beam.items(),
                          key=lambda x: self.logsumexp(*x[1][0]) + len(x[0])*np.log(self.length_pen),
                          reverse=True)
            beam = beam[:self.beam_width]
    
        # best = beam[0]
        # return best[0], -logsumexp(*best[1][0]), best[1][1]
    
        pred_lens = [len(beam[i][0]) for i in range(self.beam_width)]
        max_len = max(pred_lens)
        pred_seq, scores, pred_pobs = np.zeros((self.beam_width, max_len), dtype=np.int32), \
                                      [], np.zeros((self.beam_width, max_len))
        for bs in range(self.beam_width):
            pred_seq[bs][:pred_lens[bs]] = beam[bs][0]
            scores.append(-self.logsumexp(*beam[bs][1][0]))
            # pred_pobs[bs][:pred_lens[bs]] = np.exp(beam[bs][1][1])
        return pred_seq, scores, None, pred_lens