# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:52:07 2020

@author: 14048
"""
from itertools import groupby
import os

def convert_and_clean(inp, vocab):
    temp = []
    for x in inp:
        if vocab[x] == '<unk>':
            t = '\<unk\>'
            temp.append(t)
        else:
            temp.append(vocab[x])
    temp = '\ '.join(temp)
    cmd = "sh ./evaluation_relaxation/clean.sh {:s}".format(temp)
    f = os.popen(cmd, 'r')
    cleaned = f.read()
    f.close()
    # print(cleaned)
    cleaned = cleaned.strip().split(' ')
    res = []
    for x in cleaned:
        try:
            res.append(vocab.index(x))
        except:
            continue
    return res


def get_wer_delsubins(ref, hyp, debug=False, vocab=None, save_dir=None, id=None):
    if debug:
        assert vocab is not None
        assert save_dir is not None
        assert id is not None
        ref, hyp = convert_and_clean(ref, vocab), convert_and_clean(hyp, vocab)
    
    DEL_PENALTY = 1
    SUB_PENALTY = 1
    INS_PENALTY = 1
    r = ref
    h = hyp
    # costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h) + 1)] for outer in range(len(r) + 1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r) + 1):
        costs[i][0] = DEL_PENALTY * i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i - 1][j - 1] + SUB_PENALTY  # penalty is always 1
                insertionCost = costs[i][j - 1] + INS_PENALTY  # penalty is always 1
                deletionCost = costs[i - 1][j] + DEL_PENALTY  # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        # print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("OK " + vocab[r[i]] + " " + vocab[h[j]])
        elif backtrace[i][j] == OP_SUB:
            numSub += 1
            i -= 1
            j -= 1
            if debug:
                lines.append("SUB " + vocab[r[i]] + " " + vocab[h[j]])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j -= 1
            if debug:
                lines.append("INS " + "****" + " " + vocab[h[j]])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i -= 1
            if debug:
                lines.append("DEL " + vocab[r[i]] + " " + "****")
    if debug and numSub + numDel + numIns > 0:
        with open(save_dir+'/badcase.txt', 'a+') as f:
            f.write(id + '\n')
            f.write(' '.join([vocab[x] for x in ref]) + '\n')
            f.write(' '.join([vocab[x] for x in hyp]) + '\n')
            # f.write("OP\tREF\tHYP\n")
            lines = reversed(lines)
            for line in lines:
                if 'OK' in line:
                    continue
                f.write(line+'\n')
            f.write('\n')
            # print("#cor " + str(numCor))
            # print("#sub " + str(numSub))
            # print("#del " + str(numDel))
            # print("#ins " + str(numIns))
    return (numSub + numDel + numIns) / (float)(len(r)), numSub / float(len(r)), numIns / float(len(r)), numDel / float(len(r))


# if __name__ == '__main__':
#     ref = [1,2,3,4,5]
#     hyp = [2,3,4]
#     print(get_wer_delsubins(ref, hyp))