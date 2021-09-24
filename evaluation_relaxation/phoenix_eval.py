import os

def get_phoenix_wer(hyp, phase, tmp_prefix, shell_dir='evaluation_relaxation', dataset='2014'):
    if dataset == '2014':
        shell_file = os.path.join(shell_dir, 'phoenix_eval.sh')
    elif dataset == '2014T':
        shell_file = os.path.join(shell_dir, 'phoenix_T_eval.sh')
    elif dataset == '2014SI':
        shell_file = os.path.join(shell_dir, 'phoenix_SI_eval.sh')
    elif 'csl' in dataset:
        return [0., 0., 0., 0.]
    cmd = "sh {:s} {:s} {:s} {:s}".format(shell_file, hyp, phase, tmp_prefix)
    os.system(cmd)
    result_file = os.path.join(shell_dir, '{:s}.tmp.out.{:s}.sys'.format(tmp_prefix, hyp))

    with open(result_file, 'r') as fid:
        for line in fid:
            line = line.strip()
            if 'Sum/Avg' in line:
                result = line
                break
    tmp_err = result.split('|')[3].split()
    subs, inse, dele, wer = tmp_err[1], tmp_err[3], tmp_err[2], tmp_err[4]
    subs, inse, dele, wer = float(subs), float(inse), float(dele), float(wer)
    errs = [wer, subs, inse, dele]
    os.system('rm {:s}'.format(os.path.join(shell_dir, '{:s}.tmp.*'.format(tmp_prefix))))
    os.system('rm {:s}'.format(os.path.join(shell_dir, hyp)))
    return errs
