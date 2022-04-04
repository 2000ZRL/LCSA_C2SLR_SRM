import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import KLDivLoss

class SeqKD(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, T=1, blank_id=1232):
        super(SeqKD, self).__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T
        self.blank_id = blank_id

    def forward(self, prediction_logits, ref_logits):
        prediction_logits = F.log_softmax(prediction_logits[:, :, :self.blank_id]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - 1)
        ref_probs = F.softmax(ref_logits[:, :, :self.blank_id]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - 1)
        loss = self.kdloss(prediction_logits, ref_probs)*self.T*self.T
        # comments which are the same as original repo
        # mask_probs = F.softmax(ref_logits[:, :, 1:], dim=-1).view(-1, ref_logits.shape[2] - 1)
        # mask = torch.max(mask_probs, dim=1)[0] > 0.5
        # if torch.sum(mask) != 0:
        #     loss = torch.sum(torch.sum(loss, dim=1) * mask) / torch.sum(mask)
        # else:
        #     loss = torch.sum(torch.sum(loss, dim=1) * mask)
        return loss


class SeqJSD(nn.Module):
    def __init__(self, T=1, blank_id=1232, setting=None):
        super(SeqJSD, self).__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T
        self.blank_id = blank_id
        self.setting = setting  #confusion loss

    def forward(self, prediction_logits, ref_logits):
        if self.setting == 'conf':
            # confusion loss
            prediction_logits = F.log_softmax(prediction_logits[:, :, :self.blank_id]/self.T, dim=-1).view(-1, ref_logits.shape[2] - 1)
            ref_probs = t.ones_like(ref_logits).cuda()
            ref_probs = F.softmax(ref_probs[:, :, :self.blank_id]/self.T, dim=-1).view(-1, ref_probs.shape[2] - 1)
            loss = self.kdloss(prediction_logits, ref_probs)*self.T*self.T

        else:
            target_shape = ref_logits.shape[2] - 1
            if self.blank_id > prediction_logits.shape[-1]:
                target_shape += 1
            
            prediction_probs = F.softmax(prediction_logits[:, :, :self.blank_id]/self.T, dim=-1) \
                .view(-1, target_shape)
            ref_probs = F.softmax(ref_logits[:, :, :self.blank_id]/self.T, dim=-1) \
                .view(-1, target_shape)
            log_mean = ((prediction_probs + ref_probs)/2).log()
            loss = (self.kdloss(log_mean, prediction_probs) + self.kdloss(log_mean, ref_probs)) / 2 * self.T * self.T
        
        return loss


class ConfLoss(nn.Module):
    def __init__(self):
        super(ConfLoss, self).__init__()
    
    def forward(self, x):
        B, C = x.shape
        x = F.log_softmax(x, dim=-1)
        x = x.sum(dim=-1) / C
        x = x.mean()
        return x


class LabelSmoothCE(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothCE, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with t.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / (num_classes-1)
            lb_one_hot = t.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -t.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.0, weight=None):
        super(FocalLoss, self).__init__()
        # pre-defined weight
        self.weight = weight
        if weight is not None:
            assert weight.ndim == 1
            self.weight = weight / weight.sum()
        self.gamma = gamma
    
    def forward(self, logits, label):
        # logits [N,C], label [N]
        ce_loss = F.cross_entropy(logits, label, reduction='none')  #[N]
        p = t.exp(-ce_loss)
        if self.weight is None:
            alpha = 1.0
        else:
            alpha = self.weight.index_select(0, label)  #[N]
        loss = (alpha * (1-p)**self.gamma * ce_loss).mean()
        return loss
