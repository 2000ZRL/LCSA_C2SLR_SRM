# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 21:48:57 2020

Stochatic Gradient Stopping, SGS
@author: Zhe NIU
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


def sample_mask(xl, p):
    mask = []
    for l in xl:
        idx = torch.randperm(l)
        idx = idx[: int(l * p)]
        m = torch.zeros(l).bool()
        m[idx] = True
        mask.append(m)
    return torch.cat(mask, dim=0)


def create_sgs_applier(p_detach, lengths):
    detached = sample_mask(lengths, p_detach)
    attached = ~detached

    def sgs_apply(module, *data):
        n = len(data[0])  #number of frames

        attaching = attached.any()
        detaching = detached.any()
        # print('att sum: ', attached.sum())
        # print('det sum: ', detached.sum())

        assert attaching or detaching

        if attaching:
            attached_output = module(*[d[attached] if d is not None else None for d in data])

        if detaching:
            with torch.no_grad():
                detached_output = module(*[d[detached] if d is not None else None for d in data])

        if attaching:
            slot = torch.empty(
                n, *attached_output.shape[1:], dtype=attached_output.dtype
            )
        else:
            slot = torch.empty(
                n, *detached_output.shape[1:], dtype=detached_output.dtype
            )

        slot = slot.to(data[0].device)

        if attaching:
            slot[attached] = attached_output

        if detaching:
            slot[detached] = detached_output

        return slot

    return sgs_apply


# if __name__ == '__main__':
#     sgs_apply = create_sgs_applier(0.5, [512])
#     model = resnet18(False)
#     model.fc = nn.Identity()
#     model = model.cuda()
#     x = torch.randn(512, 3, 224, 224).cuda()
#     output = sgs_apply(model, x)
#     output.sum().backward()