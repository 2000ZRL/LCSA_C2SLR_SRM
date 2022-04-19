import torch


def gen_gaussian_hmap_hrnet_wb(coords, raw_size=(260,210), new_size=(64,64), gamma=14, flags=[True, False, False, False]):
    # HRNet pretrained on COCO-wholebody. Detect person then estimate pose
    # assume new_size_h = new_size_w. easy to implememnt
    # gamma: hyper-param, control the width of gaussian, larger gamma, SMALLER gaussian
    # flags: pose, face except for mouth, mouth, hands

    coords = torch.tensor(coords).float().permute(2,1,0)  #[3,133,T]
    coords = coords[:-1]  #[2,133,T]
    assert coords.shape[1] == 133

    # https://github.com/jin-s13/COCO-WholeBody
    idx_pose = [i for i in range(11)]  #upper body only
    idx_face_others = [i for i in range(23, 71)]  # face except for mouth
    idx_mouth = [i for i in range(71, 91)]
    idx_hands = [i for i in range(91, 133)]

    # debug
    idx = []
    idx.extend(idx_pose)
    idx.extend(idx_face_others)
    idx.extend(idx_mouth)
    idx.extend(idx_hands)
    assert len(idx) == 133-6-6

    idx = []
    if flags[0]:
        idx.extend(idx_pose)
    if flags[1]:
        idx.extend(idx_face_others)
    if flags[2]:
        idx.extend(idx_mouth)
    if flags[3]:
        idx.extend(idx_hands)
    idx = sorted(idx)
    idx = torch.tensor(idx)

    coords = coords.index_select(1, idx)  #[2,C,T]
    hmap_num, T = coords.shape[1:]
    raw_h, raw_w = raw_size
    new_h, new_w = new_size

    # generate 2d coords
    # NOTE: openpose generate opencv-style coordinates!
    coords_x =  coords[1] / (raw_h-1)
    coords_y = coords[0] / (raw_w-1)
    coords = torch.stack([coords_x, coords_y], dim=0)  #[2,C,T]

    # generate gaussian hmap
    sigma = new_h/gamma
    x, y = torch.meshgrid(torch.arange(new_h), torch.arange(new_w))
    grid = torch.stack([x,y], dim=0)  #[2,H,W]
    grid = grid.unsqueeze(0).unsqueeze(0).expand(hmap_num,T,-1,-1,-1)  #[C,T,2,H,W]
    coords = coords.unsqueeze(0).unsqueeze(0).expand(new_h,new_w,-1,-1,-1).permute(3,4,2,0,1)  #[C,T,2,H,W]
    hmap = torch.exp(-((grid-coords*(new_h-1))**2).sum(dim=2) / (2*sigma**2))  #[C,T,H,W]
    hmap = hmap.permute(1,0,2,3)  #[T,C,H,W]
    return hmap