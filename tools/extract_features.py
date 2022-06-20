# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import numpy as np
import torch
from tqdm import tqdm
import mmcv
from mmcls.apis.inference import extract_features, init_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--frames-dir', help='frames root dir')
    parser.add_argument('--out-dir', help='output features root dir')
    args = parser.parse_args()
    return args

def load_frames(dir):

    all_frames = list(sorted(glob.glob(f'{dir}/*/*img*')))
    return all_frames

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_vid_from_frame(frm):
    return frm.split('/')[-2]

def main():
    args = parse_args()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    cfg = mmcv.Config.fromfile(args.config)
    model = init_model(cfg, checkpoint=args.checkpoint, device='cuda:0')

    frames = load_frames("data/apas/rawframes")

    batch_size = 32 #1 #256
    sample_rate = 30 #1
    feats = []
    
    frames_subset = frames[::sample_rate] # frames[:100]#
    print(len(frames_subset))
    num_batches = len(frames_subset) / batch_size
    print(num_batches)
    print(frames[:3])
    total_len = 0
    
    for i, batch in enumerate(chunker(frames_subset, batch_size)):
        print((i+1)/num_batches)
        batch_feats = extract_features(model, batch).squeeze(0)
        # print(batch_feats.shape)
        total_len += len(batch_feats)
        if len(feats) > 1 and len(batch_feats) != len(feats[-1]):
            new_batch_feats = np.zeros(feats[-1].shape)
            new_batch_feats[:batch_feats.shape[0], :, :, :] = batch_feats
            batch_feats = new_batch_feats
            
        # print(f"final {batch_feats.shape}")

        feats.append(batch_feats)
        # break
    
        
        
    output = np.stack(feats).reshape((-1, feats[0].shape[1], feats[0].shape[2], feats[0].shape[3]))[:total_len, :, :, :]
    print(output.shape)
    
    vid_start_end = {}
    curr_video_start = 0
    while curr_video_start < len(frames_subset):
        frame = frames_subset[curr_video_start]
        vid=get_vid_from_frame(frame)
        
        def get_vid_end():
            j = curr_video_start
            while j < len(frames_subset) and vid == get_vid_from_frame(frames_subset[j]):
                # print(vid)
                # print(frames_subset[j])
                # print(j)
                j += 1
            return j
        
        vid_end = get_vid_end()
        
        vid_start_end[vid] = {"start": curr_video_start, "end": vid_end}
        curr_video_start = vid_end
        
    # vid_to_array = {}
    for vid, start_end in vid_start_end.items():
        vid_arr = output[start_end["start"]:start_end["end"], :, :, :]
        np.save(f"{out}/{vid}.npy", vid_arr)
        print(f'saved {out}/{vid}.npy')
    

if __name__ == '__main__':
    main()
