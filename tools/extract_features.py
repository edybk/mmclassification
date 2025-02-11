# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import numpy as np
import torch
from tqdm import tqdm
import mmcv
from mmcls.apis.inference import init_model

from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmcls.datasets.pipelines import Compose
from mmcls.models import build_classifier



def extract_features(model, img, stage='backbone'):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    def process_img(im):
        
        # build the data pipeline
        if isinstance(im, str):
            if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
                cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
            data = dict(img_info=dict(filename=im), img_prefix=None)
        else:
            if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
                cfg.data.test.pipeline.pop(0)
            data = dict(img=im)
        
        test_pipeline = Compose(cfg.data.test.pipeline)
        data = test_pipeline(data)
        return data
    
    if isinstance(img, list):
        datas = []
        for im in img:
            datas.append(process_img(im))
    else:
        data = process_img(img)
        datas = [data]
    
    data = collate(datas, samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    result = []
    # forward the model
    with torch.no_grad():
        (backbone_features, ) = model.extract_feat(data['img'], stage=stage)
        result.append(backbone_features.cpu())
        
        
    return np.stack(result, axis=0)


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
    stage = 'neck' #'backbone'  
    batch_size = 32 #1 #256
    sample_rate = 30 #1
    feats = []
    
    frames_subset =  frames[::sample_rate] #frames[:100] #
    print(len(frames_subset))
    num_batches = len(frames_subset) / batch_size
    print(num_batches)
    print(frames[:3])
    total_len = 0
    
    if stage == 'backbone':
        for i, batch in enumerate(chunker(frames_subset, batch_size)):
            print((i+1)/num_batches)
            batch_feats = extract_features(model, batch, stage=stage).squeeze(0)
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
    if stage == 'neck':
        for i, batch in enumerate(chunker(frames_subset, batch_size)):
            print((i+1)/num_batches)
            batch_feats = extract_features(model, batch, stage=stage).squeeze(0)
            # print(batch_feats.shape)
            total_len += len(batch_feats)
            if len(feats) > 1 and len(batch_feats) != len(feats[-1]):
                new_batch_feats = np.zeros(feats[-1].shape)
                new_batch_feats[:batch_feats.shape[0], :] = batch_feats
                batch_feats = new_batch_feats
                
            # print(f"final {batch_feats.shape}")

            feats.append(batch_feats)
            # break
        
        output = np.stack(feats).reshape((-1, feats[0].shape[1]))[:total_len, :]
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
        if stage == 'backbone':
            vid_arr = output[start_end["start"]:start_end["end"], :, :, :]
        if stage == 'neck':
            vid_arr = output[start_end["start"]:start_end["end"], :]
        
        final_arr = np.repeat(vid_arr, sample_rate, axis=0).transpose()
        print(final_arr.shape)
        np.save(f"{out}/{vid}.npy", final_arr)
        print(f'saved {out}/{vid}.npy')
    

if __name__ == '__main__':
    main()
