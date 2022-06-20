import subprocess
import json
import os
import numpy as np


def get_best(jsonsss):
    acc = []
    with open(jsonsss) as f:
        for l in f.readlines():
            d = json.loads(l)
            if "mode" in d and d['mode']=='val':
                acc.append((d.get("accuracy_top-1", 0), d.get("epoch", "")))
    print(max(acc))
    return max(acc)

def get_best_checkpoint(work_dir):
    acc = []
    for j in os.listdir(work_dir):
        if not j.endswith(".json"):
            continue
        try:
            acc.append(get_best(f"{work_dir}/{j}"))
        except:
            pass
    acc, epoch = max(acc)
    return f"{work_dir}/epoch_{epoch}.pth", acc

def extract(model_type = "resnet"):
    if model_type == "resnet":
        model_name = "resnet50_8xb32_in1k_apas"
    else:
        raise 'NOT IMPLEMENTED'
        
        
    accs = []
    for split in range(1, 6):
        work_dir = f"./work_dirs/{model_name}_split{split}"
        ckpt, acc = get_best_checkpoint(work_dir)
        accs.append(acc)
        print(ckpt)
        cmd = f"python tools/extract_features.py /data/home/bedward/workspace/mmpose-project/mmclassification/configs/resnet/{model_name}.py {ckpt} --frames-dir data/apas/rawframes --out-dir data/apas_classification_{model_name}_split{split}"
        print(cmd)
        # out = subprocess.check_output(cmd, shell=True)
        # print(out)
        # break
        # break
    print(f"mean validation: {np.mean(accs)}")
    
extract(model_type = "resnet")