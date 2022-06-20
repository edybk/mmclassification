import json

resnet_log_jsons= "/data/home/bedward/workspace/mmpose-project/mmclassification/work_dirs/resnet50_8xb32_in1k_apas/20220618_022257.log.json"
convnextlrgejsons="/data/home/bedward/workspace/mmpose-project/mmclassification/work_dirs/convnext-large_64xb64_in1k_apas/20220618_032742.log.json"
def get_best(jsonsss):
    acc = []
    with open(jsonsss) as f:
        for l in f.readlines():
            d = json.loads(l)
            if "mode" in d and d['mode']=='val':
                acc.append((d.get("accuracy_top-1", 0), d.get("epoch", "")))
    print(max(acc))
    return max(acc)
    

get_best(convnextlrgejsons)