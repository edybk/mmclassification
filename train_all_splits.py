import subprocess


for split in range(1, 6):
    model_config_path = f"/data/home/bedward/workspace/mmpose-project/mmclassification/configs/resnet/resnet50_8xb32_in1k_apas_split{split}.py"
    model_name = f"resnet50_8xb32_in1k_apas"
    cmd = f"python tools/train.py {model_config_path} &> logs/train_apas_classification_{model_name}_split{split}.txt"
    print(cmd)
    out = subprocess.check_output(cmd, shell=True)
    print(out)