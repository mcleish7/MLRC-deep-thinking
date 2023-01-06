import json
import os
from omegaconf import OmegaConf
import glob
from tabulate import tabulate
import pandas as pd

filepaths = ["outputs/test_mazes"] # edit this list to add the directories with the tested models in

count =0
checkpoints = []
for filepath in filepaths:
    for f_name in glob.iglob(f"{filepath}/**/*testing*/stats.json", recursive=True):
        print(f_name)
        model_path = "/".join(f_name.split("/")[1:-1])
        json_name = os.path.join("/".join(f_name.split("/")[:-1]), "stats.json")
        with open(json_name) as f:
            data = json.load(f)
        checkpoints.append([model_path,
        round(data["alpha"],2),
        round(max(data["train_acc"].values()),3),
        round(max(data["val_acc"].values()),3),
        round(max(data["test_acc"].values()),3)
        ])
        count+=1

head = ["Model Name","Alpha","Train Acc","Val Acc","Test Acc"]
print(tabulate(checkpoints, headers=head))