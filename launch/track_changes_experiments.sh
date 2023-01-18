# Change the file paths on lines 234-237 to the correct ones for your new models
# General file path structure: "outputs/{name_of_batch}/training-{model_name}/model_best.pth"

# Example:
# recall_prog_path = "batch_shells_maze/outputs/mazes_ablation/training-abased-Paden/model_best.pth" # DT-Recall-Prog path i.e alpha>0 and recall=True
# recall_path = "batch_shells_maze/outputs/mazes_ablation/training-algal-Collyn/model_best.pth" # DT-Recall path i.e. alpha=0 and recall=True
# prog_path = "batch_shells_maze/outputs/mazes_ablation/training-distinct-Cornesha/model_best.pth" # DT-Prog path i.e. alpha>0 and recall=False
# dt_path = "batch_shells_maze/outputs/mazes_ablation/training-boughten-Lao/model_best.pth" # DT path i.e. alpha=0 and recall=False

# e.g. --data_path=data

python ../track_changes.py --data_path=
