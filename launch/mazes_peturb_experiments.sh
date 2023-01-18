# 3 required command line arguments:
# --net_paths
#     The paths including "model_best.pth" of the training of the networks to be tested
#     Unlike sums here you can input multiple models at once as it is quicker to run these experiments.

# --data_path
#     The path including "data"

# --subsets
#     The size of the subset to test on
#     Is an integer <=10,000

# EXAMPLE:
# python maze_peturb.py --net_paths outputs/mazes_ablation/training-unbalanced-Nick/model_best.pth outputs/mazes_ablation/training-dotted-Liam/model_best.pth --data_path data --subsets 5000 10000

# Complete next line with your parameters:
python ../maze_peturb.py --net_paths --data_path --subsets