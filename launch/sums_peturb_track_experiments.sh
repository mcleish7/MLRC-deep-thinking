# 3 required command line arguments:
# --net_path
#     The path including "model_best.pth" of the training of the network to be tested

# --data_path
#     The path including "data"

# --size
#     The size of the bit strings to test on

# EXAMPLE:
# python ../sums_track_changes.py --net_path outputs/prefix_sums_ablation/training-unbalanced-Nick/model_best.pth --data_path data --size 512

# Complete next line with your parameters:
python ../sums_track_changes.py --net_path --data_path --size