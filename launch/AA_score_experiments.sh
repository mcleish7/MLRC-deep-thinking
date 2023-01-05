# !!Edit line 140 of AA_score, adding model paths for the code to run correctly!!

# EXAMPLE:
# python AA_score.py problem=mazes problem.hyp.test_batch_size=10 problem.hyp.train_batch_size=10 problem.test_data=59 problem.train_data=59

# Complete next line with your parameters:
# Small batch sizes tend to be needed here
python sums_peturb.py problem.hyp.test_batch_size=10 problem.hyp.train_batch_size=10 problem=