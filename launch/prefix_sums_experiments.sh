# Train on 32 bits, test on 48 bits
python train_model.py problem.hyp.alpha=1 problem/model=dt_net_1d problem=prefix_sums name=prefix_sums_48_bits problem.test_data=48
python train_model.py problem.hyp.alpha=0 problem/model=dt_net_1d problem=prefix_sums name=prefix_sums_48_bits problem.test_data=48
python train_model.py problem.hyp.alpha=1 problem/model=dt_net_recall_1d problem=prefix_sums name=prefix_sums_48_bits problem.test_data=48
python train_model.py problem.hyp.alpha=0 problem/model=dt_net_recall_1d problem=prefix_sums name=prefix_sums_48_bits problem.test_data=48

# Train on 32 bits, test on 512 bits
# You can use the same models from above and only test on 512bit data instead of training twice, this is the same for all type of models as long all hyperparameters but test_data are identical. 
python train_model.py problem.hyp.alpha=1 problem/model=dt_net_1d problem=prefix_sums name=prefix_sums_512_bits
python train_model.py problem.hyp.alpha=0 problem/model=dt_net_1d problem=prefix_sums name=prefix_sums_512_bits
python train_model.py problem.hyp.alpha=1 problem/model=dt_net_recall_1d problem=prefix_sums name=prefix_sums_512_bits
python train_model.py problem.hyp.alpha=0 problem/model=dt_net_recall_1d problem=prefix_sums name=prefix_sums_512_bits

# Testing: please refer to readme.md first.

# To test a model run:
# Use the same command as training then
# Change 'train_model.py' to 'test_model.py'
# You probably want to change the 'name' parameter too so the testing is put into a different file than the training
# To change the number of iterations you test for use: 'problem.model.test_iterations.low=' and 'problem.model.test_iterations.high='
# Append 'problem.model.model_path=../../../outputs/{batch-name}/training-{model-name}' to the end of the command
# EXAMPLE:
# python test_model.py problem/model=dt_net_recall_1d problem=prefix_sums name=test_prefix_alpha problem.test_data=512 problem.train_data=32 problem.model.test_iterations.low=0 problem.model.test_iterations.high=500 problem.model.model_path=../../../outputs/prefix_alpha/training-licenced-Erika