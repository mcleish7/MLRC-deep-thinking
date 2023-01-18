# Train on [0,600k], test on [600k,700k]
python ../train_model.py problem.hyp.alpha=0 problem/model=dt_net_2d problem=chess name=chess_700k
python ../train_model.py problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=chess name=chess_700k
python ../train_model.py problem.hyp.alpha=0.5 problem/model=dt_net_2d problem=chess name=chess_700k
python ../train_model.py problem.hyp.alpha=0.5 problem/model=dt_net_recall_2d problem=chess name=chess_700k

# Train on [0,600k], test on [1M,1.1M]
python ../train_model.py problem.hyp.alpha=0 problem/model=dt_net_2d problem=chess name=chess_1100k problem.test_data=1100000
python ../train_model.py problem.hyp.alpha=0 problem/model=dt_net_recall_2d problem=chess name=chess_1100k problem.test_data=1100000
python ../train_model.py problem.hyp.alpha=0.5 problem/model=dt_net_2d problem=chess name=chess_1100k problem.test_data=1100000
python ../train_model.py problem.hyp.alpha=0.5 problem/model=dt_net_recall_2d problem=chess name=chess_1100k problem.test_data=1100000

# If the model does not train in full in one run. 
# Use the same command and append "problem.model.model_path=../../../outputs/{batch_name}/training-{model_name}/model_best.pth" to the end  of the command.
# This will pick up the training from the best iteration of the previous run.