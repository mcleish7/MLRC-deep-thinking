# Train on 9x9, test on 13x13
python train_model.py problem.hyp.alpha=0.00 problem/model=dt_net_2d problem=mazes name=mazes_13x13 problem.test_data=13
python train_model.py problem.hyp.alpha=0.01 problem/model=dt_net_2d problem=mazes name=mazes_13x13 problem.test_data=13
python train_model.py problem.hyp.alpha=0.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_13x13 problem.test_data=13
python train_model.py problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes name=mazes_13x13 problem.test_data=13

# Train on 9x9, test on 59x59
python train_model.py problem.hyp.alpha=0.00 problem/model=dt_net_2d problem=mazes name=mazes_59x59 problem.test_data=59
python train_model.py problem.hyp.alpha=0.01 problem/model=dt_net_2d problem=mazes name=mazes_59x59 problem.test_data=59
python train_model.py problem.hyp.alpha=0.00 problem/model=dt_net_recall_2d problem=mazes name=mazes_59x59 problem.test_data=59
python train_model.py problem.hyp.alpha=0.01 problem/model=dt_net_recall_2d problem=mazes name=mazes_59x59 problem.test_data=59