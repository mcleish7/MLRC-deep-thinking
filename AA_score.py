"""
    Code to produce AA scores for DT-models, changing parameters for the models is done in the exact same way as for training
"""

import torch
import numpy as np
import deepthinking.models as models
import deepthinking as dt
from omegaconf import DictConfig, OmegaConf
import hydra
import sys
import os
from tqdm import tqdm
import nvidia_smi
import argparse

def get_mazes_net(device, path):
    """
    Returns the DT recall (progressive) network NOT in evaluation mode to solve maze problems
    Note: We are not in eval mode as we want to access interim_thought

    Args:
        path (str): The relative file path to the network to be loaded
        device (str): the device to store the network on

    Returns:
        torch.nn: the neural net
    """
    state_dict = torch.load(path, map_location=device)
    net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=50)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    return net

def get_sums_net(device, path):
    """
    Returns the DT recall (progressive) network NOT in evaluation mode to solve prefix sums problems
    Note: We are not in eval mode as we want to access interim_thought

    Args:
        path (str): The relative file path to the network to be loaded
        device (str): the device to store the network on

    Returns:
        torch.nn: the neural net
    """
    net = getattr(models, "dt_net_recall_1d")(width=400, in_channels=3, max_iters=300)
    state_dict = torch.load(path, map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    return net

def get_chess_net(device, path):
    """
    Returns the DT recall (progressive) network NOT in evaluation mode to solve chess problems
    Note: We are not in eval mode as we want to access interim_thought

    Args:
        path (str): The relative file path to the network to be loaded
        device (str): the device to store the network on

    Returns:
        torch.nn: the neural net
    """
    net = getattr(models, "dt_net_recall_2d")(width=512, in_channels=12, max_iters=30)
    state_dict = torch.load(path, map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    return net

def compute_cross_pi(net, testloader, iters, problem, device):
    """
    Computes the Asymptotic Alignment Score (AA score) for a given network
    Code provided by Cem Anil and Ashwini Pokle, a prerelease of their code base which is currently being cleaned ready for release
    Code modified by Authors, minor chnaged such as the if statement to make the code work with my models and codebase
    
    Args:
        net (torch.nn): the network to be tested
        testloader (torch.utils.data.Dataloader): The test data to run the tests on
        iters (int): The number of iterations to run each part of the test for
        device (str): The device we are working on
        alpha (int): The alpha value of the network

    Returns:
        float: the AA score
    """
    total = 0

    idx = 0
    path_indep_val = 0

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
 
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            torch.cuda.empty_cache()
            inputs, targets = inputs.to(device), targets.to(device)
            init_outputs, fp_val1 = net(inputs, return_fp=True)
            if problem.name == "prefix_sums": #prefix sums need a different shape input, to other two problems
                tiled_inputs = torch.tile(inputs, (inputs.shape[0], 1, 1))
                tiled_targets = torch.tile(targets, (targets.shape[0], 1))
            else: 
                tiled_inputs = torch.tile(inputs, (inputs.shape[0], 1, 1, 1))
                tiled_targets = torch.tile(targets, (targets.shape[0], 1, 1))
            repeated_fp = torch.repeat_interleave(fp_val1, repeats=inputs.shape[0], dim=0)
            next_outputs, fp_val2 = net(tiled_inputs, interim_thought=repeated_fp, return_fp=True)
            total += fp_val2.size(0)

            idx = np.arange(0, tiled_inputs.shape[0], inputs.shape[0])
            fp1 = repeated_fp.view(repeated_fp.shape[0], -1)
            fp2 = fp_val2.view(fp_val2.shape[0], -1)
            
            bsz = inputs.shape[0]
            for i in range(inputs.shape[0]):
                cur_idx = idx + i
                conseq_idx = np.arange(i*bsz, i*bsz + inputs.shape[0])
                path_indep_val += cos(fp1[cur_idx], fp2[conseq_idx]).sum()
    return path_indep_val/total

@hydra.main(config_path="config", config_name="test_model_config")
def main(cfg: DictConfig):
    """
    Computes the Asymptotic Alignment score (AA score) for each model in the filepaths list and writes this to a file

    Args:
        cfg (DictConfig): Uses the hydra framework like the rest of the codebase from Bansal et al to take command line arguments
    """
    problem = cfg.problem
    loaders = dt.utils.get_dataloaders(problem)

    os.chdir('../../..') # as we have created an instace we are in the file for that instance

    testloader = loaders["test"] # uses the test input paramater data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    iters = 300
    aa = []
    filepaths = [] # Add your file paths here, remember they must all be for the same problem, e.g. prefix sums, this must match the command line argument given.
    # EXAMPLE: filepaths = [outputs/prefix_sums_ablation/training-unbalanced-Nick/model_best.pth]
    for path in filepaths:
        if problem.name == "prefix_sums":
            net = get_sums_net(device, path)
        elif problem.name == "mazes":
            net = get_mazes_net(device, path)
        else:
            net = get_chess_net(device, path)
        aa_score = compute_cross_pi(net, testloader, iters, problem, device).tolist()
        print(f"for model on path {path}, the AA score is {aa_score[0]}")
        aa.append(aa_score)

    file_name = f"score_{problem.name}_{problem.train_data}.txt"
    with open(file_name, 'w+') as f:
        f.write(f"for models in paths: {filepaths} the time array is {aa}")

if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")
    main()