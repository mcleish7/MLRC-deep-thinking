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

def get_mazes_net(device, which_net):
    """
    Returns the DT recall (progressive) network NOT in evaluation mode to solve maze problems
    Note: We are not in eval mode as we want to access interim_thought

    Args:
        which_net (int): The alpha value of the network times 10, e.g. which_net=5 maps to a net with alpha value 0.5
        device (str): the device to store the network on

    Returns:
        torch.nn: the neural net
    """
    if which_net == 1:
        name = "inmost-Quenten"
    elif which_net == 2:
        name = "yolky-Dewaun"
    elif which_net == 3:
        name = "tented-Arlena"
    elif which_net == 4:
        name = "cormous-Andreah"
    elif which_net == 5:
        name = "tinsel-Rosalia"
    elif which_net == 6:
        name = "exchanged-Nyasia"
    elif which_net == 7:
        name = "feeblish-Ernesto"
    elif which_net == 8:
        name = "cosher-Taneika"
    elif which_net == 9:
        name = "praising-Kimberely"
    else:
        name = "heating-Mihcael"
    full_path = f"mismatch/outputs/mazes_ablation/training-{name}/model_best.pth"
    state_dict = torch.load(full_path, map_location=device)
    net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=50)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    return net

def get_sums_net(device,which_net):
    """
    Returns the DT recall (progressive) network NOT in evaluation mode to solve prefix sums problems
    Note: We are not in eval mode as we want to access interim_thought

    Args:
        which_net (int): The alpha value of the network times 10, e.g. which_net=5 maps to a net with alpha value 0.5
        device (str): the device to store the network on

    Returns:
        torch.nn: the neural net
    """
    net = getattr(models, "dt_net_recall_1d")(width=400, in_channels=3, max_iters=300)
    if which_net == 1:
        name = "logy-Hester"
    elif which_net == 2:
        name = "chequy-Aniel"
    elif which_net == 3:
        name = "gnomic-Rashanna"
    elif which_net == 4:
        name = "checkered-Bethani"
    elif which_net == 5:
        name = "freckly-Lonnell"
    elif which_net == 6:
        name = "stunning-Hank"
    elif which_net == 7: 
        name = "faddy-Pual"
    elif which_net == 8:
        name = "gowaned-Ayla"
    elif which_net == 9: #i.e. is 0.9
        name = "crudest-Tanda"
    else: #i.e. is 0.01
        name = "fistic-Lizzie"
    path = f"mismatch/outputs/prefix_sums_ablation/training-{name}/model_best.pth"
    state_dict = torch.load(path, map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    return net

def get_chess_net(device,which_net):
    """
    Returns the DT recall (progressive) network NOT in evaluation mode to solve chess problems
    Note: We are not in eval mode as we want to access interim_thought

    Args:
        which_net (int): The alpha value of the network times 10, e.g. which_net=5 maps to a net with alpha value 0.5
        device (str): the device to store the network on

    Returns:
        torch.nn: the neural net
    """
    net = getattr(models, "dt_net_recall_2d")(width=512, in_channels=12, max_iters=30)
    if which_net == 1:
        name = "healthful-Bakari"
    elif which_net == 2:
        name = "oaten-Rayne"
    elif which_net == 3:
        name = "sunfast-Treavor"
    elif which_net == 4:
        name = "many-Lisanne"
    elif which_net == 5:
        name = "grummest-Montague"
    elif which_net == 6:
        name = "unformed-Roisin"
    elif which_net == 7: 
        name = "tardy-Daniels"
    elif which_net == 8:
        name = "select-Corneilus"
    elif which_net == 9: #i.e. is 0.9
        name = "intense-Mikos"
    else: #i.e. is 0.01
        name = "noisette-Alberta"
    path = f"mismatch/outputs/chess_abalation/training-{name}/model_best.pth"
    state_dict = torch.load(path, map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    return net

def compute_cross_pi(net, testloader, iters, problem, device, alpha):
    """
    Computes the Asymptotic Alignment Score (AA score) for a given network
    Code provided by Cem Anil and Ashwini Pokle, a prerelease of their code base which is currently being cleaned ready for release
    Code modified by Sean McLeish, minor chnaged such as the if statement to make the code work with my models and codebase
    
    Args:
        net (torch.nn): the network to be tested
        testloader (torch.utils.data.Dataloader): The test data to run the tests on
        iters (int): The number of iterations to run each part of the test for
        device (str): The device we are working on
        alpha (int): The alpha value of the network

    Returns:
        float: the AA score
    """
    corrects = 0
    total = 0

    idx = 0
    path_indep_val = 0

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # nvidia_smi.nvmlInit()
    # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate
    # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    # print("Total memory:", info.total)
    # print("Free memory before loop:", info.free)
 
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            torch.cuda.empty_cache()
            inputs, targets = inputs.to(device), targets.to(device)
            print("inputs shape is ",inputs.shape)
            print("targets shape is ",targets.shape)
            init_outputs, fp_val1 = net(inputs, return_fp=True)
            if problem.name == "prefix_sums": #prefix sums need a different shape input
                tiled_inputs = torch.tile(inputs, (inputs.shape[0], 1, 1))
                tiled_targets = torch.tile(targets, (targets.shape[0], 1))
            # elif problem.name == "prefix_sums":
            #     tiled_inputs = torch.tile(inputs, (inputs.shape[0], 1, 1, 1))
            #     tiled_targets = torch.tile(targets, (targets.shape[0], 1, 1))
            else: 
                tiled_inputs = torch.tile(inputs, (inputs.shape[0], 1, 1, 1))
                tiled_targets = torch.tile(targets, (targets.shape[0], 1, 1))
            print("tiled_inputs is shape",tiled_inputs.shape)
            print("tiled_targets is shape",tiled_targets.shape)
            # print("Free memory before interleave:", info.free)
            # print("inputs shape 0 is ",inputs.shape[0])
            repeated_fp = torch.repeat_interleave(fp_val1, repeats=inputs.shape[0], dim=0)
            print("repeated fp is shape",repeated_fp.shape)
            # nvidia_smi.nvmlShutdown()  
            next_outputs, fp_val2 = net(tiled_inputs, interim_thought=repeated_fp, return_fp=True)
            # print("Free memory after interleave:", info.free)
            total += fp_val2.size(0)

            idx = np.arange(0, tiled_inputs.shape[0], inputs.shape[0])
            fp1 = repeated_fp.view(repeated_fp.shape[0], -1)
            fp2 = fp_val2.view(fp_val2.shape[0], -1)
            
            bsz = inputs.shape[0]
            for i in range(inputs.shape[0]):
                cur_idx = idx + i
                conseq_idx = np.arange(i*bsz, i*bsz + inputs.shape[0])
                path_indep_val += cos(fp1[cur_idx], fp2[conseq_idx]).sum()
            # break
        print("for net ",alpha," Cosine similarity", path_indep_val/total)
    return path_indep_val/total

@hydra.main(config_path="config", config_name="test_model_config")
def main(cfg: DictConfig):
    """
    Computes the Asymptotic Alignment score (AA score) for each model in the 'alphas' list and writes this to a file

    Args:
        cfg (DictConfig): Uses the hydra framework like the rest of the codebase to take command line arguments
    """
    problem = cfg.problem

    # print("train batch size is ",problem.hyp.train_batch_size)
    # print("test batch size is ",problem.hyp.test_batch_size)
    loaders = dt.utils.get_dataloaders(problem)
    # print(type(loaders["test"]))

    cwd = os.getcwd()
    # print("cwd is ",cwd)
    os.chdir('../../..')
    cwd = os.getcwd()
    # print("cwd is ",cwd)
    testloader = loaders["test"]#[loaders["test"], loaders["val"], loaders["train"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    iters = 300
    aa = []
    alphas = [-1,1,2,3,4,5,6,7,8,9]
    for alpha in alphas:
        if problem.name == "prefix_sums":
            net = get_sums_net(device, alpha)
        elif problem.name == "mazes":
            net = get_mazes_net(device, alpha)
        else:
            net = get_chess_net(device, alpha)
        aa.append(compute_cross_pi(net, testloader, iters, problem, device, alpha).tolist())
        file_name = f"score_{problem.name}_{problem.train_data}.txt"
        with open(file_name, 'w+') as f:
            f.write(f"for alpha: {alpha} the time array is {aa}")

if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")
    main()

# As in other files storing the raw data in the files for inspection and further analysis to keep work together

# python3.9 score_email.py problem=prefix_sums problem.hyp.test_batch_size=10 problem.hyp.train_batch_size=10 problem.test_data=512 problem.train_data=32
# maze models:
# for alpha: 8 the time array is 
maze91 = [0.9971649646759033, 0.9980168342590332, 0.9968770146369934, 0.931258499622345, 0.9408687949180603, 0.7856437563896179, 0.9429271817207336, 0.9065360426902771, 0.8678450584411621]
# for alpha: 5 the time array is [0.9873379468917847]


# for alpha: 9 the time array is 
maze92 = [0.916424572467804]
maze9 = maze91 + maze92
maze59 = [0.9758304953575134, 0.9607346057891846, 0.963686466217041, 0.9220239520072937, 0.8237228989601135, 0.6947765946388245, 0.8579129576683044, 0.8902077078819275, 0.8086482882499695, 0.9082356691360474]
# for alpha: 5 the time array is [0.9195083379745483]

sums32 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9987200498580933, 1.0]
sums512 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9987200498580933, 1.0]
#changed nan to 0.0
chess7 = [0.0, 0.005817557219415903, 0.8072536587715149, 0.9524214863777161, 0.9572643041610718, 0.9805848598480225, 0.9854093194007874, 0.9914458394050598, 0.9960649609565735, 0.9952036142349243]
chess11 = [0.0, 0.006417518015950918, 0.8050198554992676, 0.9460858702659607, 0.9505954384803772, 0.9736934900283813, 0.9806694984436035, 0.9879935383796692, 0.9943339228630066, 0.9931629300117493]

def graph_array(maze9, maze59, sums32, sums512, chess7, chess11):
  f, ax = plt.subplots(1, 1)
  ax.plot(maze9, label = "Maze 9x9")
  ax.plot(maze59, label = "Maze 59x59")
  ax.plot(sums32, label = "Sums 32bits")
  ax.plot(sums512, label = "Sums 512bits")
  ax.plot(chess7, label = "Chess [600k-700k]")
  ax.plot(chess11, label = "Chess [1M-1.1M]")
  labels = ["0.01","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"]
  ticks = [0,1,2,3,4,5,6,7,8,9]
  plt.xticks(ticks=ticks, labels=labels)
  ax.set(xlabel='Alpha Value', ylabel='AA score', title="AA score for maze models over alpha values")
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.savefig("mazes_AA_score", bbox_inches="tight", dpi=500)

graph_array(maze9, maze59, sums32, sums512, chess7, chess11)