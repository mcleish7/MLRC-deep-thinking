"""
In this file there are unused methods this is to faciliate quick chnages to its use in the future by other users
"""
import pytorchfi as fi
from pytorchfi.core import fault_injection
import torch
import deepthinking.models as models
import deepthinking.utils as dt
import numpy as np
import sys
import os
import json
import matplotlib.pyplot as plt
from easy_to_hard_plot import plot_maze
from easy_to_hard_plot import MazeDataset
import argparse

cuda_avil = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_no_prog_net(path):
    """
    Gets the non progressive net with an alpha value of 0 and returns it in evaluation mode

    Returns:
        Torch.nn: the non progressive neural net to solve mazes
    """
    net = getattr(models, "dt_net_2d")(width=128, in_channels=3, max_iters=30) # for Lao => not recall, alpha =0
    state_dict = torch.load(path, map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

def get_prog_net(path):
    """
    Gets the progressive net with an alpha value of 1 and returns it in evaluation mode

    Returns:
        Torch.nn: the progressive neural net to solve mazes
    """
    net = getattr(models, "dt_net_2d")(width=128, in_channels=3, max_iters=30) # for Cor => not recall, alpha =1
    state_dict = torch.load(path, map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

def get_recall_prog_net(path):
    """
    Gets the progressive recall net with an alpha value of 1 and returns it in evaluation mode

    Returns:
        Torch.nn: the progressive recall neural net to solve mazes
    """
    net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=30) # for Paden => recall, alpha =1
    state_dict = torch.load(path, map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

def get_recall_no_prog_net(path):
    """
    Gets the non progressive recall net with an alpha value of 1 and returns it in evaluation mode

    Returns:
        Torch.nn: the non progressive recall neural net to solve mazes
    """
    net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=30) # for Col => recall, alpha =0
    state_dict = torch.load(path, map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

def get_data(path):
    """
    Gets the raw maze data from their local file

    Returns:
        tensor, tensor:
         1) the full  input data set as a Torch.tensor
         2) the full target data set as a Torch.tensor
    """
    data_path = path + "/maze_data_test_13/inputs.npy"
    target_path = path + "/maze_data_test_13/solutions.npy"
    data = np.load(data_path)
    target = np.load(target_path)
    a = data[1]
    a = torch.from_numpy(a)
    input = a.to(device, dtype=torch.float).unsqueeze(0) #to account for batching in real net
    b = target[1]
    t = torch.from_numpy(b)
    t = t.to(device, dtype=torch.float)#.long()
    target = t.unsqueeze(0)
    return input, target

def convert_to_bits(output, input):
    """Convert the input string to a bits stored in an array

    Args:
        output (tensor): the array to convert
        input (tensor): the input ot he net

    Returns:
        golden_label (numpy array): the input string to a bits stored in an array
    """
    predicted = output.clone().argmax(1)
    predicted = predicted.view(predicted.size(0), -1)
    golden_label = predicted.float() * (input.max(1)[0].view(input.size(0), -1)) # maps the maze output to only possible paths 
    return golden_label

def l2_norm(output): 
    """
    Takes the L2 norm of the difference between the features of the output of the net for a run

    Args:
        output (tensor): output of a run of the net of size [1, x, 2, 32, 32] where x is the number of iterations in the run

    Returns:
        list: the ith index is the L2-norm of the ith itertaion- (i+1)th iteration
    """
    # output will be size: [1, 50, 2, 32, 32] to be split into [1, 2, 32, 32]
    out = []
    output1 = output[:, 0]
    output1 = output1.cpu().detach().numpy().flatten()
    for i in range(0,output.size(1)-1):
        output2 = output[:, i+1]
        output2 = output2.cpu().detach().numpy().flatten() # easier to work on flat vectors
        norm = np.sum(np.power((output1-output2),2)) # L2 norm implementation
        out.append(norm)
        output1= output2
    return out

def net_out_to_bits(input,output,target, log = False, graph = False): 
    """
    For a testing run of a NN finds the output of the NN with the best accuracy and what this output was.
    Can also store the run in a json file and plot it.

    Args:
        output (tensor): output from a run of the NN
        target (tensor): the target of the NN run
        log (bool, optional): If set to True, saves the number of correct bits at iteration i for each iteration of the NN in a json file. Defaults to False.
        graph (bool, optional): If set to True, graphs the number of correct bits per iteration of the NN using the graph_progress method. Defaults to False.

    Returns:
        numpy.array, int: numpy array of the closest prediction of the NN to the target and the iteration that this was at. Takes the first index if reaches 100% accuracy
    """
    output = output.clone()
    corrects = torch.zeros(output.size(1))
    for i in range(output.size(1)):
        outputi = output[:, i]
        golden_label = convert_to_bits(outputi, input)
        target = target.view(target.size(0), -1)
        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item()
        if i ==50:
            np.save("50_maze_tensor",outputi.cpu().detach().numpy())
    correct = corrects.cpu().detach().numpy()
    bestind = np.argmax(correct)
    best = output[:,bestind]
    return convert_to_bits(best, input), correct[bestind]

class custom_func(fault_injection):
    """
    Custom peturbation class to peturb the input of the mazes
    Inherits:
        fault_injection (class): pytorchfi.core.fault_injection

    Methods
    -------
    flip_all(self, module, input, output)
        called at each iteration of the NN, currently does not implement any peturbation as the point of this file is to measure the chnage in features per iteration
        but as stated this is left for users to quickly chnage the use case and perform quick tests
    """
    def __init__(self, model, batch_size, **kwargs):
        """constructor for custom_func

        Args:
            model (_type_): the network
            batch_size (int): batch size for the network
        """
        super().__init__(model, batch_size, **kwargs)

    def flip_all(self, module, input, output): 
        """
        Called in each iteration of the NN automatically using the PyTorchFI framework so we can alter the output of that layer
        Currently does nothing to the layer
        Args:
            module (specified in layer_types super class varibale): the type of the layer
            input (tensor): the input to the layer
            output (tensor): the output of the layer
        """
        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()

def tester(net, batch_size, channels, width, height, layer_types_input, input):
    """
    Runs the input network via the PyTorchFI framework with the given input

    Args:
        net (Torch.nn): the network to perform peturbation on
        batch_size (int): the batch size for the testing
        channels (int): number of channels in the input network
        width (int): width of input network
        height (int): height of input network
        layer_types_input (list): list of the types of layers to call the flip_all function at
        input (tensor): the input to run the input net on

    Returns:
        tensor: the output of the run of the network
    """
    with torch.no_grad():
        pfi_model = custom_func(net, 
                                batch_size,
                                input_shape=[channels,width,height],
                                layer_types=layer_types_input,
                                use_cuda=True
                            )
        inj = pfi_model.declare_neuron_fi(function=pfi_model.flip_all)

        return inj(input)

parser = argparse.ArgumentParser(description="Time parser")
# e.g. data/
parser.add_argument("--data_path", type=str, help="the path to the directory the data to test on is stored in", required=True)
args = parser.parse_args()

#parameters for PyTorchFI to funciton
input, target = get_data(args.data_path)
batch_size = 1
channels = 3
width = 128
height = width
layer_types_input = [torch.nn.Conv2d]

recall_prog_path = "outputs/mazes_13x13/training-repand-Natilee/model_best.pth" # DT-Recall-Prog path i.e alpha>0 and recall=True
recall_path = "outputs/mazes_13x13/training-smuggest-Bo/model_best.pth" # DT-Recall path i.e. alpha=0 and recall=True
prog_path = "outputs/mazes_13x13/training-wasted-Devonne/model_best.pth" # DT-Prog path i.e. alpha>0 and recall=False
dt_path = "outputs/mazes_13x13/training-desired-Eustacia/model_best.pth" # DT path i.e. alpha=0 and recall=False

# running a test on each of the 4 types of nets
recall_prog_output = tester(get_recall_prog_net(recall_prog_path),batch_size, channels, width, height, layer_types_input, input)
recall_no_prog_output = tester(get_recall_no_prog_net(recall_path),batch_size, channels, width, height, layer_types_input, input)
prog_output = tester(get_prog_net(prog_path),batch_size, channels, width, height, layer_types_input, input)
no_prog_output = tester(get_no_prog_net(dt_path),batch_size, channels, width, height, layer_types_input, input)

def graph_norm_progress(arr1, arr2, arr3, arr4):
    """
    graphs the changes measured by the L2 norm in the network run 

    Args:
        arr1 (list): the changes measured by the L2 norm of the run of the DT-Recall-Prog net
        arr2 (list): the changes measured by the L2 norm of the run of the DT-Recall net
        arr3 (list): the changes measured by the L2 norm of the run of the DT-Prog net
        arr4 (list): the changes measured by the L2 norm of the run of the DT net
    """
    plt.clf()
    plt.plot(arr1, linewidth = '2.0', label = "DT-Recall-Prog")
    plt.plot(arr2, linewidth = '2.0', label = "DT-Recall")
    plt.plot(arr3, linewidth = '2.0', label = "DT-Prog")
    plt.plot(arr4, linewidth = '2.0', label = "DT")
    plt.yscale('log') # makes the y axis exponential
    plt.ylim([10**-8,10**15]) # may need altering for specific models
    plt.title('Change in features over time')
    plt.xlabel('Test-Time iterations')
    plt.ylabel('Δφ')
    plt.legend(loc="upper right")
    save_path = "test_changes_correctness.png"
    plt.savefig(save_path)

# Records the norm of the chnages of each net
recall_prog_norm = l2_norm(recall_prog_output)
recall_no_prog_norm = l2_norm(recall_no_prog_output)
prog_norm = l2_norm(prog_output)
no_prog_norm = l2_norm(no_prog_output)

graph_norm_progress(recall_prog_norm, recall_no_prog_norm, prog_norm, no_prog_norm)