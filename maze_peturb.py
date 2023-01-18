"""
    Code to reproduce maze peturbation after 50 iterations
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

def get_net(device, path):
    """
    Returns the DT recall (progressive) network in evaluation mode

    Args:
        path (str): The relative path to the net to be loaded
        device (str): the device to store the network on

    Returns:
        torch.nn: the neural net
    """
    state_dict = torch.load(path, map_location=device)
    net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=50)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net


def get_data(datapath,n=10):
    """
    Returns the maze data as tensors ready to be used in the models

    Args:
        datapath (str): the filepath up to and including the data directory
        n (int, optional): the number of elements wanted. Defaults to 10.

    Returns:
        torch.tensor: [n,3,32,32] shape tensor of inputs to the networks
        torch.tensor: [n,3,32,32] shape tensor of solutiosn to the input problems
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_path = datapath + "/maze_data_test_59/inputs.npy"
    target_path = datapath + "/maze_data_test_59/solutions.npy"
    data = np.load(input_path)
    target = np.load(target_path)
    a = data[:n]
    a = torch.from_numpy(a)
    input = a.to(device, dtype=torch.float)
    b = target[:n]
    t = torch.from_numpy(b)
    t = t.to(device, dtype=torch.float)
    target = t
    return input, target

def convert_to_bits(device, output, input):
    """
    Converts the output of the net to its prediciton

    Args:
        device (str): the device we are working on
        output (tensor): the output of the net
        input (tensor): the input to the net

    Returns:
        tensor: the prediction of the net
    """
    output = output.unsqueeze(0).to(device)
    predicted = output.clone().argmax(1)
    predicted = predicted.view(predicted.size(0), -1)
    golden_label = predicted.float() * (input.max(1)[0].view(input.size(0), -1)) #used to map the output into only paths that exist in the maze
    return golden_label

class custom_func(fault_injection):
    """
    Custom peturbation class to peturb mazes
    Inherits:
        fault_injection (class): pytorchfi.core.fault_injection

    Methods
    -------
    flip_all(self, module, input, output)
        called at the 50th iteration of the NN, sets all of the features to 0
    """
    def __init__(self, model, batch_size, **kwargs):
        """constructor for custom_func

        Args:
            model (torch.nn): the network
            batch_size (int): batch size for the network
        """
        super().__init__(model, batch_size, **kwargs)

    def flip_all(self, module, input, output):
        """
        Called in each iteration of the NN automatically using the PyTorchFI framework so we can alter the output of that layer

        Args:
            module (specified in layer_types super class varibale): the type of the layer
            input (tensor): the input to the layer
            output (tensor): output is a tuple of length 1, with index 0 holding the current tensor
        """
        layer_from = 50 #for small GPU's use 25 or less, for larger ones we can use the full result of 50
        layer_to = 51
        if (self.get_current_layer() >= (layer_from*7)) and (self.get_current_layer() <= ((layer_to*7)+1)): # observation: is the direct relation to the size of the recurrent module
            output[:] = torch.zeros(output.shape) # puts all outputs from the layer to 0
        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()

def graph_helper(device, output,input,target):
    """
    Very much like convert to bits but for each iterations ouputs and converts output to numpy array

    Args:
        device (str): the device we are working on
        output (Torch.tensor): the output from a run of a net
        input (Torch.tensor): the input to the net
        target (Torch.tensor):the target of the net

    Returns:
        numpy.array: the number of bits which were predicted correctly at each iteration of the net
    """
    output = output.clone().squeeze()
    corrects = torch.zeros(output.shape[0])
    for i in range(output.shape[0]): # goes through each iteration
        outputi = output[i]
        golden_label = convert_to_bits(device, outputi, input)
        target = target.view(target.size(0), -1)
        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item() # counts the number that are the same i.e. correct predictions
    correct = corrects.cpu().detach().numpy()
    return correct

def graph_maze_mismatch(filepaths, runs):
    """
    Graphs results of maze peturbations

    Args:
        filepaths (list): the filepath to each model, overwrite this list if you want different descriptions in the legend
        runs (np.array): the accruacy outputs of the runs for each model in filepaths
    """
    plt.clf()
    denom = 15376.0 # the ouput matrix for an nxn net is size (2(n+2))^2
    for i in range(0,len(runs)):
        run = runs[i]
        plt.plot(run*(100.0/denom), linewidth = '1.0', label = filepaths[i])
    plt.title('Accuracy over time when features swapped')
    plt.xlabel('Test-Time iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    plt.savefig("maze_mismatch.png", dpi=500)

def main_module(filepaths, datapath, number=100):
    """
    Runs the testing

    Args:
        filepaths (list) : a list of filepaths, for each model to be tested
        datapath (str): the filepath up to and including the data directory, for the maze data
        number (int, optional): _description_. Defaults to 100.
    """
    # PyTorchFi parameters for the maze nets
    batch_size = 1
    channels = 3
    width = 128
    height = width
    layer_types_input = [torch.nn.Conv2d]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    averages = []
    inputs, targets = get_data(datapath,n=number)
    for path in filepaths:
        print(f"on path: {path}")
        outputs = []
        with torch.no_grad():
            net = get_net(device, path)
            pfi_model = custom_func(net, 
                                    batch_size,
                                    input_shape=[channels,width,height],
                                    layer_types=layer_types_input,
                                    use_cuda=True
                                )
            for i in range(0,inputs.size(0)): # test each example
                input = inputs[i].unsqueeze(0) # have to unsqueeze to simulate batches
                target = targets[i].unsqueeze(0) 
                inj = pfi_model.declare_neuron_fi(function=pfi_model.flip_all)
                out = inj(input)
                converted = graph_helper(device, out, input, target) # converts the nets outputs to a list of accuracies
                outputs.append(converted)
            outputs = np.array(outputs)
            average = np.mean(outputs, axis = 0) # average over all examples seen
            averages.append(average)
    graph_maze_mismatch(filepaths, averages)

def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument('--net_paths', nargs='+', help='the path to each model', required=True)
    parser.add_argument('--subsets', nargs='+', help='the size of the subset of the 10,000 tests problems to test', required=True)
    parser.add_argument("--data_path", type=str, help="the path to the directory the data to test on is stored in", required=True) #e.g. data
    args = parser.parse_args()

    ns = args.subsets # list with each item the number of examples to test over
    for n in ns:
        main_module(args.net_paths, args.data_path, number = int(n))

if __name__ == "__main__":
    main()