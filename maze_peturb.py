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

def get_prog_recall_net():
    """
    Get the progressive recall net using the hardcoded local path
    Note: here alpha = 1, for best results from a progressive net we should use alpha=0.1
    Returns:
        Torch.nn: The progressive recall net that solves mazes
    """
    net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=30) # for Paden => recall, alpha =1
    state_dict = torch.load("batch_shells_maze/outputs/mazes_ablation/training-abased-Paden/model_best.pth", map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

def get_recall_net():
    """
    Get the recall net using the hardcoded local path

    Returns:
        Torch.nn: The recall net that solves mazes
    """
    net = getattr(models, "dt_net_recall_2d")(width=128, in_channels=3, max_iters=30) # for Col => recall, alpha =0
    state_dict = torch.load("batch_shells_maze/outputs/mazes_ablation/training-algal-Collyn/model_best.pth", map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

def get_data():
    """
    Gets the 13x13 maze data from their hardcoded local file, the targets and the 2nd input
    There is a non-trivial need for just a singular input to do case analysis in the development process hence the variable "a" is returned to cover this need

    Returns:
        Torch.tensor, Torch.tensor, Torch.tensor: 
        1) The input data for 13x13 mazes
        2) the target data for 13x13 mazes
        3) the 2nd input for 13x13 data
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load("batch_reproduce_5/data/maze_data_test_13/inputs.npy")
    target = np.load("batch_reproduce_5/data/maze_data_test_13/solutions.npy")
    a = data[1]
    a = torch.from_numpy(a)
    input = a.to(device, dtype=torch.float).unsqueeze(0) #to account for batching in real net

    b = target[1]
    t = torch.from_numpy(b)
    t = t.to(device, dtype=torch.float)
    target = t.unsqueeze(0)
    return input, target, a


def convert_to_bits(output, input):
    """
    Converts the output of the net to its prediciton

    Args:
        output (tensor): the output of the net
        input (tensor): the input to the net

    Returns:
        tensor: the prediction of the net
    """
    predicted = output.clone().argmax(1)
    predicted = predicted.view(predicted.size(0), -1)
    golden_label = predicted.float() * (input.max(1)[0].view(input.size(0), -1)) #used to map the output into only paths that exist in the maze
    return golden_label

def graph_progress(arr):
    """
    Graphs the progress of the nets accuracy over a run
    All annotations to the graph are hardcoded

    Args:
        arr (tensor): the number of bits which are correct for each iteration of the net
    """
    plt.plot(arr*(100.0/1024.0), linewidth = '3.0', label = "dt_recall_prog")
    plt.title('Accuracy over time when features swapped')
    plt.xlabel('Test-Time iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    save_path = os.path.join("test_noise_outputs","test_noise_correctness.png")
    plt.savefig(save_path)

def net_out_to_bits(input,output,target, log = False, graph = False): #output from the net and the target bit string
    output = output.clone()
    corrects = torch.zeros(output.size(1))
    for i in range(output.size(1)):
        outputi = output[:, i]

        # print("outputi shape before is",outputi.shape)
        # torch.set_printoptions(profile="full")
        # print(outputi)
        golden_label = convert_to_bits(outputi, input)
        # print(golden_label)
        # torch.set_printoptions(profile="default") 
        # print("outputi shape after is",golden_label.shape)
        # print("before",target.dtype)
        target = target.view(target.size(0), -1)
        # print("after",target.dtype)
        # print("target shape is ",target.shape)
        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item()
        # if i ==50:
        #     np.save("50_maze_tensor",outputi.cpu().detach().numpy())
        # print(golden_label.dtype)
        # print(target.dtype)
        # print("corrects way is ",torch.amin(golden_label == target, dim=[0]).sum().item())
        # temp = torch.equal(golden_label,target)
        # print("new way is ",tf.reduce_sum(tf.cast(temp, tf.float32)))
        # res = golden_label.clone()
        # res[golden_label==0.0] = 1.0
        # res[golden_label==1.0] = 0.0
        # print(torch.equal(res, target))
        # sys.exit()
    
    correct = corrects.cpu().detach().numpy()
    bestind = np.argmax(correct)
    best = output[:,bestind]
    if log == True:
        stats = correct.tolist()
        save_path = os.path.join("test_noise_outputs","test_noise_stats.json")
        with open(os.path.join(save_path), "w") as fp: #taken from train_model
            json.dump(stats, fp)
    if graph == True:
        graph_progress(correct)
    print("corrects is ",correct)
    print("corrects length is ",len(correct))
    return convert_to_bits(best, input), correct[bestind] #returns the most accurate bit string and the number of bits which match with the target

cuda_avil = True if torch.cuda.is_available() else False
device = "cuda" if torch.cuda.is_available() else "cpu"
ex = torch.zeros((3, 1, 400), dtype=torch.float)
input, target, a = get_data()
net = get_prog_recall_net()
iters =300
corrects = torch.zeros(iters)

# with torch.no_grad():
#     output = net(input,iters_to_do=iters)
#     t1,t2 = net_out_to_bits(input,output,target)
#     print("predicted string is ",t1)
#     print("number of matches is ",t2)


# learning to use pytorchfi
# net = get_prog_recall_net()

# pfi_model = fault_injection(net, 
#                             batch_size,
#                             input_shape=[channels,width,height],
#                             layer_types=layer_types_input,
#                             use_cuda=True
#                             )

# b = [0]
# layer = [4]
# C = [3]
# H = [10]
# W = [10]
# err_val = [1]
# inj = pfi_model.declare_neuron_fi(batch=b, layer_num=layer, dim1=C, dim2=H, dim3=W, value=err_val) #dim 3 ignored for sums
# input, target, a = get_data()
# inj_output = inj(input)

# inj_label,matches = net_out_to_bits(input,inj_output,target)

# print("[Single Error] PytorchFI label:", inj_label)
# print("matches in inj_label is ",matches)

class custom_func(fault_injection):

    def __init__(self, model, batch_size, **kwargs):
        super().__init__(model, batch_size, **kwargs)

    def flip_all(self, module, input, output): #output is a tuple of length 1, with index 0 holding the current tensor
        # print(input)
        # print(output)
        # self.count += 1
        # if self.count == 7:
        #     sys.exit()
        # if self.count == 15:
        #     print("BEFORE:")
        #     print(output)
        #     print(len(output))
        #     print(type(output))
        #     print(output[0])
        #     print(type(output[0]))
        #     a = output[:][0]
        #     print(a.shape)
        #     for i in range(0,a.size(0)):
        #         for j in range(0,a.size(1)):
        #             if a[i][j]<0.0:
        #                 a[i][j] = 0.0
        #             else:
        #                 a[i][j] = 1.0
        #     output = (a,)
        #     print("after:")
        #     print(output)
        #     print(type(output))
        # self.count +=1
        # if (self.get_current_layer() >= self.get_total_layers()-15) and (self.get_current_layer() <= self.get_total_layers()-10):
        # if (self.get_current_layer() >= 140) and (self.get_current_layer() <= 210):
        layer_from =25 #for small GPU's use 25 or less, for larger ones we can use the full result of 50
        layer_to = 27
        if (self.get_current_layer() >= (layer_from*7)) and (self.get_current_layer() <= ((layer_to*7)+1)): # a nice observation here is the direct relation ot the size of the recurrent module
            output[:] = torch.zeros(output.shape) # puts all outputs from the layer to 0
        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            print("total layers is ",self.get_total_layers())
            self.reset_current_layer()

# PyTorchFi parameters for the maze nets
batch_size = 1
channels = 3
width = 128
height = width
layer_types_input = [torch.nn.Conv2d]

#Running on DT-Recall-Prog net
net = get_prog_recall_net()
with torch.no_grad():
    pfi_model_2 = custom_func(net, 
                            batch_size,
                            input_shape=[channels,width,height],
                            layer_types=layer_types_input,
                            use_cuda=True
                        )
    inj = pfi_model_2.declare_neuron_fi(function=pfi_model_2.flip_all)
    inj_output_prog = inj(input)

#Running on DT-Recall net
other_net = get_recall_net()
with torch.no_grad():
    pfi_model_3 = custom_func(other_net, 
                            batch_size,
                            input_shape=[channels,width,height],
                            layer_types=layer_types_input,
                            use_cuda=True
                        )
    inj = pfi_model_3.declare_neuron_fi(function=pfi_model_3.flip_all)
    inj_output_non_prog = inj(input)

def graph_two_helper(output,input,target):
    """
    Very much like convert to bits but specifically trimmed down and molded to service the graph_two method

    Args:
        output (Torch.tensor): the output from a run of a net
        input (Torch.tensor): the input to the net
        target (Torch.tensor):the target of the net

    Returns:
        numpy.array: the number of bits which were predicted correctly at each iteration of the net
    """
    output = output.clone()
    corrects = torch.zeros(output.size(1))
    for i in range(output.size(1)): # goes through each iteration
        outputi = output[:, i]
        golden_label = convert_to_bits(outputi, input)
        target = target.view(target.size(0), -1)
        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item() # counts the number that are the same
    correct = corrects.cpu().detach().numpy()
    return correct

def graph_two(input,output_1,output_2,target): 
    """
    Converts the outputs of two net runs and graphs their accuracy, assuming they were run on the same input
    The annotations on the graph are all hardcoded

    Args:
        input (Torch.tensor): the input to the nets
        output_1 (Torch.tensor): the output from the DT-Recall-Prog net
        output_2 (Torch.tensor): the output from the DT-Prog net
        target (Torch.tensor): the target of the nets
    """
    data_1 = graph_two_helper(output_1,input,target)
    data_2 = graph_two_helper(output_2,input,target)
    plt.plot(data_1*(100.0/1024.0), linewidth = '3.0', label = "DT-Recall-Prog")
    plt.plot(data_2*(100.0/1024.0), linewidth = '3.0', label = "DT-Recall")
    plt.title('Accuracy over time when features swapped')
    plt.xlabel('Test-Time iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower right")
    save_path = os.path.join("test_noise_outputs","repo_7.png")
    plt.savefig(save_path)

graph_two(input,inj_output_prog,inj_output_non_prog,target)