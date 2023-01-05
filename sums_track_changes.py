import pytorchfi as fi
from pytorchfi.core import fault_injection
import torch
import sys
import os 
from deepthinking import models as models
from deepthinking import utils as dt
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse

def get_net(device, path):
    """
    Returns the DT recall (progressive) network in evaluation mode

    Args:
        type (str): the file path to the network
        device (str): the device to store the network on

    Returns:
        torch.nn: the neural net
    """
    net = getattr(models, "dt_net_recall_1d")(width=400, in_channels=3, max_iters=30)
    state_dict = torch.load(path, map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

def get_data(device, path, size):
    """
    Gets bit strings of length 48 from the local file and augments them to be the same how the DataLoader would input them
    Args:
        device (str): the device to store the output tensors on
        path (str): the path to the directoryt that the data is stored in
    Returns:
        input, target (tensor,tensor): the input and taget datasets as tensors on the device passed in
    """
    data_path = path + "/" + str(size) + "_data.pth"
    target_path = path + "/" + str(size) + "_targets.pth"
    data = torch.load(data_path).unsqueeze(1) - 0.5 #to account for batching and normalisation in real net
    target = torch.load(target_path)
    input = data.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    return input, target

def convert_to_bits(input):
    """Convert the input string to a bits stored in an array

    Args:
        input (tensor): the numpy array to convert

    Returns:
        golden_label (numpy array): the input string to a bits stored in an array
    """
    predicted = input.clone().argmax(1)
    golden_label = predicted.view(predicted.size(0), -1)
    return golden_label

def graph_progress(arr):
    """
    Graph the input array as a line graph
    This method is only called in the net_out_to_bits method so the title and save path are fixed
    Args:
        arr (numpy.array or list): the array/list to be graphed
    """
    plt.plot(arr)
    plt.title('Values of correct array')
    save_path = os.path.join("test_noise_outputs","test_noise_correctness.png")
    plt.savefig(save_path)

def net_out_to_bits(output,target, log = False, graph = False): 
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
        golden_label = convert_to_bits(outputi)
        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item() 
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
    return convert_to_bits(best), correct[bestind] #returns the most accurate bit string and the number of bits which match with the target

def first_diff(arr1,arr2):
    """
    Finds the index of the first difference in the two input arrays
    Args:
        arr1 (list): first list to compare
        arr2 (list): second list to compare

    Returns:
        int: the index of the first difference in the two input arrays
    """
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return i
    return -1

def same(arr1,arr2):
    """
    Checks if the two input arrays are elemnt wise identical, therefore identical overall
    Args:
        arr1 (list): first list to compare
        arr2 (list): second list to compare

    Returns:
        bool: whether the two input arrays are identical 
    """
    if len(arr1)!=len(arr2):
        return False
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    return True

def num_diff(arr1,arr2):
    """
    Finds the number of differences in elements of the same index between the two inputs
    Note: assumes two arrays are of same length
    Args:
        arr1 (list): first list to compare
        arr2 (list): second list to compare

    Returns:
        int: the number of differences between the inputs
    """
    count = 0
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            count += 1
    return count

def track_after_peturb(output,target):
    """
    Tracks 

    Args:
        output (tensor): output of run of neural net
        target (tensor): _description_

    Returns:
        list, list, list:
        1) the index of the first difference in output between (50+i)th and (50+i-1)th iteration
        2) the number of differences in output between (50+i)th and (50+i-1)th iteration
        3) a boolean list to check if the (50+i)th output is correct
    """
    target = target.cpu().detach().numpy().astype(int)[0] # Takes the target to a numpy array
    out = [] # store for the output of each iteration
    first_difference = [] # store for the index of the first difference in output between (50+i)th and (50+i-1)th iteration
    num_diffences = [] # store for the number of differences in output between (50+i)th and (50+i-1)th iteration
    correct = [] # a boolean list to check if the (50+i)th output is correct
    for i in range(output.size(1)): # for each iteration in the run
        outputi = output[:, i]
        golden_label = convert_to_bits(outputi)
        if i == 50: # have to add data at i=50 to out array to be able to read it in the next iteration
            out.append(golden_label.cpu().detach().numpy().astype(int)[0])
        if (i>50) and (i<100): # we only look up to 100th iteration as normally the nets recover in a maximum of 26 iterations
            outi=i-50
            gl = golden_label.cpu().detach().numpy().astype(int)[0]
            out.append(gl)
            prev = out[outi-1]
            match = same(gl,target)
            first_difference.append(first_diff(gl,prev))
            num_diffences.append(num_diff(gl,prev))
            correct.append(match)
    return first_difference, num_diffences, correct

class custom_func(fault_injection):
    """
    Custom peturbation class to peturb the input of the prefix sums NNs at the 50 iteration
    Inherits:
        fault_injection (class): pytorchfi.core.fault_injection

    Attributes
    ----------
    j : int
        the bit to be peturbed in the string

    Methods
    -------
    flip_all(self, module, input, output)
        called at each iteration of the NN to flip the specified bit
    """
    j = 0 
    def __init__(self, in_j,model, batch_size, **kwargs):
        """constructor for custom_func

        Args:
            in_j (int): the bit to be peturbed
            model (_type_): the network
            batch_size (int): batch size for the network
        """
        super().__init__(model, batch_size, **kwargs)
        self.j = in_j

    def flip_all(self, module, input, output): 
        """
        Called in each iteration of the NN automatically using the PyTorchFI framework so we can alter the output of that layer

        Args:
            module (specified in layer_types super class varibale): the type of the layer
            input (tensor): the input to the layer
            output (tensor): the output of the layer
        """
        if (self.get_current_layer() < 408) and (self.get_current_layer() >= 400): # there are 8 layers in the recurrent module so we apply the perutbation at each layer for the 50th application of the module
            j = self.j #between 0 and 48
            for i in range(0,output.size(1)):
                if output[0,i,j] > 0.0:
                    output[0,0,j] = -20.0 #means that 0 will not be returned as it is less than the 1 index
                else:
                    output[0,1,j] = 20.0

        self.updateLayer()
        if self.get_current_layer() >= self.get_total_layers():
            self.reset_current_layer()

def count_to_correct(output,target):
    """
    Counts the number of iterations until the network finds the correct output after peturbation

    Args:
        output (tensor): the output of the NN run
        target (tensor): the target of the run

    Returns:
        int: the number of iterations it took to recover from peturbation
    """
    output = output.clone()
    corrects = torch.zeros(output.size(1))
    for i in range(output.size(1)):
        outputi = output[:, i]
        golden_label = convert_to_bits(outputi)
        corrects[i] += torch.amin(golden_label == target, dim=[0]).sum().item() 
    correct = corrects.cpu().detach().numpy()
    bestind = np.argmax(correct[50:])
    return bestind #returns the most accurate bit string and the number of bits which match with the target

def graph_time(arr1,arr2, gtype=""):
    """
    Saves a line graph of the time to recover from a petubration for the two input arrays
    As a run for one Net takes 2 days, this method is used manuallly after collecting the data from the file it is stored in at run time

    Args:
        arr1 (list): list of the times to recover of the Recall net being used
        arr2 (list): list of the times to recover of the Recall Progressive net being used
        gtype (string, optional): changes the labels on the graph, default = ""
    """
    plt.clf()
    plt.plot(arr1, linewidth = '2.0', label = "Recall")
    plt.plot(arr2, linewidth = '2.0', label = "Recall Prog")
    plt.legend(loc="upper right")
    if gtype == "mul":
        plt.title('Number of changes after a one bit perturbation')
        plt.xlabel("Index to be flipped")
        plt.ylabel("Number of changes made to the bit string before recovering")
        save_path = os.path.join("test_time","test_time_track_mul.png")
    else:
        plt.title('Average number of changes per epoch after a one bit perturbation')
        plt.xlabel("Index to be flipped")
        plt.ylabel("Average number of changes made per epoch")
        save_path = os.path.join("test_time","test_time_track.png")
    plt.savefig(save_path)

def density(num_diffences, correct, input, target):
    """
    Works out the average number of changesper iteration to the bit string after peturbation 

    Args:
        num_diffences (list): list of the number of differences in output between (50+i)th and (50+i-1)th iteration
        correct (list): the boolean list for if the net's prediction is equal to the target to the (50+i)th iteration
        input (tensor): the input of the net run
        target (tensor): the target of the net run

    Returns:
        float: _description_
    """
    count = 0
    total = 0
    for i in range(len(correct)):
        if correct[i] == False:
            count +=1
            total += num_diffences[i]
    if count >0:
        return total/count
    else:
        #when this prints it is a very rare case where the net recovers in one step from the peturbation
        print("one step recovery")
        print(input)
        print(target)
        print(num_diffences)
        print(correct)
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Time parser")
    # e.g."outputs/{batch_name}/training-{model_name}/model_best.pth"
    parser.add_argument("--net_path", type=str, help="the path to the prefix sums model to undego testing", required=True)
    # when the prefix sums models were trained the data was automatically downloaded into a "data" file, we want the path to this file including the "data" part
    # e.g. data
    parser.add_argument("--data_path", type=str, help="the path to the directory the data to test on is stored in", required=True)
    parser.add_argument("--size", type=str, help="the size of the bit strings to collect from the data_path directory", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 500
    channels = 1
    width = 400
    layer_types_input = [torch.nn.Conv1d]

    net = get_net(device, args.net_path)
    inputs,targets = get_data(device, args.data_path, args.size)

    with torch.no_grad(): # we are evaluating so no grad needed
        average_density = [] # store for the averaged values
        for index in range(0,40): # index of which bit is to be changed
            density_list = []
            for i in range(0,inputs.size(0)):
                input = inputs[i].unsqueeze(0) # have to unsqueeze to simulate batches
                target = targets[i].unsqueeze(0)

                pfi_model_2 = custom_func(index,net, 
                                        batch_size,
                                        input_shape=[channels,width],
                                        layer_types=layer_types_input,
                                        use_cuda=True
                                    )

                inj = pfi_model_2.declare_neuron_fi(function=pfi_model_2.flip_all)
                inj_output = inj(input)
                first_difference, num_diffences, correct = track_after_peturb(inj_output,target)
                density_list.append(density(num_diffences, correct, input, target))
            mean = sum(density_list) / len(density_list)
            average_density.append(mean)
            file_path = "time_track_out.txt"
            with open(file_path, 'w+') as f: # storing the data as we do not expect reach the end of the loop in the set runtime
                f.write(f"for index: {index} density  list is {average_density}\n")
            
if __name__ == "__main__":
    main()

# Paste the arrays in from the files they are written to and pass them to the graph results method to plot
# Remember: comment out lines 361 and 362 which call main when graphing to save time

def graph_results(r,rp,tr,trp):
    """
    Takes the 4 arrays we generate (time to recover and work done to recover for both prog and non-prog) 
    and plots both the average work done and total work done by multiplying the two arrays

    Args:
        r (list): The Recall models average number of changes done after a perurbation list created by sums_track_changes.py
        rp (list): The Recall-Prog models average number of changes done after a perurbation list created by sums_track_changes.py
        tr (list): The Recall models average time to recover after a peturbation list created by sums_peturb.py
        trp (list): The Recall-Progs models average time to recover after a peturbation list created by sums_peturb.py
    """
    mul1 = []
    mul2 = []
    i=0
    while (i<len(r)) and (i<len(tr)):
        mul1.append(r[i]*tr[i])
        i+=1
    i=0
    while (i<len(rp)) and (i<len(trp)):
        mul2.append(rp[i]*trp[i])
        i+=1
    graph_time(mul1,mul2,gtype="mul")
    graph_time(r,rp)