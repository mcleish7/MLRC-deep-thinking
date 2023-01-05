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

def get_net(device, type="prog"):
    """
    Returns the DT recall (progressive) network in evaluation mode

    Args:
        type (str, optional): Set to prog if want the progressive recall network. Defaults to "prog".
        device (str): the device to store the network on

    Returns:
        torch.nn: the neural net
    """
    if type == "prog": 
        name = "enraged-Jojo" # Jojo => recall, alpha =1 
    else:
        name = "peeling-Betzaida" # Betz => recall, alpha =0
    file = f"batch_shells_sums/outputs/prefix_sums_ablation/training-{name}/model_best.pth"

    net = getattr(models, "dt_net_recall_1d")(width=400, in_channels=3, max_iters=30)
    state_dict = torch.load(file, map_location=device)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(state_dict["net"])
    net.eval()
    return net

def get_data(device):
    """
    Gets bit strings of length 48 from the local file and augments them to be the same how the DataLoader would input them
    Args:
        device (str): the device to store the output tensors on
    Returns:
        input, target (tensor,tensor): the input and taget datasets as tensors on the device passed in
    """
    data = torch.load("batch_shells_sums/data/prefix_sums_data/48_data.pth").unsqueeze(1) - 0.5
    target = torch.load("batch_shells_sums/data/prefix_sums_data/48_targets.pth")
    input = data.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    return input, target
    
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

def convert_to_bits(input):
    """Convert the input string to a bits stored in an array

    Args:
        input (tensor): the  array to convert

    Returns:
        golden_label (numpy array): the input string to a bits stored in an array
    """
    predicted = input.clone().argmax(1)
    golden_label = predicted.view(predicted.size(0), -1)
    return golden_label

def net_out_to_bits(output,target, log = False, graph = False): #output from the net and the target bit string
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
        if (self.get_current_layer() < 408) and (self.get_current_layer() >= 400):
            j = self.j #between 0 and 48
            for i in range(0,output.size(1)):
                if output[0,i,j] > 0.0:
                    output[0,0,j] = -20.0 #means that 0 will not be returned as it is less than the 1 index, i.e. a bitflip
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
    bestind = np.argmax(correct[50:]) #only looks for maximum after peturbation
    return bestind 

def graph_time(arr1,arr2):
    """
    Saves a line graph of the time to recover from a petubration for the two input arrays
    As a run for one Net takes 2 days, this method is used manuallly after collecting the data from the file it is stored in at run time

    Args:
        arr1 (list): list of the times to recover of the Recall net being used
        arr2 (list): list of the times to recover of the Recall Progressive net being used
    """
    plt.clf()
    plt.plot(arr1, linewidth = '3.0', label = "Recall")
    plt.plot(arr2, linewidth = '3.0', label = "Recall Prog")
    plt.title('Iterations to recover from a single bit perturbation')
    plt.xlabel("Index to be flipped")
    plt.ylabel("Number of iterations to recover")
    plt.legend(loc="upper right")
    plt.yticks([0,26,5,10,25,15,20])
    save_path = os.path.join("test_time","test_time_correctness_2.png")
    plt.savefig(save_path)

def main():
    """
    Runs the peturbation with the input commmand line peratmeter for which net is selected
    """
    parser = argparse.ArgumentParser(description="Time parser")
    parser.add_argument("--which_net", type=str, default="prog", help="choose between prog or non-prog, defaults to prog")
    args = parser.parse_args()

    os.chdir("/dcs/large/u2004277/deep-thinking/") # changing back to the top directory as this method can be called from bash scripts in other directories
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # parameters for pytorchfi model
    batch_size = 500
    channels = 1
    width = 400
    layer_types_input = [torch.nn.Conv1d]

    net = get_net(device, type = args.which_net)
    print("now going into loop")
    inputs,targets = get_data(device)
    with torch.no_grad(): # we are evaluating so no grad needed
        time = [] # store for the averaged values
        for index in range(0,40): # index of which bit is to be changed
            average = []
            for i in range(0,inputs.size(0)):
                input = inputs[i].unsqueeze(0) # have to unsqueeze to simulate batches
                target = targets[i].unsqueeze(0) 
                pfi_model_2 = custom_func(index,net, 
                                        batch_size,
                                        input_shape=[channels,width],
                                        layer_types=layer_types_input,
                                        use_cuda=True
                                    )

                inj = pfi_model_2.declare_neuron_fi(function=pfi_model_2.flip_all) # run the model, the number of iterations is controlled in by the default value in the forward call of each model
                inj_output = inj(input)
                average.append(count_to_correct(inj_output,target))
            mean = sum(average) / len(average)
            time.append(mean)
            name = f"time_list_tracker_{args.which_net}.txt"
            file_path = os.path.join("test_time",name)
            with open(file_path, 'w+') as f: # storing the data as we do not expect reach the end of the loop in the set runtime
                f.write(f"for index: {index} the time array is {time}")

if __name__ == "__main__":
    main()

#All of the output data is stored in text files, I have moved it to here to graph and so it can be seen in its raw format
# Runs oftern take more than two days, hence the split in the lists

# Betz data
#for index: 36 the time array is 
l = [18.1349, 19.907, 24.0738, 25.3438, 25.8389, 25.7408, 25.3065, 24.7848, 24.2047, 23.6946, 23.3269, 22.7508, 22.3547, 21.8837, 21.4098, 20.8964, 20.5066, 20.0099, 19.4742, 18.9723, 18.4464, 18.0089, 17.5026, 16.982, 16.5222, 15.9724, 15.5709, 15.006, 14.4962, 14.0435, 13.5449, 13.0145, 12.4881, 12.0132, 11.5107, 10.98, 10.5114]
# print("l is length ",len(l))
# for index: 39 the time array is  
l1 = [9.9924, 9.4904, 8.9798]
#redoing the first few:
#for index: 9 the time array is [18.1349, 19.907, 24.0738, 25.3438, 25.8389, 25.7408, 25.3065, 24.7848, 24.2047, 23.6946]

# Jojo data
# for index: 39 the time array is 
j = [6.3722, 6.0566, 5.6513]
#for index: 36 the time array is 
j1 = [17.8872, 19.4591, 19.941, 19.912, 19.8543, 19.2299, 18.8148, 18.4023, 18.0458, 17.6382, 17.1686, 16.7601, 16.27, 15.8989, 15.4843, 15.1598, 14.6714, 14.3212, 13.8789, 13.4322, 13.014, 12.6209, 12.1961, 11.7285, 11.3529, 10.9505, 10.512, 10.1154, 9.707, 9.2398, 8.893, 8.5136, 8.1195, 7.723, 7.3973, 7.024, 6.6765]

betz = l+l1
jojo = j1+j
# graph_time(betz,jojo)

# prog
# for index: 36 the time array is 
prog1 = [16.1856, 15.5111, 16.3486, 15.4524, 15.5068, 15.0618, 14.9197, 14.4535, 14.2271, 13.9523, 13.5131, 13.2493, 12.7945, 12.5634, 12.1517, 11.9928, 11.3922, 11.1332, 10.9404, 10.5358, 10.1458, 9.7926, 9.6382, 9.1171, 8.831, 8.5497, 8.1151, 7.9084, 7.5659, 7.1674, 6.9791, 6.665, 6.2769, 5.9907, 5.7131, 5.4087, 5.1498]
# for index: 39 the time array is 
prog2 = [5.1498, 4.9016, 4.5934, 4.2871]
prog = prog1+prog2

#non prog
# for index: 36 the time array is 
nprog1 =[13.1338, 14.0909, 16.5159, 18.9491, 18.6721, 18.2342, 17.807, 17.2267, 16.8972, 16.5704, 16.2457, 16.0122, 15.6696, 15.3622, 15.1253, 14.7834, 14.5677, 14.2961, 13.9309, 13.6141, 13.2166, 13.0427, 12.7052, 12.3991, 12.0407, 11.7189, 11.4802, 11.0965, 10.7525, 10.444, 10.0743, 9.6769, 9.2973, 8.9339, 8.6153, 8.2189, 7.8931]
# for index: 39 the time array is 
nprog2 = [7.8931, 7.4263, 7.0412, 6.6338]
nprog = nprog1+nprog2

graph_time(nprog,prog)