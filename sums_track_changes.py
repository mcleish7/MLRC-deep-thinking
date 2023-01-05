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
    input = data.to(device, dtype=torch.float) #to account for batching in real net
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
            # change.append(str(outi)+", "+str(first_diff(gl,prev))+", "+str(num_diff(gl,prev))+", "+correct) # string (index after 50, index of first difference, number of differences, if the output is correct)
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
        if (self.get_current_layer() < 408) and (self.get_current_layer() >= 400):
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
        #when this prints it is a very rare case where the net recovers in one step
        print("one step recovery")
        print(input)
        print(target)
        print(num_diffences)
        print(correct)
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Time parser")
    parser.add_argument("--which_net", type=str, default="prog", help="choose between prog or non-prog, defaults to prog")
    args = parser.parse_args()

    os.chdir("/dcs/large/u2004277/deep-thinking/")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 500
    channels = 1
    width = 400
    layer_types_input = [torch.nn.Conv1d]

    net = get_net(device, args.which_net)
    inputs,targets = get_data(device)

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
            file_path = os.path.join("test_time","time_track_out.txt")
            with open(file_path, 'r+') as f: # storing the data as we do not expect reach the end of the loop in the set runtime
                f.write(f"for index: {index} density  list is {average_density}\n")
            
# if __name__ == "__main__":
#     main()

#All of the output data is stored in text files, I have moved it to here to graph and so it can be seen in its raw format
# Runs oftern take more than two days, hence the split in the lists

#testing for 1000 iters on jojo => recall, alpha =1 (less hook)
#for index: 13 density  list is 
l = [8.350531192150143, 8.836733070870311, 8.15026908322514, 8.169314109471728, 8.120657781738904, 7.8387408047366165, 7.871288398636948, 7.704663253001619, 7.7222812864561, 7.638716922882653, 7.529145282420763, 7.4196367780568195, 7.293277663501195, 7.294934048231607]

#testing Betz => recall, alpha =0 10,000 for 38 iters
# for index: 38 density  list is 
l1 = [2.894564798436663, 2.742658473504928, 2.9500032091398998, 2.9716900719955452, 2.9626442698783673, 2.9574257539860693, 2.9375175098130635, 2.920897184747449, 2.9207207787799887, 2.906564512562613, 2.9060831001510286, 2.898443841835443, 2.892349929487068, 2.8898419706793708, 2.877596519273487, 2.872609737382494, 2.8692976791230196, 2.863144291075841, 2.847519525391978, 2.8347488911874694, 2.8278018865221695, 2.818073908160802, 2.799248385758665, 2.798484213955967, 2.784911057895057, 2.7677313715197864, 2.759365109638811, 2.7410102599541686, 2.734705311159419, 2.7116538031249826, 2.6936397658142583, 2.667094037196456, 2.6370377826340183, 2.6112972957597895, 2.588518300310813, 2.5570871450771566, 2.508867230824737, 2.467657775280267, 2.415364841269837]

# for index: 37 density  list is 10000 iters on jojo => recall, alpha =1 (less hook)
l2 = [8.354512728221037, 8.775918890788645, 8.09659367925775, 8.0967221853939, 8.094584859871922, 7.9408257761864585, 7.876619508529945, 7.767930578984392, 7.703758035898553, 7.588311421170889, 7.500024388210057, 7.423071916345686, 7.2917123906094305, 7.2349862306174675, 7.127819093089241, 7.078964323426383, 6.894311736560213, 6.8227387977986975, 6.7258364935176465, 6.661864411857927, 6.537369552611998, 6.424532856816396, 6.343774872921189, 6.217911686573217, 6.185617925407933, 6.065921009888788, 5.961891254578772, 5.868166186868691, 5.769126171051171, 5.638341159673668, 5.557623549228581, 5.444445247807738, 5.340174433621909, 5.199739285714301, 5.069919960317445, 4.906463611111129, 4.743992182539701, 4.593787261904766]

# time to recover data from test_time file:

# Betz data
#for index: 36 the time array is 
ttl = [18.1349, 19.907, 24.0738, 25.3438, 25.8389, 25.7408, 25.3065, 24.7848, 24.2047, 23.6946, 23.3269, 22.7508, 22.3547, 21.8837, 21.4098, 20.8964, 20.5066, 20.0099, 19.4742, 18.9723, 18.4464, 18.0089, 17.5026, 16.982, 16.5222, 15.9724, 15.5709, 15.006, 14.4962, 14.0435, 13.5449, 13.0145, 12.4881, 12.0132, 11.5107, 10.98, 10.5114]

# Jojo data
#for index: 36 the time array is 
ttj = [17.8872, 19.4591, 19.941, 19.912, 19.8543, 19.2299, 18.8148, 18.4023, 18.0458, 17.6382, 17.1686, 16.7601, 16.27, 15.8989, 15.4843, 15.1598, 14.6714, 14.3212, 13.8789, 13.4322, 13.014, 12.6209, 12.1961, 11.7285, 11.3529, 10.9505, 10.512, 10.1154, 9.707, 9.2398, 8.893, 8.5136, 8.1195, 7.723, 7.3973, 7.024, 6.6765]

mul1 = []
mul2 = []
i=0
while (i<len(l1)) and (i<len(ttl)):
    mul1.append(l1[i]*ttl[i])
    i+=1
i=0
while (i<len(l2)) and (i<len(ttj)):
    mul2.append(l2[i]*ttj[i])
    i+=1
graph_time(mul1,mul2,gtype="mul")
graph_time(l1,l2)