'''
Test script for the Social LSTM model

Author: Di Wu
Date: 21st Jan 2018
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pickle
import argparse
import time

import torch
from torch.autograd import Variable
from synthia_utils import Synthia_DataLoader
from st_graph import ST_GRAPH
from model import SocialLSTM
from helper import getCoef, sample_gaussian_2d, compute_edges, get_mean_error, get_final_error
from criterion import Gaussian2DLikelihood, Gaussian2DLikelihoodInference
from grid import getSequenceGridMask, getGridMaskInference


def main():

    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=15,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=8,
                        help='Predicted length of the trajectory')
    # Test dataset
    parser.add_argument('--test_dataset', type=int, default=0,
                        help='Dataset to be tested on')

    # Model to be loaded
    parser.add_argument('--epoch', type=int, default=4,
                        help='Epoch of model to be loaded')

    # Parse the parameters
    sample_args = parser.parse_args()

    # Save directory
    save_directory = '../save/' + str(sample_args.test_dataset) + '/'

    # Define the path for the config file for saved args
    with open(os.path.join(save_directory, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    # Initialize net
    net = SocialLSTM(saved_args, True)
    net.cuda()

    # Get the checkpoint path
    checkpoint_path = os.path.join(save_directory, 'social_lstm_model_'+str(sample_args.epoch)+'.tar')
    # checkpoint_path = os.path.join(save_directory, 'srnn_model.tar')
    if os.path.isfile(checkpoint_path):
        print('Loading checkpoint')
        checkpoint = torch.load(checkpoint_path)
        # model_iteration = checkpoint['iteration']
        model_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        print('Loaded checkpoint at epoch', model_epoch)

    # Test dataset
    dataset = [sample_args.test_dataset]

    # Create the DataLoader object
    dataloader = Synthia_DataLoader(
        data_root='../data/synthia/SYNTHIA-SEQS-01',
        img_dir='/media/samsumg_1tb/synthia/Datasets',
        batch_size=1,
        seq_length=sample_args.pred_length + sample_args.obs_length,
        datasets=dataset,
        forcePreProcess=False,
        infer=True)

    dataloader.reset_batch_pointer()

    # Construct the ST-graph object
    stgraph = ST_GRAPH(1, sample_args.pred_length + sample_args.obs_length, saved_args.dataset_dim)

    results = []

    # Variable to maintain total error
    total_error = 0
    final_error = 0
    errCenter = 0
    final_error_errCenter = 0

    # For each batch
    for batch in range(dataloader.num_batches):
        start = time.time()

        # Get data
        x, _, _ = dataloader.next_batch()

        # Get the sequence
        x_seq = x[0]

        # Construct ST graph
        stgraph.readGraph(x)

        # Get nodes and nodesPresent
        nodes, _, nodesPresent, _, retNodePresentName = stgraph.getSequence(0)

        # Get the grid masks for the sequence
        grid_seq = getSequenceGridMask(x_seq, saved_args.neighborhood_size, saved_args.grid_size, retNodePresentName)

        nodes = Variable(torch.from_numpy(nodes).float(), volatile=True).cuda()

        # Extract the observed part of the trajectories
        obs_nodes, obs_nodesPresent, obs_grid = nodes[:sample_args.obs_length], nodesPresent[:sample_args.obs_length], grid_seq[:sample_args.obs_length]

        # The sample function
        ret_nodes = sample(obs_nodes, obs_nodesPresent, obs_grid, sample_args, net, saved_args)

        # Record the mean and final displacement error
        total_error_temp, errCenter_temp = get_mean_error(ret_nodes[sample_args.obs_length:].data, nodes[sample_args.obs_length:].data,
                                      nodesPresent[sample_args.obs_length-1], nodesPresent[sample_args.obs_length:],
                                      saved_args.dataset_dim)

        final_error_temp, final_error_errCenter_temp = get_final_error(ret_nodes[sample_args.obs_length:].data, nodes[sample_args.obs_length:].data,
                                       nodesPresent[sample_args.obs_length-1], nodesPresent[sample_args.obs_length:],
                                       saved_args.dataset_dim)

        total_error += total_error_temp
        errCenter += errCenter_temp
        final_error += final_error_temp
        final_error_errCenter += final_error_errCenter_temp
        end = time.time()

        print('Processed trajectory number : ', batch, 'out of', dataloader.num_batches, 'trajectories in time', end - start)

        results.append((nodes.data.cpu().numpy(), ret_nodes.data.cpu().numpy(), nodesPresent, sample_args.obs_length))

        # Reset the ST graph
        stgraph.reset()

    print('Total mean error of the model is ', total_error / dataloader.num_batches)
    print('Total final error of the model is ', errCenter / dataloader.num_batches)
    print('Total center pixel of the model is ', total_error / dataloader.num_batches)
    print('Total final center pixel error of the model is ', final_error_errCenter / dataloader.num_batches)

    print('Saving results')
    with open(os.path.join(save_directory, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    """
    Total mean error of the model is  0.009818906374781172
    Total final error of the model is  56.0193065963
    Total center pixel of the model is  0.009818906374781172
l
    """


def sample(nodes, nodesPresent, grid, args, net, saved_args):
    '''
    The sample function
    params:
    nodes: Input positions
    nodesPresent: Peds present in each frame
    args: arguments
    net: The model
    true_nodes: True positions
    true_nodesPresent: The true peds present in each frame
    true_grid: The true grid masks
    saved_args: Training arguments
    dimensions: The dimensions of the dataset
    '''
    # Number of peds in the sequence
    dimensions = saved_args.dataset_dim
    numNodes = nodes.size()[1]

    # Construct variables for hidden and cell states
    hidden_states = Variable(torch.zeros(numNodes, net.args.rnn_size), volatile=True).cuda()
    cell_states = Variable(torch.zeros(numNodes, net.args.rnn_size), volatile=True).cuda()

    # For the observed part of the trajectory
    for tstep in range(args.obs_length-1):
        # Do a forward prop
        out_obs, hidden_states, cell_states = net(nodes[tstep].view(1, numNodes, 2), [grid[tstep]], [nodesPresent[tstep]], hidden_states, cell_states)
        # loss_obs = Gaussian2DLikelihood(out_obs, nodes[tstep+1].view(1, numNodes, 2), [nodesPresent[tstep+1]])

    # Initialize the return data structure
    ret_nodes = Variable(torch.zeros(args.obs_length+args.pred_length, numNodes, 2), volatile=True).cuda()
    ret_nodes[:args.obs_length, :, :] = nodes.clone()

    # Last seen grid
    prev_grid = grid[-1].clone()

    # For the predicted part of the trajectory
    for tstep in range(args.obs_length-1, args.pred_length + args.obs_length - 1):
        # Do a forward prop
        outputs, hidden_states, cell_states = net(ret_nodes[tstep].view(1, numNodes, 2), [prev_grid], [nodesPresent[args.obs_length-1]], hidden_states, cell_states)
        # loss_pred = Gaussian2DLikelihoodInference(outputs, true_nodes[tstep+1].view(1, numNodes, 2), nodesPresent[args.obs_length-1], [true_nodesPresent[tstep+1]])

        # Extract the mean, std and corr of the bivariate Gaussian
        mux, muy, sx, sy, corr = getCoef(outputs)
        # Sample from the bivariate Gaussian
        next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, nodesPresent[args.obs_length-1])

        # Store the predicted position
        ret_nodes[tstep + 1, :, 0] = next_x
        ret_nodes[tstep + 1, :, 1] = next_y

        # List of nodes at the last time-step (assuming they exist until the end)
        list_of_nodes = Variable(torch.LongTensor(nodesPresent[args.obs_length-1]), volatile=True).cuda()
        # Get their predicted positions
        current_nodes = torch.index_select(ret_nodes[tstep+1], 0, list_of_nodes)

        # Compute the new grid masks with the predicted positions
        prev_grid = getGridMaskInference(current_nodes.data.cpu().numpy(), dimensions, saved_args.neighborhood_size, saved_args.grid_size)
        prev_grid = Variable(torch.from_numpy(prev_grid).float(), volatile=True).cuda()

    return ret_nodes


if __name__ == '__main__':
    main()
