import os
import numpy as np
import argparse
from synthia_utils import Synthia_DataLoader
import pickle
import time
from st_graph import ST_GRAPH


def kalman_xy(x, P, measurement, R, motion=np.matrix('0. 0. 0. 0.').T, Q = np.matrix(np.eye(4))):
    """
    Parameters:
    x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    measurement: observed position
    R: measurement noise
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    """
    return kalman(x, P, measurement, R, motion, Q,
                  F = np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      '''),
                  H = np.matrix('''
                      1. 0. 0. 0.;
                      0. 1. 0. 0.'''))


def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)

    See also http://en.wikipedia.org/wiki/Kalman_filter

    This version of kalman can be applied to many different situations by
    appropriately defining F and H
    '''
    # UPDATE x, P based on measurement m
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    x = x + K*y
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K*H)*P

    # PREDICT x, P based on motion
    x = F*x + motion
    P = F*P*F.T + Q

    return x, P


def ssd_2d(x, y, dataset_dim):
    s = 0
    for i in range(2):
        s += ((x[i] - y[i]) * dataset_dim[i]) ** 2
    return np.sqrt(s)


def get_mean_error(ret_nodes, nodes, dataset_dim):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent : A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    Returns
    =======

    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.shape[0]
    error = np.zeros(pred_length)
    errCenter = np.zeros(pred_length)

    # We consider only the first main car
    for tstep in range(pred_length):
        pred_pos = ret_nodes[tstep, :]
        true_pos = nodes[tstep, 0, :]
        error[tstep] = np.sqrt(np.sum((pred_pos - true_pos)**2))
        errCenter[tstep] = ssd_2d(true_pos, pred_pos, dataset_dim)
    return np.mean(error), np.mean(errCenter)


def get_final_error(ret_nodes, nodes, dataset_dim):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent : A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    Returns
    =======

    Error : Mean final euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.shape[0]
    # We consider only the first main car
    # Last time-step
    tstep = pred_length - 1

    pred_pos = ret_nodes[tstep, :]
    true_pos = nodes[tstep, 0, :]
    error = np.sqrt(np.sum((pred_pos - true_pos)**2))
    errCenter = ssd_2d(true_pos, pred_pos, dataset_dim)

    return error, errCenter


def kalman_sample(obs_nodes, saved_args):

    x = np.matrix('0. 0. 0. 0.').T
    P = np.matrix(np.eye(4))* 1e4 # initial uncertainty

    observed_x = np.array([d[0] for d in obs_nodes])
    observed_y = np.array([d[1] for d in obs_nodes])
    result = []
    R = 1 ** 2
    for meas in zip(observed_x, observed_y):
        x, P = kalman_xy(x, P, meas, R)
        result.append((x[:2]).tolist())

    result_new = (result[-1][0][0], result[-1][1][0])
    valid_pred = np.zeros(shape=(saved_args.pred_length, saved_args.input_size))

    valid_pred[0, 0] = result[-1][0][0]
    valid_pred[0, 1] = result[-1][1][0]
    for t in range(1, saved_args.pred_length):
        valid_pred[t, 0] = result_new[0] + x.item(2)
        valid_pred[t, 1] = result_new[1] + x.item(3)
        x, P = kalman_xy(x, P, (valid_pred[t, 0], valid_pred[t, 1]), R)

    return valid_pred


def main():

    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=15,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=8,
                        help='Predicted length of the trajectory')
    # Test dataset
    parser.add_argument('--test_dataset', type=int, default=1,
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

        # Construct ST graph
        stgraph.readGraph(x)

        # Get nodes and nodesPresent
        nodes, _, nodesPresent, _, retNodePresentName = stgraph.getSequence(0)

        # Extract the observed part of the trajectories
        obs_nodes, obs_nodesPresent = nodes[:sample_args.obs_length], nodesPresent[:sample_args.obs_length]

        # The sample function, we only consider the first car
        ret_nodes = kalman_sample(obs_nodes[:, 0], saved_args)

        # Record the mean and final displacement error
        total_error_temp, errCenter_temp = get_mean_error(ret_nodes, nodes[sample_args.obs_length:], saved_args.dataset_dim)

        final_error_temp, final_error_errCenter_temp = get_final_error(ret_nodes, nodes[sample_args.obs_length:], saved_args.dataset_dim)

        total_error += total_error_temp
        errCenter += errCenter_temp
        final_error += final_error_temp
        final_error_errCenter += final_error_errCenter_temp
        end = time.time()

        print('Processed trajectory number : ', batch, 'out of', dataloader.num_batches, 'trajectories in time', end - start)

        # Reset the ST graph
        stgraph.reset()

    print('Total mean error of the model is ', total_error / dataloader.num_batches)
    print('Total final error of the model is ', errCenter / dataloader.num_batches)
    print('Total center pixel of the model is ', final_error / dataloader.num_batches)
    print('Total final center pixel error of the model is ', final_error_errCenter / dataloader.num_batches)
    """
    Kalman Filter result: P=1
    divide /(tsetp+1)
    Total mean error of the model is  0.0057629116829
    Total final error of the model is  26.5588734678
    Total center pixel of the model is  0.0312387323408
    Total final center pixel error of the model is  37.2211500732


    
    Total mean error of the model is  0.0148667011345
    Total final error of the model is  17.3405704629
    Total center pixel of the model is  0.0217278201322
    Total final center pixel error of the model is  25.6021055394
    (result again)
    Total mean error of the model is  0.025117956151
    Total final error of the model is  29.3376933229
    Total center pixel of the model is  0.0323912769887
    Total final center pixel error of the model is  37.9175632732
    
    # use last velocity:
    Total mean error of the model is  0.0191087920229
    Total final error of the model is  22.8287988605
    Total center pixel of the model is  0.0266258032716
    Total final center pixel error of the model is  32.061451346
    """


if __name__ == '__main__':
    main()
