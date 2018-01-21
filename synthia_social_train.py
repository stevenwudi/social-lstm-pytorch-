'''
Train script for the Social LSTM model

Author: Di Wu
Date: 21st jan 2018
'''
import torch
from torch.autograd import Variable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import os
import time
import pickle

from model import SocialLSTM
from synthia_utils import Synthia_DataLoader
from grid import getSequenceGridMask
from st_graph import ST_GRAPH
from criterion import Gaussian2DLikelihood


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=8,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=23,
                        help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=8,
                        help='prediction length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=list, default=[760/2, 1280/2],
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=0,
                        help='The dataset index to be left out in training')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0001,
                        help='L2 regularization parameter')

    # The data dir
    parser.add_argument('--data_root', type=str, default='../data/synthia/SYNTHIA-SEQS-01',
                        help='The dataset directory')
    parser.add_argument('--img_dir', type=str, default='/media/samsumg_1tb/synthia/Datasets',
                        help='The image directory')
    parser.add_argument('--dataset_dim', type=list, default=[760, 1280],
                        help='Dataset dimensions')
    parser.add_argument('--input_dim', type=int, default=7,
                        help='[car_id, centreX, centreY, height, width, d_min, d_max]')

    args = parser.parse_args()
    train(args)


def train(args):
    datasets = [i for i in range(9)]
    # Remove the leave out dataset from the datasets
    datasets.remove(args.leaveDataset)

    # Construct the DataLoader object

    dataloader = Synthia_DataLoader(data_root=args.data_root, img_dir=args.img_dir, leaveDataset=args.leaveDataset,
                                    batch_size=args.batch_size, seq_length=args.seq_length+1,
                                    datasets=datasets, dataset_dim=args.dataset_dim, forcePreProcess=False)

    # Construct the ST-graph object
    stgraph = ST_GRAPH(args.batch_size, args.seq_length + 1, args.dataset_dim)

    # Log directory
    log_directory = '../log/'
    log_directory += str(args.leaveDataset) + '/'

    # Logging files
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')

    # Save directory
    save_directory = '../save/'
    save_directory += str(args.leaveDataset) + '/'

    # Dump the arguments into the configuration file
    with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, 'social_lstm_model_'+str(x)+'.tar')

    # Initialize net
    net = SocialLSTM(args)
    net.cuda()

    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)

    print('Training begin')

    # Training
    for epoch in range(args.num_epochs):
        dataloader.reset_batch_pointer()
        loss_epoch = 0

        # For each batch
        for batch in range(dataloader.num_batches):
            start = time.time()

            # Get batch data
            x, _, _ = dataloader.next_batch()

            # Construct the stgraph
            stgraph.readGraph(x)

            loss_batch = 0

            # For each sequence
            for sequence in range(dataloader.batch_size):
                # Get the data corresponding to the current sequence
                x_seq = x[sequence]

                # Get the node features and nodes present from stgraph
                nodes, _, nodesPresent, _, retNodePresentName = stgraph.getSequence(sequence)

                # Compute grid masks
                grid_seq = getSequenceGridMask(x_seq, args.neighborhood_size, args.grid_size, retNodePresentName)

                # Construct variables
                nodes = Variable(torch.from_numpy(nodes).float()).cuda()
                numNodes = nodes.size()[1]
                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size)).cuda()
                cell_states = Variable(torch.zeros(numNodes, args.rnn_size)).cuda()

                # Zero out gradients
                net.zero_grad()
                optimizer.zero_grad()

                # Forward prop
                outputs, _, _ = net(nodes[:-1], grid_seq[:-1], nodesPresent[:-1], hidden_states, cell_states)

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)
                loss_batch += loss.data[0]

                # Compute gradients
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

            # Reset stgraph
            stgraph.reset()
            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

            print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.
                  format(epoch * dataloader.num_batches + batch, args.num_epochs * dataloader.num_batches, epoch, loss_batch, end - start))
        loss_epoch /= dataloader.num_batches
        # Log loss values
        log_file_curve.write(str(epoch)+','+str(loss_epoch)+',')

        print('(epoch {}), valid_loss = {:.3f}'.format(epoch, loss_epoch))
        log_file_curve.write(str(loss_epoch)+'\n')

        # Save the model after each epoch
        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))

    # Close logging files
    log_file.close()
    log_file_curve.close()


if __name__ == '__main__':
    main()
