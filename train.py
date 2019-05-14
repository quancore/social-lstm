import torch
import numpy as np
from torch.autograd import Variable

import argparse
import os
import time
import pickle
import subprocess

from model import SocialModel
from utils import DataLoader
from grid import getSequenceGridMask
from helper import *


def main():
    
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=5,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='prediction length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=30,
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
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=27,
                        help='Maximum Number of Pedestrians')

    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')
    # Cuda parameter
    parser.add_argument('--use_cuda', action="store_true", default=False,
                        help='Use GPU or not')
    # GRU parameter
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    # drive option
    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    # number of validation will be used
    parser.add_argument('--num_validation', type=int, default=2,
                        help='Total number of validation dataset for validate accuracy')
    # frequency of validation
    parser.add_argument('--freq_validation', type=int, default=1,
                        help='Frequency number(epoch) of validation using validation data')
    # frequency of optimazer learning decay
    parser.add_argument('--freq_optimizer', type=int, default=8,
                        help='Frequency number(epoch) of learning decay for optimizer')
    # store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
    parser.add_argument('--grid', action="store_true", default=True,
                        help='Whether store grids and use further epoch')
    
    args = parser.parse_args()
    
    train(args)


def train(args):
    origin = (0,0)
    reference_point = (0,1)
    validation_dataset_executed = False
  
    prefix = ''
    f_prefix = '.'
    if args.drive is True:
      prefix='drive/semester_project/social_lstm_final/'
      f_prefix = 'drive/semester_project/social_lstm_final'

    
    if not os.path.isdir("log/"):
      print("Directory creation script is running...")
      subprocess.call([f_prefix+'/make_directories.sh'])

    args.freq_validation = np.clip(args.freq_validation, 0, args.num_epochs)
    validation_epoch_list = list(range(args.freq_validation, args.num_epochs+1, args.freq_validation))
    validation_epoch_list[-1]-=1



    # Create the data loader object. This object would preprocess the data in terms of
    # batches each of size args.batch_size, of length args.seq_length
    dataloader = DataLoader(f_prefix, args.batch_size, args.seq_length, args.num_validation, forcePreProcess=True)

    model_name = "LSTM"
    method_name = "SOCIALLSTM"
    save_tar_name = method_name+"_lstm_model_"
    if args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"


    # Log directory
    log_directory = os.path.join(prefix, 'log/')
    plot_directory = os.path.join(prefix, 'plot/', method_name, model_name)
    plot_train_file_directory = 'validation'



    # Logging files
    log_file_curve = open(os.path.join(log_directory, method_name, model_name,'log_curve.txt'), 'w+')
    log_file = open(os.path.join(log_directory, method_name, model_name, 'val.txt'), 'w+')

    # model directory
    save_directory = os.path.join(prefix, 'model/')
    
    # Save the arguments int the config file
    with open(os.path.join(save_directory, method_name, model_name,'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, method_name, model_name, save_tar_name+str(x)+'.tar')

    # model creation
    net = SocialModel(args)
    if args.use_cuda:
        net = net.cuda()

    #optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
    #optimizer = torch.optim.Adam(net.parameters(), weight_decay=args.lambda_param)

    learning_rate = args.learning_rate

    best_val_loss = 100
    best_val_data_loss = 100

    smallest_err_val = 100000
    smallest_err_val_data = 100000


    best_epoch_val = 0
    best_epoch_val_data = 0

    best_err_epoch_val = 0
    best_err_epoch_val_data = 0

    all_epoch_results = []
    grids = []
    num_batch = 0
    dataset_pointer_ins_grid = -1

    [grids.append([]) for dataset in range(dataloader.get_len_of_dataset())]

    # Training
    for epoch in range(args.num_epochs):
        print('****************Training epoch beginning******************')
        if dataloader.additional_validation and (epoch-1) in validation_epoch_list:
            dataloader.switch_to_dataset_type(True)
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0

        # For each batch
        for batch in range(dataloader.num_batches):
            start = time.time()

            # Get batch data
            x, y, d , numPedsList, PedsList ,target_ids= dataloader.next_batch()
            loss_batch = 0
            
            #if we are in a new dataset, zero the counter of batch
            if dataset_pointer_ins_grid is not dataloader.dataset_pointer and epoch is not 0:
                num_batch = 0
                dataset_pointer_ins_grid = dataloader.dataset_pointer


            # For each sequence
            for sequence in range(dataloader.batch_size):
                # Get the data corresponding to the current sequence
                x_seq ,_ , d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[sequence], PedsList[sequence]
                target_id = target_ids[sequence]

                #get processing file name and then get dimensions of file
                folder_name = dataloader.get_directory_name_with_pointer(d_seq)
                dataset_data = dataloader.get_dataset_dimension(folder_name)

                #dense vector creation
                x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
                target_id_values = x_seq[0][lookup_seq[target_id], 0:2]

                #grid mask calculation and storage depending on grid parameter
                if(args.grid):
                    if(epoch is 0):
                        grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq,args.neighborhood_size, args.grid_size, args.use_cuda)
                        grids[dataloader.dataset_pointer].append(grid_seq)
                    else:
                        grid_seq = grids[dataloader.dataset_pointer][(num_batch*dataloader.batch_size)+sequence]
                else:
                    grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq,args.neighborhood_size, args.grid_size, args.use_cuda)

                # vectorize trajectories in sequence
                x_seq, _ = vectorize_seq(x_seq, PedsList_seq, lookup_seq)

                
                
                # <---------------------- Experimental block ----------------------->
                # Main approach:
                # 1) Translate all trajectories using first frame value of target trajectory so that target trajectory will start (0,0).
                # 2) Get angle between first trajectory point of target ped and (0, 1) for turning.
                # 3) Rotate all trajectories in the sequence using this angle.
                # 4) Calculate grid mask for hidden layer pooling.
                # 5) Vectorize all trajectories (substract first frame values of each trajectories from subsequent points in the trajectory).
                #
                # Problem:
                #  Low accuracy
                #
                # Possible causes:
                # *Each function has been already checked -> low possibility.
                # *Logic errors or algorithm errors -> high possibility.
                # *Wrong order of execution each step -> high possibility.
                # <------------------------------------------------------------------------>

                # x_seq = translate(x_seq, PedsList_seq, lookup_seq ,target_id_values)

                # angle = angle_between(reference_point, (x_seq[1][lookup_seq[target_id], 0].data.numpy(), x_seq[1][lookup_seq[target_id], 1].data.numpy()))

                # x_seq = rotate_traj_with_target_ped(x_seq, angle, PedsList_seq, lookup_seq)
                
                # if(args.grid):
                #     if(epoch is 0):
                #         grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq,args.neighborhood_size, args.grid_size, args.use_cuda)
                #         grids[dataloader.dataset_pointer].append(grid_seq)
                #     else:
                #         #grid_seq1 = getSequenceGridMask(x_seq, dataset_data, PedsList_seq,args.neighborhood_size, args.grid_size, args.use_cuda)
                #         grid_seq = grids[dataloader.dataset_pointer][(num_batch*dataloader.batch_size)+sequence]
                #         #print([ torch.equal(x.data, y.data) for (x,y) in zip(grid_seq1, grid_seq)])
                #         #if not (all([ torch.equal(x.data, y.data) for (x,y) in zip(grid_seq1, grid_seq)])):
                #         #    print("not equal")
                #         #    quit()
                # else:
                #     grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq,args.neighborhood_size, args.grid_size, args.use_cuda)
                
                # x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)

                
                #print(grid_seq)

                  # Construct variables
                #print("target id : ", target_id)
                #print("look up : ", lookup_seq)
                #print("pedlist_seq: ", PedsList_seq)
                #print("before_xseq: ", x_seq)
                #x_seq, target_id_values, first_values_dict = vectorize_seq_with_ped(x_seq, PedsList_seq, lookup_seq ,target_id)
                #print("after_vectorize_seq: ", x_seq)
                #print("angle: ", np.rad2deg(angle))
                #print("after_xseq: ", x_seq)
                #x_seq = rotate_traj_with_target_ped(x_seq, -angle, PedsList_seq, lookup_seq)
                #x_seq = revert_seq(x_seq, PedsList_seq, lookup_seq, first_values_dict)


                if args.use_cuda:                    
                    x_seq = x_seq.cuda()


                #number of peds in this sequence per frame
                numNodes = len(lookup_seq)


                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:                    
                    hidden_states = hidden_states.cuda()

                cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:                    
                    cell_states = cell_states.cuda()

                # Zero out gradients
                net.zero_grad()
                optimizer.zero_grad()
                

                # Forward prop
                outputs, _, _ = net(x_seq, grid_seq, hidden_states, cell_states, PedsList_seq,numPedsList_seq ,dataloader, lookup_seq)

                
                # Compute loss
                loss = Gaussian2DLikelihood(outputs, x_seq, PedsList_seq, lookup_seq)
                loss_batch += loss.item()

                # Compute gradients
                loss.backward()
                

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch
            num_batch+=1

            print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * dataloader.num_batches + batch,
                                                                                    args.num_epochs * dataloader.num_batches,
                                                                                    epoch,
                                                                                    loss_batch, end - start))

        loss_epoch /= dataloader.num_batches
        # Log loss values
        log_file_curve.write("Training epoch: "+str(epoch)+" loss: "+str(loss_epoch)+'\n')

        if dataloader.valid_num_batches > 0:
            print('****************Validation epoch beginning******************')

            # Validation
            dataloader.reset_batch_pointer(valid=True)
            loss_epoch = 0
            err_epoch = 0

            # For each batch
            for batch in range(dataloader.valid_num_batches):
                # Get batch data
                x, y, d , numPedsList, PedsList ,target_ids= dataloader.next_valid_batch()

                # Loss for this batch
                loss_batch = 0
                err_batch = 0


                # For each sequence
                for sequence in range(dataloader.batch_size):
                    # Get data corresponding to the current sequence
                    x_seq ,_ , d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[sequence], PedsList[sequence]
                    target_id = target_ids[sequence]

                    #get processing file name and then get dimensions of file
                    folder_name = dataloader.get_directory_name_with_pointer(d_seq)
                    dataset_data = dataloader.get_dataset_dimension(folder_name)
                    
                    #dense vector creation
                    x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)

                    target_id_values = x_seq[0][lookup_seq[target_id], 0:2]
                    
                    #get grid mask
                    grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda)

                    x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)


                    # <---------------------- Experimental block ----------------------->
                    # x_seq = translate(x_seq, PedsList_seq, lookup_seq ,target_id_values)
                    # angle = angle_between(reference_point, (x_seq[1][lookup_seq[target_id], 0].data.numpy(), x_seq[1][lookup_seq[target_id], 1].data.numpy()))
                    # x_seq = rotate_traj_with_target_ped(x_seq, angle, PedsList_seq, lookup_seq)
                    # grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda)
                    # x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)


                    if args.use_cuda:                    
                        x_seq = x_seq.cuda()

                    #number of peds in this sequence per frame
                    numNodes = len(lookup_seq)

                    hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                    if args.use_cuda:                    
                        hidden_states = hidden_states.cuda()
                    cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                    if args.use_cuda:                    
                        cell_states = cell_states.cuda()

                    # Forward prop
                    outputs, _, _ = net(x_seq[:-1], grid_seq[:-1], hidden_states, cell_states, PedsList_seq[:-1], numPedsList_seq , dataloader, lookup_seq)

                    # Compute loss
                    loss = Gaussian2DLikelihood(outputs, x_seq[1:], PedsList_seq[1:], lookup_seq)
                    # Extract the mean, std and corr of the bivariate Gaussian
                    mux, muy, sx, sy, corr = getCoef(outputs)
                    # Sample from the bivariate Gaussian
                    next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, PedsList_seq[-1], lookup_seq)
                    next_vals = torch.FloatTensor(1,numNodes,2)
                    next_vals[:,:,0] = next_x
                    next_vals[:,:,1] = next_y
                    err = get_mean_error(next_vals, x_seq[-1].data[None, : ,:], [PedsList_seq[-1]], [PedsList_seq[-1]], args.use_cuda, lookup_seq)

                    loss_batch += loss.item()
                    err_batch += err

                loss_batch = loss_batch / dataloader.batch_size
                err_batch = err_batch / dataloader.batch_size
                loss_epoch += loss_batch
                err_epoch += err_batch

            if dataloader.valid_num_batches != 0:            
                loss_epoch = loss_epoch / dataloader.valid_num_batches
                err_epoch = err_epoch / dataloader.num_batches


                # Update best validation loss until now
                if loss_epoch < best_val_loss:
                    best_val_loss = loss_epoch
                    best_epoch_val = epoch

                if err_epoch<smallest_err_val:
                    smallest_err_val = err_epoch
                    best_err_epoch_val = epoch

                print('(epoch {}), valid_loss = {:.3f}, valid_err = {:.3f}'.format(epoch, loss_epoch, err_epoch))
                print('Best epoch', best_epoch_val, 'Best validation loss', best_val_loss, 'Best error epoch',best_err_epoch_val, 'Best error', smallest_err_val)
                log_file_curve.write("Validation epoch: "+str(epoch)+" loss: "+str(loss_epoch)+" err: "+str(err_epoch)+'\n')


        # Validation dataset
        if dataloader.additional_validation and (epoch) in validation_epoch_list:
            dataloader.switch_to_dataset_type()
            print('****************Validation with dataset epoch beginning******************')
            dataloader.reset_batch_pointer(valid=False)
            dataset_pointer_ins = dataloader.dataset_pointer
            validation_dataset_executed = True

            loss_epoch = 0
            err_epoch = 0
            f_err_epoch = 0
            num_of_batch = 0
            smallest_err = 100000

            #results of one epoch for all validation datasets
            epoch_result = []
            #results of one validation dataset
            results = []



            # For each batch
            for batch in range(dataloader.num_batches):
                # Get batch data
                x, y, d , numPedsList, PedsList ,target_ids = dataloader.next_batch()

                if dataset_pointer_ins is not dataloader.dataset_pointer:
                    if dataloader.dataset_pointer is not 0:
                        print('Finished prosessed file : ', dataloader.get_file_name(-1),' Avarage error : ', err_epoch/num_of_batch)
                        num_of_batch = 0
                        epoch_result.append(results)

                    dataset_pointer_ins = dataloader.dataset_pointer
                    results = []



                # Loss for this batch
                loss_batch = 0
                err_batch = 0
                f_err_batch = 0

                # For each sequence
                for sequence in range(dataloader.batch_size):
                    # Get data corresponding to the current sequence
                    x_seq ,_ , d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[sequence], PedsList[sequence]
                    target_id = target_ids[sequence]

                    #get processing file name and then get dimensions of file
                    folder_name = dataloader.get_directory_name_with_pointer(d_seq)
                    dataset_data = dataloader.get_dataset_dimension(folder_name)
                    
                    #dense vector creation
                    x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
                    
                    #will be used for error calculation
                    orig_x_seq = x_seq.clone() 
                    
                    target_id_values = orig_x_seq[0][lookup_seq[target_id], 0:2]
                    
                    #grid mask calculation
                    grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda)
                    
                    if args.use_cuda:
                        x_seq = x_seq.cuda()
                        orig_x_seq = orig_x_seq.cuda()

                    #vectorize datapoints
                    x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)

                    # <---------------------- Experimental block ----------------------->
                    # x_seq = translate(x_seq, PedsList_seq, lookup_seq ,target_id_values)
                    # angle = angle_between(reference_point, (x_seq[1][lookup_seq[target_id], 0].data.numpy(), x_seq[1][lookup_seq[target_id], 1].data.numpy()))
                    # x_seq = rotate_traj_with_target_ped(x_seq, angle, PedsList_seq, lookup_seq)
                    # grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda)
                    # x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)

                    #sample predicted points from model
                    ret_x_seq, loss = sample_validation_data(x_seq, PedsList_seq, grid_seq, args, net, lookup_seq, numPedsList_seq, dataloader)

                    #revert the points back to original space
                    ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, lookup_seq, first_values_dict)

                    # <---------------------- Experimental block revert----------------------->
                    # Revert the calculated coordinates back to original space:
                    # 1) Convert point from vectors to absolute coordinates
                    # 2) Rotate all trajectories in reverse angle
                    # 3) Translate all trajectories back to original space by adding the first frame value of target ped trajectory
                    
                    # *It works without problems which mean that it reverts a trajectory back completely
                    
                    # Possible problems:
                    # *Algoritmical errors caused by first experimental block -> High possiblity
                    # <------------------------------------------------------------------------>

                    # ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, lookup_seq, first_values_dict)

                    # ret_x_seq = rotate_traj_with_target_ped(ret_x_seq, -angle, PedsList_seq, lookup_seq)

                    # ret_x_seq = translate(ret_x_seq, PedsList_seq, lookup_seq ,-target_id_values)

                    #get mean and final error
                    err = get_mean_error(ret_x_seq.data, orig_x_seq.data, PedsList_seq, PedsList_seq, args.use_cuda, lookup_seq)
                    f_err = get_final_error(ret_x_seq.data, orig_x_seq.data, PedsList_seq, PedsList_seq, lookup_seq)
                    
                    loss_batch += loss.item()
                    err_batch += err
                    f_err_batch += f_err
                    print('Current file : ', dataloader.get_file_name(0),' Batch : ', batch+1, ' Sequence: ', sequence+1, ' Sequence mean error: ', err,' Sequence final error: ',f_err,' time: ', end - start)
                    results.append((orig_x_seq.data.cpu().numpy(), ret_x_seq.data.cpu().numpy(), PedsList_seq, lookup_seq, dataloader.get_frame_sequence(args.seq_length), target_id))

                loss_batch = loss_batch / dataloader.batch_size
                err_batch = err_batch / dataloader.batch_size
                f_err_batch = f_err_batch / dataloader.batch_size
                num_of_batch += 1
                loss_epoch += loss_batch
                err_epoch += err_batch
                f_err_epoch += f_err_batch

            epoch_result.append(results)
            all_epoch_results.append(epoch_result)


            if dataloader.num_batches != 0:            
                loss_epoch = loss_epoch / dataloader.num_batches
                err_epoch = err_epoch / dataloader.num_batches
                f_err_epoch = f_err_epoch / dataloader.num_batches
                avarage_err = (err_epoch + f_err_epoch)/2

                # Update best validation loss until now
                if loss_epoch < best_val_data_loss:
                    best_val_data_loss = loss_epoch
                    best_epoch_val_data = epoch

                if avarage_err<smallest_err_val_data:
                    smallest_err_val_data = avarage_err
                    best_err_epoch_val_data = epoch

                print('(epoch {}), valid_loss = {:.3f}, valid_mean_err = {:.3f}, valid_final_err = {:.3f}'.format(epoch, loss_epoch, err_epoch, f_err_epoch))
                print('Best epoch', best_epoch_val_data, 'Best validation loss', best_val_data_loss, 'Best error epoch',best_err_epoch_val_data, 'Best error', smallest_err_val_data)
                log_file_curve.write("Validation dataset epoch: "+str(epoch)+" loss: "+str(loss_epoch)+" mean_err: "+str(err_epoch)+'final_err: '+str(f_err_epoch)+'\n')

            optimizer = time_lr_scheduler(optimizer, epoch, lr_decay_epoch = args.freq_optimizer)


        # Save the model after each epoch
        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))




    if dataloader.valid_num_batches != 0:        
        print('Best epoch', best_epoch_val, 'Best validation Loss', best_val_loss, 'Best error epoch',best_err_epoch_val, 'Best error', smallest_err_val)
        # Log the best epoch and best validation loss
        log_file.write('Validation Best epoch:'+str(best_epoch_val)+','+' Best validation Loss: '+str(best_val_loss))

    if dataloader.additional_validation:
        print('Best epoch acording to validation dataset', best_epoch_val_data, 'Best validation Loss', best_val_data_loss, 'Best error epoch',best_err_epoch_val_data, 'Best error', smallest_err_val_data)
        log_file.write("Validation dataset Best epoch: "+str(best_epoch_val_data)+','+' Best validation Loss: '+str(best_val_data_loss)+'\n')
        #dataloader.write_to_plot_file(all_epoch_results[best_epoch_val_data], plot_directory)

    #elif dataloader.valid_num_batches != 0:
    #    dataloader.write_to_plot_file(all_epoch_results[best_epoch_val], plot_directory)

    #else:
    if validation_dataset_executed:
        dataloader.switch_to_dataset_type(load_data=False)
        create_directories(plot_directory, [plot_train_file_directory])
        dataloader.write_to_plot_file(all_epoch_results[len(all_epoch_results)-1], os.path.join(plot_directory, plot_train_file_directory))

    # Close logging files
    log_file.close()
    log_file_curve.close()



if __name__ == '__main__':
    main()