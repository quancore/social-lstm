import os
import argparse
import time
import itertools


import torch
from torch.autograd import Variable

import numpy as np
from utils import DataLoader
from helper import get_mean_error, get_final_error

from helper import *
from grid import getSequenceGridMask

class parameters():
    def __init__(self, args):
        args = args.parse_args()
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds = args.maxNumPeds
        self.seq_length = args.seq_length
        self.num_samples = args.num_samples
        self.num_epochs = args.num_epochs
        self.use_cuda = args.use_cuda
        self.drive = args.drive
        self.num_validation = args.num_validation
        self.gru = args.gru
        self.method = args.method
        self.best_n = args.best_n
        self.batch_size = args.batch_size





def sample_hyperparameters():
    """
    Yield possible hyperparameter choices.
    """

    while True:
        yield {
            "rnn_size": np.random.choice([64, 128, 256]).item(),
            "learning_schedule": np.random.choice(["RMSprop", "adagrad", "adam"]).item(),
            "grad_clip": np.random.uniform(7, 12),
            "learning_rate": np.random.uniform(0.001, 0.01),
            "decay_rate": np.random.uniform(0.7,1),
            "lambda_param" : np.random.uniform(0.0001,0.001),
            "dropout": np.random.uniform(0.3,1),
            "embedding_size": np.random.choice([64, 128, 256]).item(),
            "neighborhood_size": np.random.choice([8, 16, 32, 64]).item(),
            "grid_size": np.random.choice([2, 4, 8, 16]).item(),

        }



def write_to_file(file, args):
    file.write("rnn_size: "+str(args.rnn_size)+" learning_schedule: "+str(args.learning_schedule)+" grad_clip: "+str(args.grad_clip)+" learning_rate: "+str(args.learning_rate)+
                #"decay_rate: "+str(args.decay_rate)+
                " dropout: "+str(args.dropout)+" embedding_size: "+str(args.embedding_size)+" neighborhood_size: "+str(args.neighborhood_size)+" grid_size: "+str(args.grid_size)+'\n')

def print_to_screen(args):
    print("rnn_size: "+str(args.rnn_size)," learning_schedule: ",str(args.learning_schedule)," grad_clip: ",str(args.grad_clip)," learning_rate: ",str(args.learning_rate),
                #"decay_rate: ",str(args.decay_rate),
                " dropout: ",str(args.dropout)," embedding_size: ",str(args.embedding_size)," neighborhood_size: ",str(args.neighborhood_size)," grid_size: ",str(args.grid_size))



def main():
    
    parser = argparse.ArgumentParser()
    # Model to be loaded
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')

    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=10,
                        help='minibatch size')

    parser.add_argument('--num_samples', type=int, default=500,
                        help='NUmber of random configuration will be tested')

    parser.add_argument('--num_epochs', type=int, default=3,
                        help='number of epochs')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=27,
                        help='Maximum Number of Pedestrians')
    # cuda support
    parser.add_argument('--use_cuda', action="store_true", default=False,
                        help='Use GPU or not')
    # drive support
    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    # number of validation dataset will be used
    parser.add_argument('--num_validation', type=int, default=1,
                        help='Total number of validation dataset will be visualized')
    # gru model
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    # method selection for hyperparameter
    parser.add_argument('--method', type=int, default=1,
                        help='Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)')
    # number of parameter set will be logged
    parser.add_argument('--best_n', type=int, default=100,
                        help='Number of best n configuration will be logged')

    # Parse the parameters
    #sample_args = parser.parse_args()
    args = parameters(parser)
    
    args.best_n = np.clip(args.best_n, 0, args.num_samples)

    #for drive run
    prefix = ''
    f_prefix = '.'
    if args.drive is True:
      prefix='drive/semester_project/social_lstm_final/'
      f_prefix = 'drive/semester_project/social_lstm_final'
    

    method_name = get_method_name(args.method)
    model_name = "LSTM"
    save_tar_name = method_name+"_lstm_model_"
    if args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

    #plot directory for plotting in the future
    param_log = os.path.join(f_prefix)
    param_log_file = "hyperparameter"


    origin = (0,0)
    reference_point = (0,1)
    score = []
    param_set = []
    
   
    # Create the DataLoader object
    create_directories(param_log, [param_log_file])
    log_file = open(os.path.join(param_log, param_log_file, 'log.txt'), 'w+')
    
    dataloader_t = DataLoader(f_prefix, args.batch_size, args.seq_length, num_of_validation = args.num_validation, forcePreProcess = True, infer = True)
    dataloader_v = DataLoader(f_prefix, 1, args.seq_length, num_of_validation = args.num_validation, forcePreProcess = True, infer = True)


    for hyperparams in itertools.islice(sample_hyperparameters(), args.num_samples):
        args = parameters(parser)
        # randomly sample a parameter set
        args.rnn_size = hyperparams.pop("rnn_size")
        args.learning_schedule = hyperparams.pop("learning_schedule")
        args.grad_clip = hyperparams.pop("grad_clip")
        args.learning_rate = hyperparams.pop("learning_rate")
        args.lambda_param = hyperparams.pop("lambda_param")
        args.dropout = hyperparams.pop("dropout")
        args.embedding_size = hyperparams.pop("embedding_size")
        args.neighborhood_size = hyperparams.pop("neighborhood_size")
        args.grid_size = hyperparams.pop("grid_size")

        log_file.write("##########Parameters########"+'\n')
        print("##########Parameters########")
        write_to_file(log_file, args)
        print_to_screen(args)

        

        net = get_model(args.method, args)
        
        if args.use_cuda:        
            net = net.cuda()


        if(args.learning_schedule == "RMSprop"):
            optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
        elif(args.learning_schedule == "adagrad"):
            optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
        else:
            optimizer = torch.optim.Adam(net.parameters(), weight_decay=args.lambda_param)


        learning_rate = args.learning_rate

        total_process_start = time.time()


        # Training
        for epoch in range(args.num_epochs):
            print('****************Training epoch beginning******************')
            dataloader_t.reset_batch_pointer()
            loss_epoch = 0

            # For each batch
            for batch in range(dataloader_t.num_batches):
                start = time.time()

                # Get batch data
                x, y, d , numPedsList, PedsList ,target_ids = dataloader_t.next_batch()

                loss_batch = 0

                # For each sequence
                for sequence in range(dataloader_t.batch_size):
                    # Get the data corresponding to the current sequence
                    x_seq ,_ , d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[sequence], PedsList[sequence]
                    target_id = target_ids[sequence]

                    #get processing file name and then get dimensions of file
                    folder_name = dataloader_t.get_directory_name_with_pointer(d_seq)
                    dataset_data = dataloader_t.get_dataset_dimension(folder_name)

                    #dense vector creation
                    x_seq, lookup_seq = dataloader_t.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
                    target_id_values = x_seq[0][lookup_seq[target_id], 0:2]
                    #grid mask calculation
                    if args.method == 2: #obstacle lstm
                        grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda, True)
                    elif  args.method == 1: #social lstm   
                        grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda)
                    # vectorize trajectories in sequence
                    x_seq, _ = vectorize_seq(x_seq, PedsList_seq, lookup_seq)


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
                    if args.method == 3: #vanilla lstm
                        outputs, _, _ = net(x_seq, hidden_states, cell_states, PedsList_seq,numPedsList_seq ,dataloader_t, lookup_seq)

                    else:
                        outputs, _, _ = net(x_seq, grid_seq, hidden_states, cell_states, PedsList_seq,numPedsList_seq ,dataloader_t, lookup_seq)



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
                loss_batch = loss_batch / dataloader_t.batch_size
                loss_epoch += loss_batch

                print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * dataloader_t.num_batches + batch,
                                                                                        args.num_epochs * dataloader_t.num_batches,
                                                                                        epoch,
                                                                                        loss_batch, end - start))

            loss_epoch /= dataloader_t.num_batches
            # Log loss values
            log_file.write("Training epoch: "+str(epoch)+" loss: "+str(loss_epoch)+'\n')




        net = get_model(args.method, args, True)
        
        if args.use_cuda:        
            net = net.cuda()


        if(args.learning_schedule == "RMSprop"):
            optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
        elif(args.learning_schedule == "adagrad"):
            optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
        else:
            optimizer = torch.optim.Adam(net.parameters(), weight_decay=args.lambda_param)


        print('****************Validation dataset batch processing******************')
        dataloader_v.reset_batch_pointer()
        dataset_pointer_ins = dataloader_v.dataset_pointer

        loss_epoch = 0
        err_epoch = 0
        f_err_epoch = 0
        num_of_batch = 0
        smallest_err = 100000

        
        # For each batch
        for batch in range(dataloader_v.num_batches):
            start = time.time()
            # Get batch data
            x, y, d , numPedsList, PedsList ,target_ids = dataloader_v.next_batch()

            if dataset_pointer_ins is not dataloader_v.dataset_pointer:
                if dataloader_v.dataset_pointer is not 0:
                    print('Finished prosessed file : ', dataloader_v.get_file_name(-1),' Avarage error : ', err_epoch/num_of_batch)
                    num_of_batch = 0
                dataset_pointer_ins = dataloader_v.dataset_pointer



            # Loss for this batch
            loss_batch = 0
            err_batch = 0
            f_err_batch = 0

            # For each sequence
            for sequence in range(dataloader_v.batch_size):
                # Get data corresponding to the current sequence
                x_seq ,_ , d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[sequence], PedsList[sequence]
                target_id = target_ids[sequence]

                folder_name = dataloader_v.get_directory_name_with_pointer(d_seq)
                dataset_data = dataloader_v.get_dataset_dimension(folder_name)


                #dense vector creation
                x_seq, lookup_seq = dataloader_v.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
                
                #will be used for error calculation
                orig_x_seq = x_seq.clone() 
                target_id_values = x_seq[0][lookup_seq[target_id], 0:2]
                
                #grid mask calculation
                if args.method == 2: #obstacle lstm
                    grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda, True)
                elif  args.method == 1: #social lstm   
                    grid_seq = getSequenceGridMask(x_seq, dataset_data, PedsList_seq, args.neighborhood_size, args.grid_size, args.use_cuda)
                # vectorize trajectories in sequence
                x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)


                # <--------------Experimental block --------------->
                # Construct variables
                # x_seq, lookup_seq = dataloader_v.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
                # x_seq, target_id_values, first_values_dict = vectorize_seq_with_ped(x_seq, PedsList_seq, lookup_seq ,target_id)
                # angle = angle_between(reference_point, (x_seq[1][lookup_seq[target_id], 0].data.numpy(), x_seq[1][lookup_seq[target_id], 1].data.numpy()))
                # x_seq = rotate_traj_with_target_ped(x_seq, angle, PedsList_seq, lookup_seq)

                if args.use_cuda:                    
                    x_seq = x_seq.cuda()

                if args.method == 3: #vanilla lstm
                    ret_x_seq, loss = sample_validation_data_vanilla(x_seq, PedsList_seq, args, net, lookup_seq, numPedsList_seq, dataloader_v)

                else:
                    ret_x_seq, loss = sample_validation_data(x_seq, PedsList_seq, grid_seq, args, net, lookup_seq, numPedsList_seq, dataloader_v)
                
                #revert the points back to original space
                ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, lookup_seq, first_values_dict)

                err = get_mean_error(ret_x_seq.data, orig_x_seq.data, PedsList_seq, PedsList_seq, args.use_cuda, lookup_seq)
                f_err = get_final_error(ret_x_seq.data, orig_x_seq.data, PedsList_seq, PedsList_seq, lookup_seq)
                
                # ret_x_seq = rotate_traj_with_target_ped(ret_x_seq, -angle, PedsList_seq, lookup_seq)
                # ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, lookup_seq, target_id_values, first_values_dict)

                loss_batch += loss.item()
                err_batch += err
                f_err_batch += f_err
            
            end = time.time()
            print('Current file : ', dataloader_v.get_file_name(0),' Batch : ', batch+1, ' Sequence: ', sequence+1, ' Sequence mean error: ', err,' Sequence final error: ',f_err,' time: ', end - start)
            loss_batch = loss_batch / dataloader_v.batch_size
            err_batch = err_batch / dataloader_v.batch_size
            f_err_batch = f_err_batch / dataloader_v.batch_size
            num_of_batch += 1
            loss_epoch += loss_batch
            err_epoch += err_batch
            f_err_epoch += f_err_batch

        total_process_end = time.time()
        if dataloader_v.num_batches != 0:            
            loss_epoch = loss_epoch / dataloader_v.num_batches
            err_epoch = err_epoch / dataloader_v.num_batches
            f_err_epoch = f_err_epoch / dataloader_v.num_batches
            # calculate avarage error and time
            avg_err = (err_epoch+f_err_epoch)/2
            elapsed_time = (total_process_end - total_process_start)
            args.time = elapsed_time
            args.avg_err = avg_err

            score.append(avg_err)
            param_set.append(args)

            print('valid_loss = {:.3f}, valid_mean_err = {:.3f}, valid_final_err = {:.3f}, score = {:.3f}, time = {:.3f}'.format(loss_epoch, err_epoch, f_err_epoch, avg_err, elapsed_time))
            log_file.write('valid_loss = {:.3f}, valid_mean_err = {:.3f}, valid_final_err = {:.3f}, score = {:.3f}, time = {:.3f}'.format(loss_epoch, err_epoch, f_err_epoch, avg_err, elapsed_time)+'\n')


    print("--------------------------Best ", args.best_n," configuration------------------------")
    log_file.write("-----------------------------Best "+str(args.best_n) +" configuration---------------------"+'\n')
    biggest_indexes = np.array(score).argsort()[-args.best_n:]
    print("biggest_index: ", biggest_indexes)
    for arr_index, index in enumerate(biggest_indexes):
        print("&&&&&&&&&&&&&&&&&&&& ", arr_index," &&&&&&&&&&&&&&&&&&&&&&")
        log_file.write("&&&&&&&&&&&&&&&&&&&& "+ str(arr_index)+" &&&&&&&&&&&&&&&&&&&&&&"+'\n')
        curr_arg = param_set[index]
        write_to_file(log_file, curr_arg)
        print_to_screen(curr_arg)
        print("score: ",score)
        print('error = {:.3f}, time = {:.3f}'.format(curr_arg.avg_err, curr_arg.time))
        log_file.write('error = {:.3f}, time = {:.3f}'.format(curr_arg.avg_err, curr_arg.time)+'\n')
           
        



if __name__ == '__main__':
    main()
