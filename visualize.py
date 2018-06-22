import numpy as np
import torch
from torch.autograd import Variable

import os
import random
import matplotlib
import matplotlib.animation as animation
import itertools
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from random import randint
from random import choice
from textwrap import wrap
from cycler import cycler
from random import shuffle
import matplotlib as mpl
from adjustText import adjust_text
import math


import pickle
#from graphviz import Digraph
from torch.autograd import Variable
import argparse
from helper import get_all_file_names, vectorize_seq, angle_between, rotate \
, remove_file_extention, delete_file, clear_folder, translate, rotate_traj_with_target_ped\
,get_mean_error, get_final_error, get_method_name

def save_video(sequence_path, video_path, video_name ,frame):
    # save plots as video
    image_input_path = os.path.join(sequence_path, "sequence%05d.png")
    video_output_path = os.path.join(video_path, video_name+'.mp4')
    command = "ffmpeg -r "+str(frame)+" -i "+image_input_path+" -vcodec mpeg4 -y "+video_output_path
    print("Creating video: input sequence --> ",image_input_path," | output video --> ",video_output_path)
    os.system(str(command))

def min_max_scaler(arr, min, max, min_range, max_range):
    # scale trajectory data points given min max
    arr_std = (arr - min) / (max - min)
    arr_scaled = (arr_std*(max_range - min_range)) + min_range
    return arr_scaled

def annotate_plot(plt, x_vals, y_vals, annotation):
    # annotate plot using external library
    texts =[]
    for index,(i,j) in enumerate(zip(x_vals,y_vals)):
        texts.append(plt.text(i, j , annotation[index], fontsize=10, fontweight ='bold'))
    adjust_text(texts, force_points=1.5, expand_points=(3, 3), arrowprops=dict(arrowstyle="->", color='b', alpha=0.5, lw=0.5))


def get_min_max_in_trajectories(traj):
    # get minimum and maximum values from whole trajectory set
    #value index x = 0 , y= 1

    dict_max_min = {}

    concat_list_vals = [np.concatenate(seq[:,:,0]) for seq in traj]
    max_val_x = np.max(([np.nanmax(val_list) for val_list in concat_list_vals]))
    min_val_x = np.min(([np.nanmin(val_list) for val_list in concat_list_vals]))
    concat_list_vals = [np.concatenate(seq[:,:,1]) for seq in traj]
    max_val_y = np.max(([np.nanmax(val_list) for val_list in concat_list_vals]))
    min_val_y = np.min(([np.nanmin(val_list) for val_list in concat_list_vals]))

    dict_max_min['x'] = [max_val_x, min_val_x]
    dict_max_min['y'] = [max_val_y, min_val_y]
    return dict_max_min    

def get_min_max_in_sequence(traj):
    # get minimum and maximum x, y points given sequence
    #value index x = 0 , y= 1
    dict_max_min = {}
    x_vals = np.concatenate(traj[:,:,0])
    max_val_x = np.nanmax(x_vals)
    min_val_x = np.nanmin(x_vals)
    y_vals = np.concatenate(traj[:,:,1])
    max_val_y = np.nanmax(y_vals)
    min_val_y = np.nanmin(y_vals)

    dict_max_min['x'] = [max_val_x, min_val_x]
    dict_max_min['y'] = [max_val_y, min_val_y]
    return dict_max_min    

def get_marker_style(markers, marker_cycle, index, num_of_color):
    # get a marker for line graph
    if num_of_color == index:
        shuffle(markers)
        marker = itertools.cycle(markers)
        return marker
    return marker_cycle

def get_line_style(linestyle, line_cycle, index, num_of_color):
    # get a line style for line graph
    if num_of_color == index:
        shuffle(linestyle)
        lines = itertools.cycle((linestyle))
        return lines
    return line_cycle

def plot_trajectories(true_trajs, pred_trajs, nodesPresent, look_up, frames, name, plot_directory, min_threshold, max_ped_ratio, target_id , style, obs_length = 20):
    '''
    Parameters
    ==========

    true_trajs : Numpy matrix of shape seq_length x numNodes x 2
    Contains the true trajectories of the nodes

    pred_trajs : Numpy matrix of shape seq_length x numNodes x 2
    Contains the predicted trajectories of the nodes

    nodesPresent : A list of lists, of size seq_length
    Each list contains the nodeIDs present at that time-step in pred traj

    obs_length : Length of observed trajectory

    lookup : Lookup table converting ped ids to array indices

    plot_directory : directory that plots will be stored

    min_threshold : minimum trajectory lenght to be filtered

    max_ped_ratio : percentage of peds in a frame to be visualized

    target_id : target ped id for this sequence
    
    style : an array includes max_r, min_r and plot ofset (for centering trajectories)


    name : Name of the plot

    '''

    fig_width = 10
    fig_height = 10

    min_r = style[0]
    max_r = style[1]
    plot_offset = style[2]

    reference_point = (0, max_r)
    target_frame = 1 # frame will be taken for rotation vector (starting point (0,0), ending point: this frame)
    video_plot_trajs = []
    '''
    overall_trajs = []

    overall_max_arr_x = []
    overall_max_arr_y = []
    overall_min_arr_x = []
    overall_min_arr_y = []
    '''

    line_width = 3
    num_of_color = 10
    mpl.rcParams['lines.markersize'] = 10
    markers = ['+', '.', 'o', '*', 'v','^', '<', '>', '1','2','3','4','8','s','p','P','h','x','X','D']
    line_style = ["--","-.",":"]
    
    print("****************************")
    print("orig true_")
    print(true_trajs[:, look_up[target_id], :])
    print("orig pred_")
    print(pred_trajs[:, look_up[target_id], :])
    
    # first frame was not predicted, so only take predicted parts
    pred_trajs = pred_trajs[1:]
    true_trajs = true_trajs[1:]
    nodesPresent = nodesPresent[1:]
    
    # print('lookup: ', look_up)
    # print('target_id: ', target_id)
    # translate all trajectories using first frame of target ped to center target ped traj
    true_trajs = translate(torch.from_numpy(true_trajs), nodesPresent, look_up ,torch.from_numpy(true_trajs[0][look_up[target_id], 0:2]))
    pred_trajs = translate(torch.from_numpy(pred_trajs), nodesPresent, look_up ,torch.from_numpy(pred_trajs[0][look_up[target_id], 0:2]))


    #true_trajs, _= vectorize_seq_with_ped(torch.from_numpy(true_trajs), nodesPresent, look_up ,target_id)
    #pred_trajs, _= vectorize_seq_with_ped(torch.from_numpy(pred_trajs), nodesPresent, look_up ,target_id)
    print("orig true_")
    print(true_trajs[:, look_up[target_id], :])
    print("orig pred_")
    print(pred_trajs[:, look_up[target_id], :])
    
    #angle_true = angle_between(reference_point, (true_trajs[target_frame][look_up[target_id], 0], true_trajs[target_frame][look_up[target_id], 1]))
    #angle_pred = angle_between(reference_point, (pred_trajs[target_frame][look_up[target_id], 0], pred_trajs[target_frame][look_up[target_id], 1]))

    #true_trajs = rotate_traj_with_target_ped(true_trajs, angle_true, nodesPresent, look_up)
    #pred_trajs = rotate_traj_with_target_ped(pred_trajs, angle_true, nodesPresent, look_up)

    #print("+++++++++++++++++++++++++++++++++++++++++++++++44")
    #print("angle: ", np.rad2deg(angle_true))
    print("true_")
    print(true_trajs[:, look_up[target_id], :])
    print("pred_")
    print(pred_trajs[:, look_up[target_id], :])
    
    dict_true = get_min_max_in_sequence(true_trajs.numpy())
    dict_pred = get_min_max_in_sequence(pred_trajs[:, look_up[target_id], :][:,None,:].numpy())#only consider prediciton of target id because of model
    #print(dict_pred)
    #print(dict_true)
    overaall_max_x, overaall_min_x, overaall_max_y, overaall_min_y = get_overall_max_min(dict_true, dict_pred)

    
    #print('max_x: ', overaall_max_x,'max_y: ', overaall_max_y,'min_x: ', overaall_min_x,'min_y: ', overaall_min_y)

    
    '''
    print("Orig. true traj")
    print(true_trajs)
    print("*********************")
    print("Orig. pred traj")
    print(pred_trajs)
    '''

    # figure parameters
    fig = plt.figure(figsize=(fig_width, fig_width))
    props = dict(fontsize=12)

    # title string which includes id of frames in this sequence
    frames_str = ' | '.join(str(int(e)) for e in frames)
    frames_str = "frame ids: " + frames_str
    plt.gca().set_title("\n".join(wrap(frames_str,80)), props, loc ='center')

    # adjust color palette, markers and line style of plot
    cm = plt.get_cmap('tab20')
    num_of_color = 20
    colors = [cm(i) for i in np.linspace(0, 1, num_of_color)]
    shuffle(colors)
    plt.gca().set_prop_cycle(cycler('color', colors))
    shuffle(markers)
    shuffle(line_style)
    marker = itertools.cycle(markers)
    lines = itertools.cycle((line_style))

    traj_length, numNodes, _  = true_trajs.shape
    traj_data = {}
    #look_up = dict(look_up)
    inv_lookup = {v: k for k, v in look_up.items()} # inverse lookup table -> array indices : ped_ids
    
    # create a dict that includes datapoints for each peds in the frame
    for tstep in range(traj_length-1):#real traj lenght is traj_lenght-1
            pred_pos = pred_trajs[tstep, :]
            true_pos = true_trajs[tstep, :]

            for ped in range(numNodes):
                ped_index = ped
                ped_id = inv_lookup[ped]
                if ped not in traj_data:
                    traj_data[ped_index] = [[], []]

                
                if ped_id in nodesPresent[tstep]:
                    traj_data[ped_index][0].append(true_pos[ped_index, :])
                    traj_data[ped_index][1].append(pred_pos[ped_index, :])
    

    #print(np.array(traj_data.values()))


    processed_ped_number = 0 # number of peds already processed
    num_of_peds = math.ceil(max_ped_ratio * len(traj_data)) # maximum number of peds will be processed
    print("Max number of peds in this  seq.: ",num_of_peds)
    
    # choose num_of_peds randomly
    shuffled_ped_ids = list(range(0, len(traj_data)))
    random.shuffle(shuffled_ped_ids)

    #add target id to at the beginnnig of the list therefore it will process always
    shuffled_ped_ids.remove(look_up[target_id])
    shuffled_ped_ids.insert(0, look_up[target_id])


    processed_ped_index = []
    real_inv_lookup = {} # create a subset of inverse lookup for index out of range error

    true_target_id_values = None
    pred_target_id_values = None
    target_sequence_true = None
    target_sequence_pred = None

    # process dict for each ped
    for j in shuffled_ped_ids:

        #format_params = []
        #print("Processing ped ", j)
        if processed_ped_number >= num_of_peds: # finished processed peds
            break
        true_traj_ped = traj_data[j][0]  # List of [x,y] elements
        pred_traj_ped = traj_data[j][1]
        ped_id = inv_lookup[j]
        #print("ped id : ", ped_id, "target id :", target_id)

        # get corresponding ped and coordinates
        true_x = [p[0] for p in true_traj_ped]
        true_y = [p[1] for p in true_traj_ped]

        pred_x = [p[0] for p in pred_traj_ped]
        pred_y = [p[1] for p in pred_traj_ped]

        real_inv_lookup[processed_ped_number] = ped_id # assign processed ped number and ped id

        if not len(true_x) > min_threshold or not len(true_x) > 2: # skip the trajectory if len is smaller than 2 or min_threshold
            print("Ped processing is aborted because its trajectory lenght in this sequence is smaller than threshold or 1 points")
            continue
        else:
            processed_ped_index.append(processed_ped_number)

            processed_ped_number = processed_ped_number +1

        # get a marker and line style
        marker = get_marker_style(markers, marker, j, num_of_color)
        lines = get_marker_style(line_style, lines, j, num_of_color)

        # print("................................")
        # print("ped_id: ", ped_id)
        # print("true_x")
        # print(true_x)
        # print("true_y")
        # print(true_y)
        # print("pred_x")
        # print(pred_x)
        # print("pred_y")
        # print(pred_y)
        
        # exctract non-nan values and scale it according to min and max overall values in this frame 
        filtered_true_x = min_max_scaler(np.array([x for x in true_x if not np.isnan(x)]), overaall_min_x, overaall_max_x, min_r, max_r)
        filtered_true_y = min_max_scaler(np.array([y for y in true_y if not np.isnan(y)]), overaall_min_y, overaall_max_y, min_r, max_r)
        filtered_pred_x = min_max_scaler(np.array([x for x in pred_x if not np.isnan(x)]), overaall_min_x, overaall_max_x, min_r, max_r)
        filtered_pred_y = min_max_scaler(np.array([y for y in pred_y if not np.isnan(y)]), overaall_min_y, overaall_max_y, min_r, max_r)


        # <-----------------------------Experimental block ------------------------------>
        #filtered_true_x = vectorize_traj_point_arr(min_max_scaler(np.array([x for x in true_x if not np.isnan(x)]), overaall_min_x, overaall_max_x, min_r, max_r))
        #filtered_true_y = vectorize_traj_point_arr(min_max_scaler(np.array([y for y in true_y if not np.isnan(y)]), overaall_min_y, overaall_max_y, min_r, max_r))
        #filtered_pred_x = vectorize_traj_point_arr(min_max_scaler(np.array([x for x in pred_x if not np.isnan(x)]), overaall_min_x, overaall_max_x, min_r, max_r))
        #filtered_pred_y = vectorize_traj_point_arr(min_max_scaler(np.array([y for y in pred_y if not np.isnan(y)]), overaall_min_y, overaall_max_y, min_r, max_r))
        '''
        print("**********************************************************************")
        print("filtered_true_x")
        print(filtered_true_x)
        print("filtered_true_y y")
        print(filtered_true_y)
        print("filtered_pred_x")
        print(filtered_pred_x)
        print("filtered_pred_y")
        print(filtered_pred_y)
        '''
        
        '''
        initial_angle_true  = angle_between(reference_point, (filtered_true_x[1], filtered_true_y[1]))
        initial_angle_pred  = angle_between(reference_point, (filtered_pred_x[1], filtered_pred_y[1]))

    
        

        #print("Angles true: ", np.rad2deg(initial_angle_true), "pred: ", np.rad2deg(initial_angle_pred))
        #print("******************************************************************************************")

        rotate_traj(filtered_true_x, filtered_true_y, initial_angle_true)
        rotate_traj(filtered_pred_x, filtered_pred_y, initial_angle_pred)
        print("filtered_true_x x trajectory rotated:")
        print(filtered_true_x)
        print("filtered_true_y y. trajectory rotated:")
        print(filtered_true_y)
        print("filtered_pred_x x traj. rotated")
        print(filtered_pred_x)
        print("filtered_pred_y y traj. rotated")
        print(filtered_pred_y)
        print("------------------------------------")

        
        filtered_true_x = vectorize_traj_point_arr((filtered_true_x))
        filtered_true_y = vectorize_traj_point_arr((filtered_true_y))
        filtered_pred_x = vectorize_traj_point_arr((filtered_pred_x))
        filtered_pred_y = vectorize_traj_point_arr((filtered_pred_y))
        <----------------------------------------------------------------------------->
        '''
        # labels for legend
        true_ped_text = 'ped '+str(ped_id)+' true'
        pred_ped_text = 'ped '+str(ped_id)+' pred.'
        #print("ped id : ", ped_id, "target id :", target_id)
        if ped_id == target_id: # if this is target ped, plot predicted line as well
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print("target_id")
            print(filtered_true_x, filtered_true_y)
            print(filtered_pred_x, filtered_pred_y)

            true_target_id_values = [filtered_true_x[0], filtered_true_y[0]]
            pred_target_id_values = [filtered_pred_x[0], filtered_pred_y[0]]
            true_ped_text = 'target ped '+str(ped_id)+' true'
            pred_ped_text = 'target ped '+str(ped_id)+' pred.'
            #filtered_pred_x = filtered_pred_x - pred_target_id_values[0]
            #filtered_pred_y = filtered_pred_y - pred_target_id_values[1]
            target_sequence_pred = [filtered_pred_x, filtered_pred_y]
            pred_line = plt.plot(filtered_pred_x, filtered_pred_y, linestyle=next(lines), marker=next(marker), linewidth = line_width, label = pred_ped_text)
            #annotate_plot(plt, filtered_pred_x, filtered_pred_y, list(range(1, obs_length+1))) #annotation is off
        #filtered_true_x = filtered_true_x - true_target_id_values[0]
        #filtered_true_y = filtered_true_y - true_target_id_values[1]
        if ped_id == target_id:
            target_sequence_true = [filtered_true_x, filtered_true_y]

        true_line = plt.plot(filtered_true_x, filtered_true_y, linestyle=next(lines), marker=next(marker), linewidth = line_width ,label = true_ped_text)
        video_plot_trajs.append([[filtered_true_x, filtered_true_y], [filtered_pred_x, filtered_pred_y]]) # will be used for a video for each frame

    plt.gca().legend(loc='best') # legend adjustment
    # center trajectories in plot
    plt.gca().set_xlim([min_r - true_target_id_values[0] - plot_offset, max_r - true_target_id_values[0]+ plot_offset])
    plt.gca().set_ylim([min_r - true_target_id_values[1]- plot_offset, max_r - true_target_id_values[1]+ plot_offset])

    # save figure
    plt.savefig(plot_directory+'/'+name+'.png')
    plt.gcf().clear()
    # video animation
    create_plot_animation(plt, video_plot_trajs, processed_ped_index, target_id, real_inv_lookup, obs_length, markers, colors, name, frames, true_target_id_values, plot_directory, style, num_of_color)
    plt.close()
    return [target_sequence_true, target_sequence_pred]

def create_plot_animation(plt, trajs, shuffled_ped_ids, target_id, inv_lookup ,seq_lenght, marker, colors, name, data_frames, true_target_id_values, plot_directory, style, num_of_color):
    # method to create a video for each frame
    print("Video creation for ", name, " is starting...")
    min_r = style[0]
    max_r = style[1]
    plot_offset = style[2]
    plt.gca().set_xlim([min_r - true_target_id_values[0] - plot_offset, max_r - true_target_id_values[0]+ plot_offset])
    plt.gca().set_ylim([min_r - true_target_id_values[1]- plot_offset, max_r - true_target_id_values[1]+ plot_offset])
    frames = []
    marker_cycle = itertools.cycle(marker)
    marker_arr = []

    for frame_num in range(0, seq_lenght-1):
        frame_obj = []
        plt.gca().set_prop_cycle(cycler('color', colors))
        for elem_index ,index in enumerate(shuffled_ped_ids):
            filtered_true_x = trajs[index][0][0]
            filtered_true_y = trajs[index][0][1]
            filtered_pred_x = trajs[index][1][0]
            filtered_pred_y = trajs[index][1][1]
            real_true_index = min(frame_num, len(filtered_true_x))
            real_pred_index = min(frame_num, len(filtered_pred_x))
            marker_cycle = get_marker_style(marker, marker_cycle , elem_index, num_of_color)
            if frame_num == 0:
                curr_marker = next(marker_cycle)
                marker_arr.append(curr_marker)
            #if len(filtered_true_x) < frame_num or len(filtered_pred_x) < frame_num:
            #    del trajs[index]
            #    continue
            ped_id = inv_lookup[index]
            selected_frames = data_frames[0:frame_num+1]
            frame_str = "Frame: "+str(int(selected_frames[-1]))
            true_ped_text = 'ped '+str(ped_id)+' true'
            pred_ped_text = 'ped '+str(ped_id)+' pred.'
            if ped_id == target_id:
                true_ped_text = 'target ped '+str(ped_id)+' true'
                pred_ped_text = 'target ped '+str(ped_id)+' pred.'
                if frame_num == 0:
                    pred_sc = plt.scatter(filtered_pred_x[0:real_pred_index+1], filtered_pred_y[0:real_pred_index+1], marker= marker_arr[elem_index], label = pred_ped_text)


                else:
                    pred_sc = plt.scatter(filtered_pred_x[0:real_pred_index+1], filtered_pred_y[0:real_pred_index+1], marker= marker_arr[elem_index])


                frame_obj.append(pred_sc)

            if frame_num == 0:
                true_sc = plt.scatter(filtered_true_x[0:real_true_index+1], filtered_true_y[0:real_true_index+1], marker= marker_arr[elem_index], label = true_ped_text)

            else: 
                true_sc = plt.scatter(filtered_true_x[0:real_true_index+1], filtered_true_y[0:real_true_index+1], marker= marker_arr[elem_index])

            frame_obj.append(true_sc)

        title = plt.text(0.5,1.01,frame_str, ha="center",va="bottom",color=np.random.rand(3),
                 transform=plt.gca().transAxes, fontsize="large")
        frame_obj.append(title)
        frames.append(frame_obj)
    
    plt.gca().legend(loc='best')
    ani = animation.ArtistAnimation(plt.gcf(), frames, interval=1200, blit=False, repeat_delay=1000)
    ani.save(plot_directory+'/'+name+'.mp4')
    print("Video creation ended.")

def vectorize_traj(traj, nodesPresent, look_up):
    #wrap up for vectorizing traj
    traj, _ = vectorize_seq(Variable(torch.FloatTensor(traj)), nodesPresent, look_up)
    traj = np.array([seq.data.numpy() for seq in traj])
    return traj

def vectorize_traj_point_arr(traj):
    # convert absolute points to vectors
    first_val = traj[0]
    traj = [(point - first_val) for point in traj]
    return traj

def rotate_traj(traj_x, traj_y, angle):
    # rotate trajectory
    origin = (0, 0)
    for p_index in range(len(traj_x)):
        rotated_points = rotate(origin, (traj_x[p_index], traj_y[p_index]), angle)
        traj_x[p_index] = rotated_points[0]
        traj_y[p_index] = rotated_points[1]

def get_overall_max_min(dict_true, dict_pred):
    # aggregate predicted and true min - max values
    overaall_max_x = max(dict_true['x'][0], dict_pred['x'][0])
    overaall_min_x = min(dict_true['x'][1], dict_pred['x'][1])
    overaall_max_y = max(dict_true['y'][0], dict_pred['y'][0])
    overaall_min_y = min(dict_true['y'][1], dict_pred['y'][1])

    return overaall_max_x, overaall_min_x, overaall_max_y, overaall_min_y

def calculate_traj_errors(trajs, num_of_traj):
    # clacluation of avarage error givent trajectory
    err_values = []
    return_errs = []
    for index, traj in enumerate(trajs):


        true_traj = traj[0]
        pred_traj = traj[1]

        Pedlist_seq = traj[2]
        lookup_seq = traj[3]
        # create arrya for calculation of erros
        concated_pred = np.transpose(np.vstack((pred_traj[0], pred_traj[1])))
        concated_true = np.transpose(np.vstack((true_traj[0], true_traj[1])))
        # get error values
        err = get_mean_traj_error((concated_pred), (concated_true))
        f_err = get_final_traj_error((concated_pred), (concated_true))
        return_errs.append([err, f_err])
        #find avarage error
        av_err = (err + f_err)/2
        err_values.append(av_err)               
    biggest_indexes = np.array(err_values).argsort()[-num_of_traj:]
    return biggest_indexes, return_errs

def get_mean_traj_error(true_trajs, pred_trajs):
    # calculate mean trajectory error
    error = 0
    counter = 0
    for (true_traj, pred_traj) in zip(true_trajs, pred_trajs):
        print("true traj: ", true_traj, "pred_traj: " , pred_traj)
        error += np.linalg.norm(pred_traj - true_traj)
        print("error: ", error)
        counter += 1

    if counter != 0:
        error = error / counter

    return error

def get_final_traj_error(true_trajs, pred_trajs):
    # calculate final trajectory error
    last_true_point = true_trajs[-1]
    last_pred_point = pred_trajs[-1]
    error = np.linalg.norm(last_pred_point - last_true_point)
    print("true traj: ", last_true_point, "pred_traj: " , last_pred_point, "error: ", error)

    return error


def plot_target_trajs(trajs, plot_directory, num_of_traj, plot_offset):
    # method will be plot all target ped trajectories for each sequence in final
    line_width = 3
    num_of_color = 20
    fig_width = 10
    fig_height = 10
    precision = 4

    min_r = -10
    max_r = 10
    mpl.rcParams['lines.markersize'] = 10
    marker = ['+', '.', 'o', '*', 'v','^', '<', '>', '1','2','3','4','8','s','p','P','h','x','X','D']
    line_style = ["--","-.",":"]
    name = "target_trajs"
    shuffle(marker)
    shuffle(line_style)

    fig = plt.figure(figsize=(fig_width, fig_width))
    props = dict(fontsize=12)
    plt.title('Final plot of target sequences')

    cm = plt.get_cmap('tab20')
    colors = [cm(i) for i in np.linspace(0, 1, num_of_color)]

    plt.gca().set_prop_cycle(cycler('color', colors))
    marker_cycle = itertools.cycle(marker)
    line_cycle = itertools.cycle((line_style))

    
    shuffle(colors)

    # calculate error for each trajectory and return error values and their indices in descending order
    biggest_err_index, err_values = calculate_traj_errors(trajs, num_of_traj)
    #print(biggest_err_index)
    #selected_err_trajs = np.array(trajs)[biggest_err_index]
    seq_str = "| "
    #dict_true = np.max([np.max(true_traj) for true_traj in selected_err_trajs[:, 0]])
    #print(selected_err_trajs)

    #dict_pred = get_min_max_in_sequence(selected_err_trajs)#only consider prediciton of target id because of model
    #print(dict_pred)
    #print(dict_true)
    #overaall_max_x, overaall_min_x, overaall_max_y, overaall_min_y = get_overall_max_min(dict_true, dict_pred)
    overall_max_x = -1000
    overall_max_y = -1000
    overall_min_x = 1000
    overall_min_y = 1000

    for arr_index, index in enumerate(biggest_err_index):
        true_traj = trajs[index][0]
        pred_traj = trajs[index][1]
        #seq_num = biggest_err_index[arr_index]
        marker_cycle = get_marker_style(marker, marker_cycle , arr_index*2, num_of_color)
        line_cycle = get_marker_style(line_style, line_cycle, arr_index*2, num_of_color)

        mean_err = err_values[index][0]
        final_err = err_values[index][1]
        seq_str = seq_str + 'R: '+str(arr_index+1) + ' S: '+ str(index) + ' ME: '+ f"{mean_err:.{precision}}"+ ' FE: '+ f"{final_err:.{precision}}"+' | '
        true_txt = 'sequence '+str(index)+ ' T'
        pred_txt = 'sequence '+str(index)+ ' P'

        # aggragate min - max values for scaling
        overall_max_x = max(np.max(true_traj[0]), np.max(pred_traj[0]), overall_max_x)
        overall_min_x = min(np.min(true_traj[0]), np.min(pred_traj[0]), overall_min_x)
        overall_max_y = max(np.max(true_traj[1]), np.max(pred_traj[1]), overall_max_y)
        overall_min_y = min(np.min(true_traj[1]), np.min(pred_traj[1]), overall_min_y)

        print("sequence: ", str(index))
        print(true_traj[0], true_traj[1])
        print("pred: ")
        print(pred_traj[0], pred_traj[1])
        print("**************************")
        true_line = plt.plot(true_traj[0], true_traj[1], linestyle=next(line_cycle), marker=next(marker_cycle), linewidth = line_width ,label = true_txt)
        pred_line = plt.plot(pred_traj[0], pred_traj[1], linestyle=next(line_cycle), marker=next(marker_cycle), linewidth = line_width, label = pred_txt)
    
    
    #plot adjustmment
    plt.gca().set_xlim([overall_min_x - plot_offset, overall_max_x + plot_offset])
    plt.gca().set_ylim([overall_min_y - plot_offset, overall_max_y + plot_offset])
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, loc='upper left', prop={'size':6}, bbox_to_anchor=(1,1))
    plt.gca().set_title("\n".join(wrap(seq_str,80)), props, loc ='center')
    plt.tight_layout(pad=20),
    plt.subplots_adjust(left=0.1, right=0.87, top=0.8, bottom=0.05)
    plt.savefig(plot_directory+'/'+name+'.png')
    plt.gcf().clear()
    plt.close()






def main():


    parser = argparse.ArgumentParser()

    # frame rate of video
    parser.add_argument('--frame', type=int, default=1,
                        help='Frame of video created from plots')
    # gru model
    parser.add_argument('--gru', action="store_true", default=False,
                        help='Visualization of GRU model')
    # number of validation dataset
    parser.add_argument('--num_of_data', type=int, default=3,
                        help='Number of validation data will be visualized (If 0 is given, will work on test data mode)')
    # drive support
    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    # minimum lenght of trajectory
    parser.add_argument('--min_traj', type=int,  default=3,
                        help='Min. treshold of number of frame to be removed from a sequence')
    # percentage of peds will be taken for each frame
    parser.add_argument('--max_ped_ratio', type=float,  default=0.8,
                        help='Percentage of pedestrian will be illustrated in a plot for a sequence')
    # maximum ped numbers
    parser.add_argument('--max_target_ped', type=int,  default=20,
                        help='Maximum number of peds in final plot')
    # method to be visualized
    parser.add_argument('--method', type=int, default=1,
                        help='Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)')


    # Parse the parameters
    args = parser.parse_args()

    prefix = ''
    f_prefix = '.'
    if args.drive is True:
      prefix='drive/semester_project/social_lstm_final/'
      f_prefix = 'drive/semester_project/social_lstm_final'

    model_name = "LSTM"
    method_name = get_method_name(args.method)
    if args.gru:
        model_name = "GRU"

    
    plot_file_directory = 'validation'

    # Directories
    if args.num_of_data is 0:
        plot_file_directory = 'test'

    # creation of paths
    save_plot_directory = os.path.join(f_prefix, 'plot',method_name, model_name,'plots/')
    plot_directory = os.path.join(f_prefix, 'plot', method_name, model_name, plot_file_directory)
    video_directory = os.path.join(f_prefix, 'plot',method_name, model_name,'videos/')
    plot_file_name = get_all_file_names(plot_directory)
    num_of_data = np.clip(args.num_of_data, 0, len(plot_file_name))
    plot_file_name = random.sample(plot_file_name, num_of_data)

    
    for file_index in range(len(plot_file_name)):
        file_name = plot_file_name[file_index]
        folder_name = remove_file_extention(file_name)
        print("Now processing: ", file_name)

        file_path = os.path.join(plot_directory, file_name)
        video_save_directory = os.path.join(video_directory, folder_name)
        figure_save_directory = os.path.join(save_plot_directory, folder_name)

        # remove existed plots
        clear_folder(video_save_directory)
        clear_folder(figure_save_directory)


        if not os.path.exists(video_save_directory):
            os.makedirs(video_save_directory)
        if not os.path.exists(figure_save_directory):
            os.makedirs(figure_save_directory)
        

        try:
            f = open(file_path, 'rb')
        except FileNotFoundError:
            print("File not found: %s"%file_path)
            continue


        results = pickle.load(f)
        result_arr = np.array(results)
        true_trajectories = np.array(result_arr[:,0])
        pred_trajectories = np.array(result_arr[:,1])
        frames = np.array(result_arr[:, 4])

        target_id_trajs = []
        args.max_target_ped = np.clip(args.max_target_ped, 0, len(results)-1)
        
        min_r = -10
        max_r = 10
        plot_offset = 1

        for i in range(len(results)):
            print("##########################################################################################")
            name = 'sequence' + str(i).zfill(5)
            print("Now processing seq: ",name)

            if args.num_of_data is 0: #test data visualization
                target_traj = plot_trajectories(results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], name, figure_save_directory,  args.min_traj ,args.max_ped_ratio, results[i][5], [min_r, max_r, plot_offset], results[i][6])
            else:
                target_traj =  plot_trajectories(results[i][0], results[i][1], results[i][2], results[i][3],results[i][4], name, figure_save_directory, args.min_traj ,args.max_ped_ratio, results[i][5], [min_r, max_r, plot_offset], 20)
            target_traj.append(results[i][2])#pedlist
            target_traj.append(results[i][3])#lookup
            target_id_trajs.append(target_traj)

        
        save_video(figure_save_directory, video_save_directory, plot_file_name[file_index], args.frame)
        plot_target_trajs(target_id_trajs, figure_save_directory, args.max_target_ped, plot_offset)

if __name__ == '__main__':
    main()
