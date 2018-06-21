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

#proper text positioning not to overlap annotations
def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = zip(y_data, x_data)
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height) 
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions

def save_video(sequence_path, video_path, video_name ,frame):
    image_input_path = os.path.join(sequence_path, "sequence%05d.png")
    video_output_path = os.path.join(video_path, video_name+'.mp4')
    command = "ffmpeg -r "+str(frame)+" -i "+image_input_path+" -vcodec mpeg4 -y "+video_output_path
    print("Creating video: input sequence --> ",image_input_path," | output video --> ",video_output_path)
    os.system(str(command))

def min_max_scaler(arr, min, max, min_range, max_range):
    arr_std = (arr - min) / (max - min)
    #print(min_range, " - ", max_range," scaled: ", arr_std)
    arr_scaled = (arr_std*(max_range - min_range)) + min_range
    return arr_scaled

def annotate_plot(plt, x_vals, y_vals, annotation):
    texts =[]
    for index,(i,j) in enumerate(zip(x_vals,y_vals)):
        texts.append(plt.text(i, j , annotation[index], fontsize=10, fontweight ='bold'))
    adjust_text(texts, force_points=1.5, expand_points=(3, 3), arrowprops=dict(arrowstyle="->", color='b', alpha=0.5, lw=0.5))

def annotation_plotter(x_data, y_data, text_positions, annotation, axis, txt_width,txt_height):
    for index,(x,y,t) in enumerate(zip(x_data, y_data, text_positions)):
        plt.annotate(annotation[index], xy=(x-txt_width, t), size=12, fontweight ='bold')
        if y != t:
            plt.arrow(x, t,0,y-t, color='red',alpha=0.3, width=txt_width*0.1, 
            head_width=txt_width, head_length=txt_height*0.5, 
            zorder=0,length_includes_head=True)



def calculate_mean_trajectory(traj, nodesPresent, lookup):
    print("nodepresent")
    print(nodesPresent)
    print("lookup table")
    print(lookup)
    traj_length, numNodes, _  = traj.shape
    x_mean = []
    y_mean = []
    seq_pedlist = np.unique(np.concatenate(nodesPresent)).astype(int)
    for tstep in range(traj_length):
        pos = traj[tstep, :]
        pedlist  = nodesPresent[tstep]
        pedlist_index = [lookup[ped] for ped in pedlist]
        pos_selected = pos[pedlist_index,:]
        x_mean.append(pos_selected[:, 0].mean(axis = 0))
        y_mean.append(pos_selected[:, 1].mean(axis = 0))
    print("selected x values (before mean)")
    print(x_mean)
    print("selected y values (before mean)")
    print(y_mean)
    x_mean = np.array(x_mean)
    y_mean = np.array(y_mean)

    x_mean_cleared = x_mean[~(np.isnan(x_mean))]
    y_mean_cleared = y_mean[~(np.isnan(y_mean))]

    return x_mean_cleared, y_mean_cleared, seq_pedlist

def get_min_max_in_trajectories(traj):
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
    if num_of_color == index:
        shuffle(markers)
        marker = itertools.cycle(markers)
        return marker
    return marker_cycle

def get_line_style(linestyle, line_cycle, index, num_of_color):
    if num_of_color == index:
        shuffle(linestyle)
        lines = itertools.cycle((linestyle))
        return lines
    return line_cycle

def plot_trajectories(true_trajs, pred_trajs, nodesPresent, look_up, frames, name, is_mean, plot_directory, min_threshold, max_ped_ratio, target_id , style, obs_length = 20):
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

    name : Name of the plot

    withBackground : Include background or not
    '''
    #FFMpegWriter = manimation.writers['ffmpeg']
    #metadata = dict(title=name, artist='Baran nama',
    #            comment='Trajctory plotting')

    #writer = FFMpegWriter(fps=15, metadata=metadata)


    fig_width = 10
    fig_height = 10

    min_r = style[0]
    max_r = style[1]
    plot_offset = style[2]

    reference_point = (0, max_r)
    target_frame = 1
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
    
    #first points are copy of true traj so we copy it back first predicted vals
    #pred_trajs[0] = pred_trajs[1]
    print("****************************")
    print("orig true_")
    print(true_trajs[:, look_up[target_id], :])
    print("orig pred_")
    print(pred_trajs[:, look_up[target_id], :])
    
    pred_trajs = pred_trajs[1:]
    true_trajs = true_trajs[1:]
    nodesPresent = nodesPresent[1:]
    
    # print('lookup: ', look_up)
    # print('target_id: ', target_id)
    true_trajs = translate(torch.from_numpy(true_trajs), nodesPresent, look_up ,torch.from_numpy(true_trajs[0][look_up[target_id], 0:2]))
    pred_trajs = translate(torch.from_numpy(pred_trajs), nodesPresent, look_up ,torch.from_numpy(pred_trajs[0][look_up[target_id], 0:2]))


    #true_trajs, _= vectorize_seq_with_ped(torch.from_numpy(true_trajs), nodesPresent, look_up ,target_id)
    #pred_trajs, _= vectorize_seq_with_ped(torch.from_numpy(pred_trajs), nodesPresent, look_up ,target_id)
    print("orig true_")
    print(true_trajs[:, look_up[target_id], :])
    print("orig pred_")
    print(pred_trajs[:, look_up[target_id], :])
    
    angle_true = angle_between(reference_point, (true_trajs[target_frame][look_up[target_id], 0], true_trajs[target_frame][look_up[target_id], 1]))
    #angle_pred = angle_between(reference_point, (pred_trajs[target_frame][look_up[target_id], 0], pred_trajs[target_frame][look_up[target_id], 1]))

    true_trajs = rotate_traj_with_target_ped(true_trajs, angle_true, nodesPresent, look_up)
    #pred_trajs = rotate_traj_with_target_ped(pred_trajs, angle_pred, nodesPresent, look_up)
    pred_trajs = rotate_traj_with_target_ped(pred_trajs, angle_true, nodesPresent, look_up)

    print("+++++++++++++++++++++++++++++++++++++++++++++++44")
    print("angle: ", np.rad2deg(angle_true))
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


    fig = plt.figure(figsize=(fig_width, fig_width))
    props = dict(fontsize=12)
    frames_str = ' | '.join(str(int(e)) for e in frames)
    frames_str = "frame ids: " + frames_str
    plt.gca().set_title("\n".join(wrap(frames_str,80)), props, loc ='center')

    txt_height = 0.04*(plt.ylim()[1] - plt.ylim()[0])
    txt_width = 0.02*(plt.xlim()[1] - plt.xlim()[0])

    if is_mean:
        cm = plt.get_cmap('tab10')
    else:
        cm = plt.get_cmap('tab20')
        num_of_color = 20

    colors = [cm(i) for i in np.linspace(0, 1, num_of_color)]
    shuffle(colors)
    plt.gca().set_prop_cycle(cycler('color', colors))
    shuffle(markers)
    shuffle(line_style)
    marker = itertools.cycle(markers)
    lines = itertools.cycle((line_style))
    #plt.text(0.05, 0.95, frames_str, transform=plt.gca().transAxes, loc = 'best', bbox=props)

    traj_length, numNodes, _  = true_trajs.shape
    traj_data = {}
    #look_up = dict(look_up)
    inv_lookup = {v: k for k, v in look_up.items()}
    if not is_mean:
        for tstep in range(traj_length-1):#real traj lenght is traj_lenght-1
                pred_pos = pred_trajs[tstep, :]
                true_pos = true_trajs[tstep, :]

                for ped in range(numNodes):
                    ped_index = ped
                    ped_id = inv_lookup[ped]
                    if ped not in traj_data:
                        traj_data[ped_index] = [[], []]

                    
                    if ped_id in nodesPresent[tstep]:

                    # #no prediction(first frame of observed part)
                    #     if tstep is 0:
                    #         traj_data[ped_index][0].append(true_pos[ped_index, :])
                    #         traj_data[ped_index][1].append([float('nan'), float('nan')])

                    #    else:
                        traj_data[ped_index][0].append(true_pos[ped_index, :])
                        traj_data[ped_index][1].append(pred_pos[ped_index, :])
        

        #print(np.array(traj_data.values()))


        processed_ped_number = 0
        num_of_peds = math.ceil(max_ped_ratio * len(traj_data))
        print("Max number of peds in this  seq.: ",num_of_peds)
        shuffled_ped_ids = list(range(0, len(traj_data)))
        random.shuffle(shuffled_ped_ids)

        #add target id to at the beginnnig of the list therefore it will process always
        shuffled_ped_ids.remove(look_up[target_id])
        shuffled_ped_ids.insert(0, look_up[target_id])


        processed_ped_index = []
        real_inv_lookup = {}
        true_target_id_values = None
        pred_target_id_values = None
        target_sequence_true = None
        target_sequence_pred = None
        for j in shuffled_ped_ids:

            #format_params = []
            #print("Processing ped ", j)
            if processed_ped_number >= num_of_peds:
                break
            true_traj_ped = traj_data[j][0]  # List of [x,y] elements
            pred_traj_ped = traj_data[j][1]
            ped_id = inv_lookup[j]
            #print("ped id : ", ped_id, "target id :", target_id)

            #true_x = [(p[0]+1)/2*height if not np.isnan(p[0]) else p[0] for p in true_traj_ped]
            true_x = [p[0] for p in true_traj_ped]
            #true_y = [(p[1]+1)/2*width if not np.isnan(p[0]) else p[0] for p in true_traj_ped]
            true_y = [p[1] for p in true_traj_ped]
            #pred_x = [(p[0]+1)/2*height if not np.isnan(p[0]) else p[0] for p in pred_traj_ped]

            pred_x = [p[0] for p in pred_traj_ped]
            #pred_y = [(p[1]+1)/2*width if not np.isnan(p[0]) else p[0] for p in pred_traj_ped]
            pred_y = [p[1] for p in pred_traj_ped]

            real_inv_lookup[processed_ped_number] = ped_id

            if not len(true_x) > min_threshold or not len(true_x) > 2:
                print("Ped processing is aborted because its trajectory lenght in this sequence is smaller than threshold or 1 points")
                continue
            else:
                processed_ped_index.append(processed_ped_number)

                processed_ped_number = processed_ped_number +1

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
            

            filtered_true_x = min_max_scaler(np.array([x for x in true_x if not np.isnan(x)]), overaall_min_x, overaall_max_x, min_r, max_r)
            filtered_true_y = min_max_scaler(np.array([y for y in true_y if not np.isnan(y)]), overaall_min_y, overaall_max_y, min_r, max_r)
            filtered_pred_x = min_max_scaler(np.array([x for x in pred_x if not np.isnan(x)]), overaall_min_x, overaall_max_x, min_r, max_r)
            filtered_pred_y = min_max_scaler(np.array([y for y in pred_y if not np.isnan(y)]), overaall_min_y, overaall_max_y, min_r, max_r)


            
            #overaall_max_x, overaall_min_x, overaall_max_y, overaall_min_y = get_overall_max_min(dict_arr[0], dict_arr[1])



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
            '''
            true_ped_text = 'ped '+str(ped_id)+' true'
            pred_ped_text = 'ped '+str(ped_id)+' pred.'
            #print("ped id : ", ped_id, "target id :", target_id)
            if ped_id == target_id:
                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                print("target_id")
                print(filtered_true_x, filtered_true_y)
                print(filtered_pred_x, filtered_pred_y)
                true_target_id_values = [filtered_true_x[0], filtered_true_y[0]]
                pred_target_id_values = [filtered_pred_x[0], filtered_pred_y[0]]
                true_ped_text = 'target ped '+str(ped_id)+' true'
                pred_ped_text = 'target ped '+str(ped_id)+' pred.'
                filtered_pred_x = filtered_pred_x - pred_target_id_values[0]
                filtered_pred_y = filtered_pred_y - pred_target_id_values[1]
                target_sequence_pred = [filtered_pred_x, filtered_pred_y]
                pred_line = plt.plot(filtered_pred_x, filtered_pred_y, linestyle=next(lines), marker=next(marker), linewidth = line_width, label = pred_ped_text)
                #annotate_plot(plt, filtered_pred_x, filtered_pred_y, list(range(1, obs_length+1)))
            filtered_true_x = filtered_true_x - true_target_id_values[0]
            filtered_true_y = filtered_true_y - true_target_id_values[1]
            if ped_id == target_id:
                target_sequence_true = [filtered_true_x, filtered_true_y]

            true_line = plt.plot(filtered_true_x, filtered_true_y, linestyle=next(lines), marker=next(marker), linewidth = line_width ,label = true_ped_text)
            video_plot_trajs.append([[filtered_true_x, filtered_true_y], [filtered_pred_x, filtered_pred_y]])




            #annotate_plot(plt, filtered_true_x, filtered_true_y, list(range(1, obs_length+1)))

            '''
            overall_trajs.append([filtered_true_x, filtered_true_y, filtered_pred_x, filtered_pred_y, [next(lines), next(marker), true_ped_text, pred_ped_text]])
            overall_max_arr_x.append(max(max(filtered_true_x), max(filtered_pred_x)))
            overall_min_arr_y.append(min(min(filtered_true_y), min(filtered_pred_y)))
            overall_max_arr_y.append(max(max(filtered_true_y), max(filtered_pred_y)))
            overall_min_arr_x.append(min(min(filtered_true_x), min(filtered_pred_x)))
            '''

            
        #plot_trajs(plt, overall_trajs, max(overall_max_arr_x), min(overall_min_arr_x), max(overall_max_arr_y), min(overall_min_arr_y),max_r, min_r)
        plt.gca().legend(loc='best')
        plt.gca().set_xlim([min_r - true_target_id_values[0] - plot_offset, max_r - true_target_id_values[0]+ plot_offset])
        plt.gca().set_ylim([min_r - true_target_id_values[1]- plot_offset, max_r - true_target_id_values[1]+ plot_offset])


    else:
        print("calculating mean of true traj.")
        true_x_mean, true_y_mean, true_seq_pedlist = calculate_mean_trajectory(true_trajs, nodesPresent, look_up)
        print("calculating mean of pred traj.")
        pred_x_mean, pred_y_mean, pred_seq_pedlist = calculate_mean_trajectory(pred_trajs[1::], nodesPresent, look_up)
        
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("true_x after mean")
        print(true_x_mean)
        print("true_y after mean")
        print(true_y_mean)
        print("pred_x after mean")
        print(pred_x_mean)
        print("pred_y after mean")
        print(pred_y_mean)

        min_x = np.min(np.concatenate((true_x_mean, pred_x_mean),axis=0))
        max_x = np.max(np.concatenate((true_x_mean, pred_x_mean),axis=0))
        min_y = np.min(np.concatenate((true_y_mean, pred_y_mean),axis=0))
        max_y = np.max(np.concatenate((true_y_mean, pred_y_mean), axis=0))

        
        true_x_mean = min_max_scaler(true_x_mean, min_x, max_x, min_r, max_r)
        true_y_mean = min_max_scaler(true_y_mean, min_y, max_y, min_r, max_r)

        pred_x_mean = min_max_scaler(pred_x_mean, min_x, max_x, min_r, max_r)
        pred_y_mean = min_max_scaler(pred_y_mean, min_y, max_y, min_r, max_r)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("true_x after mean and scaling")
        print(true_x_mean)
        print("true_y after mean and scaling")
        print(true_y_mean)
        print("pred_x after mean and scaling")
        print(pred_x_mean)
        print("pred_y after mean and scaling")
        print(pred_y_mean)
        pedlist_str = ' | '.join(str(e) for e in true_seq_pedlist)
        true_line = plt.plot(true_x_mean, true_y_mean, linestyle=next(lines), linewidth = line_width, marker=next(marker), label = 'true_traj.')
        pred_line = plt.plot(pred_x_mean, pred_y_mean, linestyle=next(lines), linewidth = line_width, marker=next(marker), label = 'predicted traj. ')
        #text_positions_true = get_text_positions(true_x_mean, true_y_mean, txt_width, txt_height)
        #text_positions_pred = get_text_positions(pred_x_mean, pred_y_mean, txt_width, txt_height)
        #annotation_plotter(true_x_mean, true_y_mean, text_positions_true, list(range(1, obs_length+1)), plt, txt_width,txt_height)
        #annotation_plotter(pred_x_mean, pred_y_mean, text_positions_pred, list(range(1, traj_length)), plt, txt_width,txt_height)

        annotate_plot(plt, true_x_mean, true_y_mean, list(range(1, obs_length+1)))
        annotate_plot(plt, pred_x_mean, pred_y_mean, list(range(1, traj_length)))
        
        plt.gca().legend(loc = 'best', title='ped ids: '+ "\n".join(wrap(pedlist_str, 20)))


    plt.savefig(plot_directory+'/'+name+'.png')
    plt.gcf().clear()
    create_plot_animation(plt, video_plot_trajs, processed_ped_index, target_id, real_inv_lookup, obs_length, markers, colors, name, frames, true_target_id_values, plot_directory, style, num_of_color)
    plt.close()
    return [target_sequence_true, target_sequence_pred]

def create_plot_animation(plt, trajs, shuffled_ped_ids, target_id, inv_lookup ,seq_lenght, marker, colors, name, data_frames, true_target_id_values, plot_directory, style, num_of_color):
    print("Video creation for ", name, " is starting...")
    min_r = style[0]
    max_r = style[1]
    plot_offset = style[2]
    plt.gca().set_xlim([min_r - true_target_id_values[0] - plot_offset, max_r - true_target_id_values[0]+ plot_offset])
    plt.gca().set_ylim([min_r - true_target_id_values[1]- plot_offset, max_r - true_target_id_values[1]+ plot_offset])
    frames = []
    marker_cycle = itertools.cycle(marker)
    marker_arr = []
    #marker_arr = random.sample(marker, len(trajs))
    #print(shuffled_ped_ids)
    #print(inv_lookup)
    #print(target_id)
    #print(len(trajs))
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
                    #pred_sc = plt.scatter(pred_x, pred_y, marker= marker_arr[index], label = pred_ped_text)
                    pred_sc = plt.scatter(filtered_pred_x[0:real_pred_index+1], filtered_pred_y[0:real_pred_index+1], marker= marker_arr[elem_index], label = pred_ped_text)


                else:
                    #pred_sc = plt.scatter(pred_x, pred_y, marker= marker_arr[index])
                    pred_sc = plt.scatter(filtered_pred_x[0:real_pred_index+1], filtered_pred_y[0:real_pred_index+1], marker= marker_arr[elem_index])


                frame_obj.append(pred_sc)

            if frame_num == 0:
                #true_sc = plt.scatter(true_x, true_y, marker=marker_arr[index], label = true_ped_text)
                true_sc = plt.scatter(filtered_true_x[0:real_true_index+1], filtered_true_y[0:real_true_index+1], marker= marker_arr[elem_index], label = true_ped_text)

            else: 
                #true_sc = plt.scatter(true_x, true_y, marker=marker_arr[index])
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
    first_val = traj[0]
    traj = [(point - first_val) for point in traj]
    return traj

def rotate_traj(traj_x, traj_y, angle):
    origin = (0, 0)
    for p_index in range(len(traj_x)):
        rotated_points = rotate(origin, (traj_x[p_index], traj_y[p_index]), angle)
        traj_x[p_index] = rotated_points[0]
        traj_y[p_index] = rotated_points[1]

def get_overall_max_min(dict_true, dict_pred):

    overaall_max_x = max(dict_true['x'][0], dict_pred['x'][0])
    overaall_min_x = min(dict_true['x'][1], dict_pred['x'][1])
    overaall_max_y = max(dict_true['y'][0], dict_pred['y'][0])
    overaall_min_y = min(dict_true['y'][1], dict_pred['y'][1])

    return overaall_max_x, overaall_min_x, overaall_max_y, overaall_min_y

# def plot_trajs(plt, traj_arr, max_x, min_x, max_y, min_y, max_r, min_r):
#     for traj in traj_arr:
#         print("max_x: ", max_x, "min_x", min_x)
#         print("max_y: ", max_y, "min_y", min_y)
#         print("filtered_true_x x trajectory:")
#         print(traj[0])
#         print("filtered_true_y y. trajectory:")
#         print(traj[1])
#         print("filtered_pred_x x traj.")
#         print(traj[2])
#         print("filtered_pred_y y traj.")
#         print(traj[3])
#         print("##########################################3")
#         filtered_true_x = vectorize_traj_point_arr(min_max_scaler(traj[0], min_x, max_x, min_r, max_r))
#         filtered_true_y = vectorize_traj_point_arr(min_max_scaler(traj[1], min_y, max_y, min_r, max_r))
#         filtered_pred_x = vectorize_traj_point_arr(min_max_scaler(traj[2], min_x, max_x, min_r, max_r))
#         filtered_pred_y = vectorize_traj_point_arr(min_max_scaler(traj[3], min_y, max_y, min_r, max_r))
#         print("filtered_true_x x trajectory:")
#         print(filtered_true_x)
#         print("filtered_true_y y. trajectory:")
#         print(filtered_true_y)
#         print("filtered_pred_x x traj.")
#         print(filtered_pred_x)
#         print("filtered_pred_y y traj.")
#         print(filtered_pred_y)
#         print("**************************************")
#         line_style = traj[4][0]
#         line_marker = traj[4][1]
#         true_ped_text = traj[4][2]
#         pred_ped_text = traj[4][3]
#         line_width = 3
#         true_line = plt.plot(filtered_true_x, filtered_true_y, linestyle=line_style, marker=line_marker, linewidth = line_width ,label = true_ped_text)
#         pred_line = plt.plot(filtered_pred_x, filtered_pred_y, linestyle=line_style, marker=line_marker, linewidth = line_width, label = pred_ped_text)

def calculate_traj_errors(trajs, num_of_traj):
    err_values = []
    return_errs = []
    for index, traj in enumerate(trajs):


        true_traj = traj[0]
        pred_traj = traj[1]

        Pedlist_seq = traj[2]
        lookup_seq = traj[3]
        #print(true_traj[0], true_traj[1])
        #print(pred_traj[0], pred_traj[1])
        concated_pred = np.transpose(np.vstack((pred_traj[0], pred_traj[1])))
        concated_true = np.transpose(np.vstack((true_traj[0], true_traj[1])))
        #concated = concated.reshape(concated.shape[0], 1, concated.shape[1])
        err = get_mean_traj_error((concated_pred), (concated_true))
        f_err = get_final_traj_error((concated_pred), (concated_true))
        return_errs.append([err, f_err])
        av_err = (err + f_err)/2
        err_values.append(av_err)               
    biggest_indexes = np.array(err_values).argsort()[-num_of_traj:]
    return biggest_indexes, return_errs

def get_mean_traj_error(true_trajs, pred_trajs):
    
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
    
    last_true_point = true_trajs[-1]
    last_pred_point = pred_trajs[-1]
    error = np.linalg.norm(last_pred_point - last_true_point)
    print("true traj: ", last_true_point, "pred_traj: " , last_pred_point, "error: ", error)

    return error


def plot_target_trajs(trajs, plot_directory, num_of_traj, plot_offset):
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


    plot_file_name = ['biwi_eth', 
                    #'crowds_zara01',
                    #'uni_examples', 'coupa_0',
                    #'coupa_1', 'gates_2','hyang_0','hyang_1','hyang_3','hyang_8',
                    #'little_0','little_1','little_2','little_3','nexus_5','nexus_6',
                    #'quad_0','quad_1','quad_2','quad_3'
                    ]
    

    parser = argparse.ArgumentParser()

    # Experiments

    parser.add_argument('--mean', action="store_true", default=False,
                        help='Take mean position of peds for each frame')

    parser.add_argument('--frame', type=int, default=1,
                        help='Frame of video created from plots')

    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')

    parser.add_argument('--num_of_data', type=int, default=3,
                        help='Number of validation data will be visualized (If 0 is given, will work on test data mode)')
    
    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')

    parser.add_argument('--min_traj', type=int,  default=3,
                        help='Min. treshold of number of frame to be removed from a sequence')

    parser.add_argument('--max_ped_ratio', type=float,  default=0.8,
                        help='Percentage of pedestrian will be illustrated in a plot for a sequence')

    parser.add_argument('--max_target_ped', type=int,  default=20,
                        help='Maximum number of peds in final plot')

    parser.add_argument('--method', type=int, default=1,
                        help='Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)')


    # Parse the parameters
    args = parser.parse_args()

    prefix = ''
    f_prefix = '.'
    if args.drive is True:
      prefix='drive/semester_project/new_social_LSTM_pytorch_v2/'
      f_prefix = 'drive/semester_project/new_social_LSTM_pytorch_v2'

    model_name = "LSTM"
    method_name = get_method_name(args.method)
    if args.gru:
        model_name = "GRU"

    
    plot_file_directory = 'validation'

    # Directories
    if args.num_of_data is 0:
        plot_file_directory = 'test'

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

        #delete_file(save_plot_directory, [folder_name])
        #delete_file(video_directory, [folder_name])
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

        #dict_true = get_min_max_in_trajectories(true_trajectories)
        #dict_pred = get_min_max_in_trajectories(pred_trajectories)


        big_lookup_table = {}
        [big_lookup_table.update(data[3]) for data in results]

        unique_ids = np.unique(list(big_lookup_table.keys())).astype(int)
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
                target_traj = plot_trajectories(results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], name, args.mean, figure_save_directory,  args.min_traj ,args.max_ped_ratio, results[i][5], [min_r, max_r, plot_offset], results[i][6])
            else:
                target_traj =  plot_trajectories(results[i][0], results[i][1], results[i][2], results[i][3],results[i][4], name, args.mean, figure_save_directory, args.min_traj ,args.max_ped_ratio, results[i][5], [min_r, max_r, plot_offset], 20)
            target_traj.append(results[i][2])#pedlist
            target_traj.append(results[i][3])#lookup
            target_id_trajs.append(target_traj)

        
        save_video(figure_save_directory, video_save_directory, plot_file_name[file_index], args.frame)
        plot_target_trajs(target_id_trajs, figure_save_directory, args.max_target_ped, plot_offset)

if __name__ == '__main__':
    main()
