import torch
import numpy as np
from torch.autograd import Variable
import math

import argparse
import os
import time
import pickle
import subprocess
import random

from utils import DataLoader, WriteOnceDict
from helper import get_all_file_names, delete_file, create_directories, remove_file_extention, add_file_extention, rotate, vectorize_seq


class data_augmentator():
	# class for data augmentation

	def __init__(self,f_prefix, num_of_data, seq_length, val_percent):

		self.base_train_path = 'data/train/'
		self.base_validation_path = 'data/validation/'

		# list of angles will be use for rotation
		self.angles = list(range(0,360,30))
		self.num_of_data = np.clip(num_of_data, 0, len(self.angles) -1)
		self.num_validation_data = math.ceil(self.num_of_data * val_percent) # number of validation dataset
		self.num_train_data = self.num_of_data - self.num_validation_data # number of train dataset
		print("For each dataset -----> Number of additional training dataset: ", self.num_train_data, " Number of validation dataset: ", self.num_validation_data)

		self.num_validation_data =+1
		self.seq_length = seq_length
		self.val_percent = val_percent
		self.f_prefix = f_prefix


		self.dataloader = DataLoader(f_prefix, 1, seq_length , 0 ,forcePreProcess = True, infer = False, generate=True)
		
		# noise parameter definition
		self.noise_std_min = 0.05
		self.noise_std_max = 0.15
		self.noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
		self.noise_mean = 0.0

		# remove datasets from directories for new creation
		self.clear_directories(self.base_train_path)
		self.clear_directories(self.base_validation_path, True)
		self.random_dataset_creation()

	def random_dataset_creation(self):

		self.dataloader.reset_batch_pointer(valid=False)
		dataset_pointer_ins = self.dataloader.dataset_pointer
		dataset_instances = {}
		whole_dataset = []
		random_angles = random.sample(self.angles, self.num_of_data)
		file_name = self.dataloader.get_file_name()
		print("Dataset creation for: ", file_name, " angles: ", random_angles)



		for batch in range(self.dataloader.num_batches):
			start = time.time()
			# Get data
			x, y, d , numPedsList, PedsList, _= self.dataloader.next_batch()
			dir_name = self.dataloader.get_directory_name()
			file_name = self.dataloader.get_file_name()

			# Get the sequence
			x_seq,d_seq ,numPedsList_seq, PedsList_seq = x[0], d[0], numPedsList[0], PedsList[0]

			# convert dense vector
			x_seq , lookup_seq = self.dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)

			if dataset_pointer_ins is not self.dataloader.dataset_pointer:
			    if self.dataloader.dataset_pointer is not 0:
			        whole_dataset.append(dataset_instances)
			        dataset_instances = {}
			        random_angles = random.sample(self.angles, self.num_of_data) # sample new angle
			        self.noise_std = random.uniform(self.noise_std_min, self.noise_std_max) #sample new noise
			        print("Dataset creation for: ", file_name, " angles: ", random_angles)

			    dataset_pointer_ins = self.dataloader.dataset_pointer

			for index, angle in enumerate(random_angles):
				self.noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
				# modify and preprocess dataset
				modified_x_seq = self.submision_seq_preprocess(self.handle_seq(x_seq, lookup_seq, PedsList_seq, angle), self.seq_length, lookup_seq)
				# store modified data points to dict
				self.dataloader.add_element_to_dict(dataset_instances, (dir_name, file_name, index), modified_x_seq)

			end = time.time()
			print('Current file : ', file_name,' Processed trajectory number : ', batch+1, 'out of', self.dataloader.num_batches, 'trajectories in time', end - start)

		# write modified datapoints to txt files
		whole_dataset.append(dataset_instances)
		create_directories(os.path.join(self.f_prefix, self.base_validation_path), self.dataloader.get_all_directory_namelist())
		self.write_modified_datasets(whole_dataset)


	def handle_seq(self, x_seq, lookup_seq, PedsList_seq, angle):
		# add noise and rotate a trajectory
		vectorized_x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)
		modified_x_seq = vectorized_x_seq.clone()
		mean = torch.FloatTensor([self.noise_mean, self.noise_mean])
		stddev =torch.FloatTensor([self.noise_std, self.noise_std])
		origin = (0, 0)

		for ind, frame in enumerate(vectorized_x_seq):
			for ped in PedsList_seq[ind]:
				selected_point = frame[lookup_seq[ped], :]
				# rotate a frame point
				rotated_point = rotate(origin, selected_point, math.radians(angle))
				noise =  torch.normal(mean, stddev).clone()
				# add random noise
				modified_x_seq[ind, lookup_seq[ped], 0] = rotated_point[0] + noise[0]
				modified_x_seq[ind, lookup_seq[ped], 1] = rotated_point[1] + noise[1]
				#modified_x_seq[ind, lookup_seq[ped], :] = torch.cat(rotate(origin, first_values_dict[ped], math.radians(angle))) + modified_x_seq[ind, lookup_seq[ped], :]
				#roatate first frame value as well and add it back to get absoute coordinates
				modified_x_seq[ind, lookup_seq[ped], 0] = (rotate(origin, first_values_dict[ped], math.radians(angle)))[0] + modified_x_seq[ind, lookup_seq[ped], 0]
				modified_x_seq[ind, lookup_seq[ped], 1] = (rotate(origin, first_values_dict[ped], math.radians(angle)))[1] + modified_x_seq[ind, lookup_seq[ped], 1]


		return modified_x_seq
    
	def submision_seq_preprocess(self, x_seq, seq_lenght, lookup_seq):
		# create original txt structure for modified datapoints 
		ret_x_seq_c = x_seq.data.numpy()
		ped_ids = self.dataloader.get_id_sequence(seq_lenght)
		positions_of_peds = [lookup_seq[ped] for ped in ped_ids]
		ret_x_seq_c = ret_x_seq_c[:, positions_of_peds, :]
		ret_x_seq_c_selected = ret_x_seq_c[:,0,:]
		ret_x_seq_c_selected[:,[0,1]] = ret_x_seq_c_selected[:,[1,0]]
		frame_numbers = self.dataloader.get_frame_sequence(seq_lenght)
		id_integrated_seq = np.append(np.array(ped_ids)[:,None], ret_x_seq_c_selected, axis=1)
		frame_integrated_seq = np.append(frame_numbers[:, None], id_integrated_seq, axis=1)

		return frame_integrated_seq

	def write_modified_datasets(self, dataset_instances_store):
		# write constructed txt structure to txt file
		self.dataloader.reset_batch_pointer()

		for dataset_index in range(self.dataloader.numDatasets):
			dataset_instances = dataset_instances_store[dataset_index]
			train_sub_instances = dict(random.sample(dataset_instances.items(), self.num_train_data))
			validation_sub_instances = {k: v for k, v in dataset_instances.items() if k not in train_sub_instances}
			print("*********************************************************************************")
			print("Training datasets are writing for: ", self.dataloader.get_file_name(dataset_index))
			self.write_dict(train_sub_instances, self.base_train_path)
			print("*********************************************************************************")
			print("Validation datasets are writing for: ", self.dataloader.get_file_name(dataset_index))
			self.write_dict(validation_sub_instances, self.base_validation_path)

	def write_dict(self, dict, base_path):
		cleared_direcories = []
		for key, value in dict.items():
			path = os.path.join(self.f_prefix, base_path, key[0])
			ext_removed_file_name = remove_file_extention(key[1])
			file_name = ext_removed_file_name + "_" + str(key[2])
			file_name = add_file_extention(file_name, 'txt')
			self.dataloader.write_dataset(value, file_name, path)
	
	def clear_directories(self, base_path, delete_all = False):
		# delete all files from a directory
		print("Clearing directories...")
		dir_names = self.dataloader.get_all_directory_namelist()
		base_path = os.path.join(self.f_prefix, base_path)
		for dir_ in dir_names:
			dir_path = os.path.join(base_path, dir_)
			file_names = get_all_file_names(dir_path)
			if delete_all:
				base_file_names = []
			else:
				base_file_names = self.dataloader.get_base_file_name(dir_)
			[delete_file(dir_path, [file_name]) for file_name in file_names if file_name not in base_file_names]





def main():
    
	parser = argparse.ArgumentParser()
	# RNN size parameter (dimension of the output/hidden state)
	parser.add_argument('--num_data', type=int, default=5,
	                    help='Number of additional dataset for each one ')
	# lenght of sequence
	parser.add_argument('--seq_length', type=int, default=20,
	                    help='Processing sequence length')
	# allocation percentage between train and validation datasets
	parser.add_argument('--validation', type=float, default=0.1,
	                    help='Percentage of data will be allocated for validation in additional datasets')
	# use of gogle drive
	parser.add_argument('--drive', action="store_true", default=False,
	                    help='Use Google drive or not')

	args = parser.parse_args()
	print(args.num_data," additional dataset will be created for each train dataset")
	print("Sequence lenght: ", args.seq_length)
	print("Percentage of data will be allocated for validation: %", args.validation*100)


	#for drive run
	prefix = ''
	f_prefix = '.'
	if args.drive is True:
	  prefix='drive/semester_project/social_lstm_final/'
	  f_prefix = 'drive/semester_project/social_lstm_final'

	augmentator = data_augmentator(f_prefix, args.num_data, args.seq_length, args.validation)

if __name__ == '__main__':
    main()