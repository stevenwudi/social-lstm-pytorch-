import os
import pickle
import numpy as np
import random
import json


class Synthia_DataLoader():

    def __init__(self,
                 data_root='../data/synthia/SYNTHIA-SEQS-01',
                 img_dir='/media/samsumg_1tb/synthia/Datasets',
                 leaveDataset=0,
                 batch_size=50,
                 seq_length=24,
                 datasets=[i for i in range(9)],
                 dataset_dim=(760, 1280),
                 forcePreProcess=False,
                 infer=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        datasets : The indices of the datasets to use
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        # List of data directories where raw data resides
        self.data_dirs = os.listdir(data_root)
        self.data_root = data_root
        self.dataset_dim = dataset_dim
        # Data directory where the pre-processed pickle file resides
        self.img_dir = img_dir
        self.used_data_dirs = [self.data_dirs[x] for x in datasets]
        self.infer = infer
        # Number of datasets
        self.numDatasets = len(self.data_dirs)

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Define the path in which the process data would be stored
        if infer:
            data_file = os.path.join(os.path.abspath(os.path.join(self.data_root, os.pardir)),
                                     self.data_root.split('/')[-1] +"_use_%d.cpkl"%datasets[0])
        else:
            data_file = os.path.join(os.path.abspath(os.path.join(self.data_root, os.pardir)),
                                     self.data_root.split('/')[-1] +"_leave_%d.cpkl"%leaveDataset)

        # If the file doesn't exist or forcePreProcess is true
        if not(os.path.exists(data_file)) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            # Preprocess the data from the csv files of the datasets
            # Note that this data is processed in frames
            self.frame_preprocess(self.data_root, self.used_data_dirs, data_file)

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)
        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer()

    def frame_preprocess(self, data_root, data_dirs, data_file):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        '''
        # all_frame_data would be a list of list of numpy arrays corresponding to each dataset
        # Each numpy array will correspond to a frame and would be of size (numPeds, 3) each row
        # containing pedID, x, y
        all_frame_data = []

        # numCars_data would be a list of lists corresponding to each dataset
        # Ech list would contain the number of cars in each frame in the dataset
        numCars_data = []
        # Index of the current dataset
        dataset_index = 0

        # For each dataset
        for directory in data_dirs:
            # Define path of the csv file of the current dataset
            json_file_path = os.path.join(data_root, directory)
            with open(json_file_path) as fp:
                car_tracking_dict = json.load(fp)

            # Frame IDs of the frames in the current dataset
            car_id_frames_dict = self.retrieve_car_id_frames(car_tracking_dict)

            # Initialize the list of numPeds for the current dataset
            numCars_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])

            for instance_number_in_seq in sorted(car_tracking_dict.keys()):
                for track_num in range(len(car_tracking_dict[instance_number_in_seq]['tracking_rect'])):
                    num_tracked_frames = len(car_tracking_dict[instance_number_in_seq]['tracking_rect'][track_num])
                    # we only consider tracked cars with more than total_frames (23 frames)
                    if num_tracked_frames >= self.seq_length:

                        for im_num in range(len(car_tracking_dict[instance_number_in_seq]['tracking_rect'][track_num])-self.seq_length-1):

                            for frame_num in range(im_num, im_num + self.seq_length):
                                carsWithPos = []
                                feature_img_name = self.img_dir + '/' + directory[:-5] + '/' +car_tracking_dict[instance_number_in_seq]['img_list'][track_num][frame_num]
                                # frames are save as 'xxxxxx.png' format
                                [centreX, centreY, height, width, d_min, d_max, d_centre, d_mean] = car_tracking_dict[instance_number_in_seq]['tracking_rect'][track_num][frame_num]
                                carsWithPos.append([instance_number_in_seq, int(centreX), int(centreY), int(height),
                                                    int(width), int(d_min), int(d_max), feature_img_name])
                                actual_frame = int(car_tracking_dict[instance_number_in_seq]['img_list'][track_num][frame_num][:-4])

                                # find the neighouring cars
                                cars_in_frame = self.find_car_in_the_frame(car_id_frames_dict, actual_frame,  instance_number_in_seq)
                                for c in cars_in_frame:
                                    [c, centreX, centreY, height, width, d_min, d_max, feature_img_name] = self.retrieve_car_info(car_tracking_dict, c, actual_frame)
                                    carsWithPos.append([c,  int(centreX), int(centreY), int(height), int(width),
                                                        int(d_min), int(d_max)])

                                numCars_data[dataset_index].append(len(cars_in_frame)+1)
                                all_frame_data[dataset_index].append(np.array(carsWithPos))

            dataset_index += 1

        f = open(data_file, "wb")
        pickle.dump((all_frame_data, numCars_data), f, protocol=2)
        f.close()
        # Save the tuple (all_frame_data, frameList_data, numPeds_data) in the pickle file
        for i in range(len(all_frame_data)):
            print('Total frame: %d, max cars in frame: %d, mean car in frame : %.2f' %
                  (len(all_frame_data[i]), max(numCars_data[i]), np.mean(np.array(numCars_data[i]))))
        """
        Total frame: 28920, max cars in frame: 5, mean car in frame : 2.09
        Total frame: 26016, max cars in frame: 4, mean car in frame : 1.21
        Total frame: 33720, max cars in frame: 6, mean car in frame : 2.56
        Total frame: 42504, max cars in frame: 5, mean car in frame : 1.68
        Total frame: 30240, max cars in frame: 6, mean car in frame : 2.10
        Total frame: 35832, max cars in frame: 5, mean car in frame : 2.09
        Total frame: 31800, max cars in frame: 4, mean car in frame : 1.96
        Total frame: 38712, max cars in frame: 7, mean car in frame : 2.13
        """

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()
        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.numPedsList = self.raw_data[1]

        counter = 0

        # For each dataset
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            print('Training data from dataset', dataset, ':', len(all_frame_data))
            # Increment the counter with the number of sequences in the current dataset
            counter += int(len(all_frame_data) / (self.seq_length))

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)
        print('Total number of training batches:', self.num_batches * 2)
        # On an average, we need twice the number of batches to cover the data
        # due to randomization introduced
        self.num_batches = self.num_batches * 2

    def next_batch(self):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.data[self.dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)

                # advance the frame pointer to a random point
                # Can not do random update like previously
                self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer()

        return x_batch, y_batch, d

    def next_valid_batch(self, randomUpdate=True):
        '''
        Function to get the next Validation batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.valid_data[self.valid_dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.valid_frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)

                # advance the frame pointer to a random point
                if randomUpdate:
                    self.valid_frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.valid_frame_pointer += self.seq_length

                d.append(self.valid_dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d

    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''
        if not valid:
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0

    def reset_batch_pointer(self):
        '''
        Reset all pointers
        '''
        self.dataset_pointer = 0
        self.frame_pointer = 0

    def retrieve_car_id_frames(self, car_tracking_dict):

        car_id_frames = {}
        for instance_number_in_seq in car_tracking_dict.keys():
            car_id_frames[instance_number_in_seq] = []
            for track_num in range(len(car_tracking_dict[instance_number_in_seq]['tracking_rect'])):
                # we only consider tracked cars with more than total_frames (23 frames)
                # frames are save as 'xxxxxx.png' format
                for im_num in range(len(car_tracking_dict[instance_number_in_seq]['tracking_rect'][track_num])):
                    actual_frame = int(car_tracking_dict[instance_number_in_seq]['img_list'][track_num][im_num][:-4])
                    car_id_frames[instance_number_in_seq].append(actual_frame)
        return car_id_frames

    def find_car_in_the_frame(self, car_id_frames_dict, actual_frame,  instance_number_car):
        cars_in_frame = []
        for instance_number_in_seq in car_id_frames_dict.keys():
            if not instance_number_in_seq == instance_number_car:
                frames = car_id_frames_dict[instance_number_in_seq]
                if actual_frame in frames:
                    # this is car is in the same frame
                    cars_in_frame.append(instance_number_in_seq)
        return cars_in_frame

    def retrieve_car_info(self, car_tracking_dict, c, actual_frame):

        car_info = car_tracking_dict[c]
        for track_num in range(len(car_info['tracking_rect'])):
            # frames are save as 'xxxxxx.png' format
            for im_num in range(len(car_info['tracking_rect'][track_num])):
                frame = int(car_info['img_list'][track_num][im_num][:-4])
                if frame == actual_frame:
                    feature_img_name = self.img_dir + '/' + car_info['img_list'][track_num][im_num]
                    [centreX, centreY, height, width, d_min, d_max, d_centre, d_mean] = \
                        car_info['tracking_rect'][track_num][im_num]
                    return [c, centreX, centreY, height, width, d_min, d_max, feature_img_name]