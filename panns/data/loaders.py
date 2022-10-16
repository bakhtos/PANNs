import numpy as np
import h5py
import csv
import time
import logging

__all__ = ['AudioSetDataset',
           'SamplerBase',
           'TrainSampler',
           'BalancedTrainSampler',
           'AlternateTrainSampler',
           'EvaluateSampler',
           'collate_fn']

class AudioSetDataset(object):
    """Take meta of the audio clip, return waveform and target vector."""
    def __init__(self):
        pass
    
    def __getitem__(self, meta):
        """Load waveform and target of an audio clip.
        
        :param dict meta: Dictionary which maps 'hdf5_path': str
                          and 'index_in_hdf5': int
        :return data_dict: Dictionary which maps 'audio_name': str,
                          'waveform': numpy.ndarray (shape=(clip_samples,)),
                          'target': numpy.ndarray (shape=(classes_num,))
        :rtype: dict
        """
        hdf5_path = meta['hdf5_path']
        index_in_hdf5 = meta['index_in_hdf5']

        with h5py.File(hdf5_path, 'r') as hf:
            audio_name = hf['audio_name'][index_in_hdf5].decode()
            waveform = hf['waveform'][index_in_hdf5]
            target = hf['target'][index_in_hdf5].astype(np.float32)

        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': target}
            
        return data_dict


class SamplerBase(object):
    def __init__(self, hdf5_index_path, batch_size, random_seed):
        """Base class of train sampler.

          :param str hdf5_index_path: HDF5 index contaning the datatset
          :param int batch_size: Size of the returned batch-meta
          :param int random_seed: Seed for the randomization of sampling
        """
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        # Load target
        load_time = time.time()

        with h5py.File(hdf5_index_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.hdf5_paths = [hdf5_path.decode() for hdf5_path in hf['hdf5_path'][:]]
            self.indexes_in_hdf5 = hf['index_in_hdf5'][:]
            self.targets = hf['target'][:].astype(np.float32)
        
        (self.audios_num, self.classes_num) = self.targets.shape
        logging.info('Training number: {}'.format(self.audios_num))
        logging.info('Load target time: {:.3f} s'.format(time.time() - load_time))


class TrainSampler(SamplerBase):
    def __init__(self, hdf5_index_path, batch_size, random_seed=1234):
        """Regular sampler. Samples by randomly shuffling the list of files.
        
           :param string hdf5_index_path: HDF5 index containing the dataset
           :param int batch_size: Size of the returned batch-meta
           :param int random_seed: Seed for randomization of sampling
        """
        super(TrainSampler, self).__init__(hdf5_index_path, batch_size, random_seed)
        
        self.indexes = np.arange(self.audios_num)
            
        # Shuffle indexes
        self.random_state.shuffle(self.indexes)
        
        self.pointer = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        :return batch_meta: List of dictionaries, each dictionary maps
                            'hdf5_path': str, 'index_in_hdf5': int
        :rtype: list
        """

        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                index = self.indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)
                
                batch_meta.append({
                    'hdf5_path': self.hdf5_paths[index], 
                    'index_in_hdf5': self.indexes_in_hdf5[index]})
                i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'indexes': self.indexes,
            'pointer': self.pointer}
        return state
            
    def load_state_dict(self, state):
        self.indexes = state['indexes']
        self.pointer = state['pointer']


class BalancedTrainSampler(SamplerBase):
    def __init__(self, hdf5_index_path, batch_size, random_seed=1234):
        """Balanced sampler. Data are equally sampled from different sound classes.
        
           :param string hdf5_index_path: HDF5 index contaning the dataset
           :param int batch_size: Size of the returned batch-meta
           :param int random_seed: Seed for sampling randomization
        """
        super(BalancedTrainSampler, self).__init__(hdf5_index_path, 
            batch_size, random_seed)
        
        self.samples_num_per_class = np.sum(self.targets, axis=0)
        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int32)))
        
        # Training indexes of all sound classes. E.g.: 
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []
        
        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.targets[:, k] == 1)[0])
            
        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])
        
        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate batch meta for training. 
        
        :return batch_meta: List of dictionaries, each dictionary maps
                            'hdf5_path': str, 'index_in_hdf5': int
        :rtype: list
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                index = self.indexes_per_class[class_id][pointer]
                
                # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])

                batch_meta.append({
                    'hdf5_path': self.hdf5_paths[index], 
                    'index_in_hdf5': self.indexes_in_hdf5[index]})
                i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'indexes_per_class': self.indexes_per_class, 
            'queue': self.queue, 
            'pointers_of_classes': self.pointers_of_classes}
        return state
            
    def load_state_dict(self, state):
        self.indexes_per_class = state['indexes_per_class']
        self.queue = state['queue']
        self.pointers_of_classes = state['pointers_of_classes']


class AlternateTrainSampler(SamplerBase):
    def __init__(self, hdf5_index_path, batch_size, random_seed=1234):
        """Alternatively sample data from Sampler and Balanced Sampler.
        
           :param string hdf5_index_path: HDF5 index contaning the dataset
           :param int batch_size: Size of the returned batch-meta
           :param int random_seed: Seed for sampling randomization
        """

        self.sampler1 = TrainSampler(hdf5_index_path, batch_size, random_seed)

        self.sampler2 = BalancedTrainSampler(hdf5_index_path, batch_size, random_seed)

        self.batch_size = batch_size
        self.count = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        :return batch_meta: List of dictionaries, each dictionary maps
                            'hdf5_path': str, 'index_in_hdf5': int
        :rtype: list
        """

        batch_size = self.batch_size

        while True:
            self.count += 1

            if self.count % 2 == 0:
                batch_meta = []
                i = 0
                while i < batch_size:
                    index = self.sampler1.indexes[self.sampler1.pointer]
                    self.sampler1.pointer += 1

                    # Shuffle indexes and reset pointer
                    if self.sampler1.pointer >= self.sampler1.audios_num:
                        self.sampler1.pointer = 0
                        self.sampler1.random_state.shuffle(self.sampler1.indexes)
                    
                    batch_meta.append({
                        'hdf5_path': self.sampler1.hdf5_paths[index], 
                        'index_in_hdf5': self.sampler1.indexes_in_hdf5[index]})
                    i += 1

            elif self.count % 2 == 1:
                batch_meta = []
                i = 0
                while i < batch_size:
                    if len(self.sampler2.queue) == 0:
                        self.sampler2.queue = self.sampler2.expand_queue(self.sampler2.queue)

                    class_id = self.sampler2.queue.pop(0)
                    pointer = self.sampler2.pointers_of_classes[class_id]
                    self.sampler2.pointers_of_classes[class_id] += 1
                    index = self.sampler2.indexes_per_class[class_id][pointer]
                    
                    # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                    if self.sampler2.pointers_of_classes[class_id] >= self.sampler2.samples_num_per_class[class_id]:
                        self.sampler2.pointers_of_classes[class_id] = 0
                        self.sampler2.random_state.shuffle(self.sampler2.indexes_per_class[class_id])

                    batch_meta.append({
                        'hdf5_path': self.sampler2.hdf5_paths[index], 
                        'index_in_hdf5': self.sampler2.indexes_in_hdf5[index]})
                    i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'sampler1': self.sampler1.state_dict(), 
            'sampler2': self.sampler2.state_dict()}
        return state

    def load_state_dict(self, state):
        self.sampler1.load_state_dict(state['sampler1'])
        self.sampler2.load_state_dict(state['sampler2'])


class EvaluateSampler(object):
    def __init__(self, hdf5_index_path, batch_size):
        """Evaluation sampler. Samples all data in eval dataset without randomization.
        
           :param string hdf5_index_path: HDF5 index contaning the dataset
           :param int batch_size: Size of the returned batch-meta
        """

        self.batch_size = batch_size

        with h5py.File(hdf5_index_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.hdf5_paths = [hdf5_path.decode() for hdf5_path in hf['hdf5_path'][:]]
            self.indexes_in_hdf5 = hf['index_in_hdf5'][:]
            self.targets = hf['target'][:].astype(np.float32)
            
        self.audios_num = len(self.audio_names)

    def __iter__(self):
        """Generate batch meta for training. 
        
        :return batch_meta: List of dictionaries, each dictionary maps
                            'hdf5_path': str, 'index_in_hdf5': int
        :rtype: list
        """

        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(pointer, 
                min(pointer + batch_size, self.audios_num))

            batch_meta = []

            for index in batch_indexes:
                batch_meta.append({
                    'audio_name': self.audio_names[index], 
                    'hdf5_path': self.hdf5_paths[index], 
                    'index_in_hdf5': self.indexes_in_hdf5[index], 
                    'target': self.targets[index]})

            pointer += batch_size
            yield batch_meta


def collate_fn(list_data_dict):
    """Collate data.

       :param list list_data_dict: List of dictionaries with similar keys
                                   mapping to numpy.ndarray's

       :return np_dict_data: One dictionary with same keys as input dictionary,
                             each key maps to the same arrays as in input dicts,
                             but concatenated across a new (leftmost) dimension
       :rtype: dict
    """
    np_data_dict = {}
    
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict
