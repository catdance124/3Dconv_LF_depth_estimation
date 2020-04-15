import csv
import numpy as np

class two_input_generator():
    def __init__(self, class_list_path, val_mode=False):
        self.clear()
        self.val_mode = val_mode
        with open(class_list_path) as f:
            reader = csv.reader(f)
            self.classes = np.array([row for row in reader]).flatten()
        if val_mode:
            self.data_index = [f'../patch_data/{class_name}/full'
                                for class_name in self.classes]
        else:
            self.data_index = [f'../patch_data/{class_name}/{i:04}'
                                for i in range(1444)
                                for class_name in self.classes]
    
    def clear(self):
        self.images_h = []
        self.images_v = []
        self.images_disp = []
    
    def flow_from_directory(self, batch_size=64, seed=None):
        while True:
            np.random.shuffle(self.data_index)
            for target_index in self.data_index:
                self.images_h.append(np.load(f'{target_index}_h.npy'))
                self.images_v.append(np.load(f'{target_index}_v.npy'))
                self.images_disp.append(np.load(f'{target_index}_disp.npy'))
                if len(self.images_h) == batch_size:
                    images_batch_h = np.array(self.images_h, dtype=np.float32)
                    images_batch_v = np.array(self.images_v, dtype=np.float32)
                    images_batch_disp = np.array(self.images_disp, dtype=np.float32)
                    self.clear()
                    yield [images_batch_h, images_batch_v], images_batch_disp