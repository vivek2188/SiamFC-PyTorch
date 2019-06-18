import os
from Utils import *
from torch.utils.data.dataset import Dataset

class DataCL(Dataset):

    def __init__(self, data_dir, data_list, config, z_transforms, x_transforms):
        self.data_dir = data_dir
        self.data_list = data_list
        self.config = config

        self.z_transforms = z_transforms
        self.x_transforms = x_transforms

        self.num = config.num_pairs
        self.dataLen = len(data_list)

    def __getitem__(self, idx):
        '''
        reading (z, x) pair
        '''
        # read z and x
        instance = data_list[np.random.choice(dataLen)]
        z_path, x_path = instance[0], instance[1]

        img_z = cv2.imread(os.path.join(self.data_dir, z_path))
        img_z = cv2.cvtColor(img_z, cv2.COLOR_BGR2RGB)
        img_z = self.z_transforms(img_z)

        img_x = cv2.imread(os.path.join(self.data_dir, x_path))
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        img_x = self.x_transforms(img_x)
        
        return img_z, img_x

    def __len__(self):
        return self.num
