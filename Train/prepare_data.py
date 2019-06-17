from VIDDataset import *
from Config import *
from Utils import *
import os
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

# fix random seed
np.random.seed(1357)
torch.manual_seed(1234)

def prepare_data(data_dir, train_imdb, use_gpu=True):

    # initialize configuration
    config = Config()

    # load data (see details in VIDDataset.py)
    train_dataset = VIDDataset(train_imdb, data_dir, config, None, None)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, drop_last=True)

    # It stores the (exemplar, candidate, frame_distance) pairs
    img_pairs = set() 

    for i in range(config.num_epoch):
        for j, data in enumerate(tqdm(train_loader)):
            # fetch data, i.e., B x C x W x H (batchsize x channel x width x height)
            exemplar_imgs, instance_imgs, frame_distances = data
            for z, x, frame_distance in zip(exemplar_imgs, instance_imgs, frame_distances):
                img_pairs.add((z, x, frame_distance))
        print('Size after Epoch[{}]: {}'.format(i+1, len(img_pairs))) # Checking the progress
    
    # Storing img_pairs to a pickle file
    with open('img_pairs.pickle', 'wb') as fptr:
        pickle.dump(img_pairs, fptr)

if __name__ == "__main__":
    data_dir = "PATH/TO/THE/DATA_DIRECTORY"
    train_imdb = "PATH/TO/THE/TRAIN_JSON_FILE"

    # Preparing image pairs (z, x)
    prepare_data(data_dir, train_imdb)
