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

    # initialize training configuration
    config = Config()

    # load data (see details in VIDDataset.py)
    train_dataset = VIDDataset(train_imdb, data_dir, config, None, None)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.train_num_workers, drop_last=True)

    img_pairs = set() # It stores the (exemplar, candidate) frame pairs

    for i in range(config.num_epoch):
        for j, data in enumerate(tqdm(train_loader)):
            # fetch data, i.e., B x C x W x H (batchsize x channel x width x height)
            exemplar_imgs, instance_imgs = data
            for z, x in zip(exemplar_imgs, instance_imgs):
                img_pairs.add((z, x))
        print('Size after Epoch[{}]: {}'.format(i+1, len(img_pairs))) # Checking the progress
    
    # Storing img_pairs to a pickle file
    with open('img_pairs.pickle', 'wb') as fptr:
        pickle.dump(img_pairs, fptr)

if __name__ == "__main__":
    data_dir = "PATH/TO/THE/DATA_DIRECTORY"
    train_imdb = "PATH/TO/THE/TRAIN_JSON_FILE"

    # training SiamFC network, using GPU by default
    prepare_data(data_dir, train_imdb)
