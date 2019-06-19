import os
import time
import pickle
from tqdm import tqdm
import torchvision.transforms as transforms
from SiamNet import *
from VIDDataset import *
from torch.autograd import Variable
from Config import *
from DataAugmentation import *
from torch.utils.data import DataLoader
from CurriculumLearning.scoring_functions import scoring_function

# fix random seed
np.random.seed(1357)
torch.manual_seed(1234)

def generate_score(data_dir, pickle_file, pretrained_model, use_gpu=True):
    
    # Loading the image pairs
    start = time.time()
    fptr = open(pickle_file, 'rb')
    img_pairs = pickle.load(fptr) # (z, x, frame_distance)
    end = time.time()
    print('Number of image pairs: {}\tTime: {} mins'.format(len(img_pairs), round((end-start)/60., 2)))
    
    # initialize configuration
    config = Config()

    # Some parameters
    pos_pair_range = float(config.pos_pair_range)
    norm_factor = 2466
    alpha = 0.3

    # Data Augmentation
    center_crop_size = config.instance_size - config.stride
    random_crop_size = config.instance_size - 2 * config.stride

    z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor()
    ])
    x_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((center_crop_size, center_crop_size)),
        RandomCrop((random_crop_size, random_crop_size)),
        ToTensor()
    ])

    # Dataloader
    img_pairs = list(img_pairs)
    data_loader = DataLoader(img_pairs, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.train_num_workers)
    
    img_data = set()
    label_mask = None
    
    # Specifying SiamFC Network Architecture
    net = torch.load(pretrained_model)
    if use_gpu:
        net.cuda()

    net.eval()

    for j, data in enumerate(tqdm(data_loader)):
        exemplars, candidates, frame_distances = data
        
        # Normalised frame distance
        frame_distances = frame_distances.cpu().numpy()
        frame_distances = 1 - frame_distances / pos_pair_range

        # Input images
        exemplar_imgs, candidate_imgs = list(), list()
        for z, x in zip(exemplars, candidates):
            # Reading z and x
            z = cv2.imread(os.path.join(data_dir, z))
            z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
            z = z_transforms(z)
            exemplar_imgs.append(z.numpy())

            x = cv2.imread(os.path.join(data_dir, x))
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = x_transforms(x)
            candidate_imgs.append(x.numpy())
        exemplar_imgs = torch.tensor(exemplar_imgs)
        candidate_imgs = torch.tensor(candidate_imgs)
        if use_gpu:
            exemplar_imgs = exemplar_imgs.cuda()
            candidate_imgs = candidate_imgs.cuda()

        # Forward pass
        correlation = net.forward(Variable(exemplar_imgs), Variable(candidate_imgs)) / norm_factor
        normalised_correlation = torch.sigmoid(correlation).detach().cpu().numpy()
        
        # Creating the mask for the score map
        if label_mask is None:
            response_size = normalised_correlation.shape[2: 4] # Score Map
            half = response_size[0] // 2 + 1

            label_mask, label_weight = create_label(response_size, config, use_gpu) 
            label_mask = label_mask.cpu().numpy()[0].reshape(response_size)
            for row in range(label_mask.shape[0]):
                for col in range(label_mask.shape[1]):
                    if label_mask[row, col] == 0.:
                        continue
                    counter = abs(half-row-1) + abs(half-col-1)
                    if counter == 2.:
                        label_mask[row, col] = 1 / 8.
                    elif counter == 1.:
                        label_mask[row, col] = 1 / 4.
            label_mask = np.reshape(label_mask, normalised_correlation.shape[1:])

        # Get Correlation Score
        score = normalised_correlation * label_mask
        score = score.sum((1, 2, 3)) / 3.

        # Scoring function
        cumulative_score = [scoring_function(alpha, scr, fr) for scr, fr in zip(score, frame_distances)]
      
        for z, x, cuscr in zip(exemplars, candidates, cumulative_score):
            img_data.add((z, x, cuscr))
    
    # Storing the computed cumulative score value
    final_pickle = "PATH/TO/STORE/FINAL_RESULTS"
    with open(final_pickle, 'wb') as fptr:
        pickle.dump(img_data, fptr)


if __name__ == "__main__":
    data_dir = "PATH/TO/THE/DATA_DIRECTORY"
    pickle_file = "PATH/TO/THE/img_pairs_pickle_file"
    pretrained_model = "PATH/TO/THE/PRETRAINED_MODEL"

    # Calculate socre for each (z, x) pair
    generate_score(data_dir, pickle_file, pretrained_model)
