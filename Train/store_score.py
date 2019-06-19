import os
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

def generate_score(data_dir, train_json, store_folder, pretrained_model_path, use_gpu=True):
    
    # initialize training configuration
    config = Config()

    # Some parameters
    pos_pair_range = float(config.pos_pair_range)
    norm_factor = 2466
    alpha = list(np.arange(0.0, 1.1, 0.1))

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

    # Dataset
    train_dataset = VIDDataset(train_json, data_dir, config, z_transforms, x_transforms)

    # Dataloader
    data_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.train_num_workers)
    
    img_data = [set() for i in range(len(alpha))]
    label_mask = None
    
    # Specifying SiamFC Network Architecture
    net = torch.load(pretrained_model_path)
    if use_gpu:
        net.cuda()

    net.eval()

    for j, data in enumerate(tqdm(data_loader)):
        zs, xs, z_paths, x_paths, fr_dist = data
        fr_dist = fr_dist.cpu().numpy()
        fr_dist = 1 - fr_dist / pos_pair_range

        if use_gpu:
            zs = zs.cuda()
            xs = xs.cuda()

        # Forward pass
        correlation = net.forward(Variable(zs), Variable(xs)) / norm_factor
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
        for ai in alpha:
            cumulative_score = [scoring_function(round(ai, 1), scr, fr) for scr, fr in zip(score, fr_dist)]
            idx = int(ai*10)
            for z, x, cuscr in zip(z_paths, x_paths, cumulative_score):
                img_data[idx].add((z, x, cuscr))

    # Sorting the score in descending order as higher the score corresponds to the more easy sample
    img_data = [sorted(list(imgDatum), key=lambda f: f[2], reverse=True) for imgDatum in img_data]

    #Storing to a pickle file
    for idx, imgDatum in enumerate(img_data):
        filename = str(idx) + '.pickle'
        filepath = store_folder + filename
        with open(filepath, 'wb') as pickle_file:
            pickle.dump(imgDatum, pickle_file)

if __name__ == "__main__":
    data_dir = "PATH/TO/THE/DATA_DIRECTORY"
    train_json = "PATH/TO/THE/TRAIN_JSON_FILE"
    store_folder = 'PATH/TO/STORE/THE/FINAL_RESULTS'
    pretrained_model = "PATH/TO/THE/PRETRAINED_MODEL"

    # Get score for each image
    scores = generate_score(data_dir, train_json, store_folder, pretrained_model)
