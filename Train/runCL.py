import os
import random
import pickle
from tqdm import tqdm
from math import log, ceil
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from Utils import *
from Config import *
from SiamNet import *
from VIDDataset import *
from DataCL import DataCL
from DataAugmentation import *
from CurriculumLearning.pacing_functions import fixed_exponential_pacing, varied_exponential_pacing

# fix random seed
np.random.seed(1357)
torch.manual_seed(1234)

def trainCL(data_dir, data_list, val_imdb, gv, model_save_path="./model/Curriculum/CL/", use_gpu=True):

    # Initialising Configuration
    config = Config()

    # Data Augmentation
    center_crop_size = config.instance_size - config.stride
    random_crop_size = config.instance_size - 2 * config.stride

    train_z_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        RandomStretch(),
        CenterCrop((center_crop_size, center_crop_size)),
        RandomCrop((random_crop_size, random_crop_size)),
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        CenterCrop((config.examplar_size, config.examplar_size)),
        ToTensor(),
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    # load data
    train_dataset = None
    val_dataset = VIDDataset(val_imdb, data_dir, config, valid_z_transforms, valid_x_transforms, "Validation")

    # create dataloader
    train_loader = None
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.val_num_workers, drop_last=True)

    # create SiamFC network architecture (see details in SiamNet.py)
    net = SiamNet()
    # move network to GPU if using GPU
    if use_gpu:
        net.cuda()

    # define training strategy;
    # the learning rate of adjust layer (i.e., a conv layer)
    # is set to 0 as in the original paper
    optimizer = torch.optim.SGD([
        {'params': net.feat_extraction.parameters()},
        {'params': net.adjust.bias},
        {'params': net.adjust.weight, 'lr': 0},
    ], config.lr, config.momentum, config.weight_decay)

    # adjusting learning in each epoch
    scheduler = StepLR(optimizer, config.step_size, config.gamma)

    # used to control generating label for training;
    # once generated, they are fixed since the labels for each
    # pair of images (examplar z andCopyright search region x) are the same
    train_response_flag = False
    valid_response_flag = False

    # ------------------------ training & validation process ------------------------
    for i in range(config.num_epoch):

        # Creating the dataloader using the value provided by pacing function
        if i==0 or (i<len(gv) and gv[i]!=gv[i-1]):
            dataLen = int(gv[i] * len(data_list))
            train_dataset = DataCL(data_dir, data_list[: dataLen], config, train_z_transforms, train_x_transforms)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                   shuffle=True, num_workers=config.train_num_workers, drop_last=True)

        # adjusting learning rate
        scheduler.step()

        # ------------------------------ training ------------------------------
        # indicating training (very important for batch normalization)
        net.train()

        # used to collect loss
        train_loss = []

        for j, data in enumerate(tqdm(train_loader)):

            # fetch data, i.e., B x C x W x H (batchsize x channel x wdith x heigh)
            exemplar_imgs, instance_imgs = data

            # forward pass
            if use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()
            output = net.forward(Variable(exemplar_imgs), Variable(instance_imgs))

            # create label for training (only do it one time)
            if not train_response_flag:
                # change control flag
                train_response_flag = True
                # get shape of output (i.e., response map)
                response_size = output.shape[2:4]
                # generate label and weight
                train_eltwise_label, train_instance_weight = create_label(response_size, config, use_gpu)

            # clear the gradient
            optimizer.zero_grad()/export/livia/home/vision/vtiwari/Project/SiamFC-PyTorch/Tracking/tracking_result/CurriculumLearning/CL/4/Model-14

            # loss
            loss = net.weight_loss(output, train_eltwise_label, train_instance_weight)

            # backward
            loss.backward()

            # update parameter
            optimizer.step()

            # collect training loss
            train_loss.append(loss.data.item())

        # ------------------------------ saving model ------------------------------
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(net, model_save_path + "SiamFC_" + str(i + 1) + "_model.pth")

        # ------------------------------ validation ------------------------------
        # indicate validation
        net.eval()

        # used to collect validation loss
        val_loss = []

        for j, data in enumerate(tqdm(val_loader)):

            exemplar_imgs, instance_imgs = data

            # forward pass
            if use_gpu:
                exemplar_imgs = exemplar_imgs.cuda()
                instance_imgs = instance_imgs.cuda()
            output = net.forward(Variable(exemplar_imgs), Variable(instance_imgs))

            # create label for validation (only do it one time)
            if not valid_response_flag:
                valid_response_flag = True
                response_size = output.shape[2:4]
                valid_eltwise_label, valid_instance_weight = create_label(response_size, config, use_gpu)

            # loss
            loss = net.weight_loss(output, valid_eltwise_label, valid_instance_weight)

            # collect validation loss
            val_loss.append(loss.data.item())

        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)
        print ("Epoch %d   training loss: %f, validation loss: %f" % (i+1, np.mean(train_loss), np.mean(val_loss)))


if __name__ == "__main__":
    data_dir = "PATH/TO/THE/DATA_DIRECTORY"
    val_imdb = "PATH/TO/THE/VALIDATION_JSON_FILE"

    # Loading the (z, x) pairs
    with open('./img_pairs.pickle', 'rb') as fptr:
        data_list = pickle.load(fptr)
    data_list = list(data_list)
    print('Loaded the img pairs')

    # Pacing Function
    starting_percent = 0.2
    inc = 1.1
    min_step_len = 3
    max_step_len = 10
    steps = int(ceil(-log(starting_percent, inc)))
    step_length = [random.randint(min_step_len, max_step_len) for i in range(steps)]
    gv = varied_exponential_pacing(starting_percent=starting_percent, inc=inc, steps=steps, step_length=step_length)

    # training the network
    trainCL(data_dir, data_list, val_imdb, gv)
