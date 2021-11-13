"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append('..')

import torch
import torch.nn as nn
torch.manual_seed(0)
import scipy.misc
import json
from collections import OrderedDict
import numpy as np
import os
# device_ids = [0]
from PIL import Image
import gc

import pickle
from models.stylegan1 import G_mapping,Truncation,G_synthesis
import copy
from numpy.random import choice
from torch.distributions import Categorical
import scipy.stats
from utils.utils import multi_acc, colorize_mask, get_label_stas, latent_to_image, oht_to_scalar, Interpolate
import torch.optim as optim
import argparse
import glob
from torch.utils.data import Dataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import cv2

from transformer_classifier import att_cls
from cnn_classifier import CNN_Classifier

from models.BigGAN import BigGAN
from models.BigGAN.utils import truncated_noise_sample
from models.BigGAN.config import BigGANConfig
from torchstat import stat

discard_id = [1, 29, 51, 55, 58, 59, 64, 103, 111, 130, 145, 340, 344, 345, 354, 355, 398,
                  399, 400, 401, 402, 406, 410, 413,
                  415, 416, 417, 419, 420, 421, 422, 423, 424, 428, 429, 430, 431, 432, 433, 435, 439, 444, 447, 450,
                  452, 453, 454, 456, 457,
                  466, 467, 471, 476, 480, 486, 489, 490, 491, 494, 496, 498, 500, 508, 509, 513, 514, 515, 516, 517,
                  518, 519, 520, 522, 523, 529,
                  532, 537, 541, 542, 543, 546, 552, 558, 560, 564, 566, 576, 580, 582, 585, 587, 588,
                  589, 593, 594, 595, 599, 602, 603, 611, 612, 613, 614, 617, 620, 624, 634, 641, 642, 643, 645, 646,
                  648, 650, 652, 655, 665,
                  667, 668, 669, 670, 671, 678, 683, 687, 690, 694, 695, 696, 697, 699, 702, 706, 716, 722, 723, 728,
                  730, 731, 736, 738, 739, 740,
                  743, 745, 747, 750, 752, 762, 764, 765, 768, 775, 776, 785, 786, 787, 788, 789, 792, 793, 794, 795,
                  796, 797, 798, 799, 800,
                  801, 813, 815, 816, 819, 822, 823, 825, 830, 836, 837, 841, 842, 843, 854, 856, 857, 858, 860, 862,
                  865, 868,
                  870, 875, 876, 877, 878, 879, 880, 884, 887, 889, 890, 912, 916, 917, 918, 921, 922, 929, 936, 937,
                  939, 942, 943, 945,
                  953, 954, 955, 956, 970, 972, 973, 974, 975, 976, 977, 978, 979, 981, 982, 984, 987, 998, 999]

for i in range(151, 295):
        # print(i)
        discard_id.append(i)



no_salient_id = [103, 123,  406, 398, 415, 421, 423, 454, 489, 500, 509, 519, 525, 580, 582, 599, 611,  624, 634, 669,
694, 706, 716, 723, 738, 743, 750, 762, 788, 789, 794, 815, 825, 854, 858, 860, 865, 875, 877, 884, 890, 904, 916, 917, 918,
921, 922, 936, 937, 939, 943, 945, 953, 970, 972, 973, 974, 975, 976, 977, 978, 979, 984 ]



def process_image(images):

    images = images.cpu().detach().numpy()

    drange = [-1, 1]
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)

    images = images.astype(int)
    images[images > 255] = 255
    images[images < 0] = 0

    images = np.transpose(images, (0, 2, 3, 1)).astype(np.uint8)

    return images



class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)






class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier, self).__init__()


        if numpy_class < 32:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),


                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),

                nn.Linear(32, numpy_class),
                # nn.Sigmoid()
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class),
                # nn.Sigmoid()
            )


        self.init_weights()


    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)



    def forward(self, x):

        return self.layers(x)



def test(args, checkpoint_path, num_sample, start_step=0, vis=True):

    results_path = os.path.join(checkpoint_path, 'samples12k')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    print('The generated samples will be saved in: {}'.format(results_path))

    # 1. load biggan model

    config = BigGANConfig(output_dim=256)
    model = BigGAN(config)

    model.load_state_dict(torch.load(args['weight_path']))
    # model = nn.DataParallel(model).cuda()
    model.cuda()

    # 2.  define the upsampler for setting: 256 x256
    res = args['dim'][1]
    mode = args['upsample_mode']

    upsamplers = [nn.Upsample(scale_factor=res / 4, mode=mode),
                  # nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  # nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  # nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),
                  # nn.Upsample(scale_factor=res / 64, mode=mode),
                  # nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  # nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode)
                  ]


    # 3 load the classifier

    classifier_list = []
    for MODEL_NUMBER in range(args['model_num']):
        print('MODEL_NUMBER', MODEL_NUMBER)

        classifier = pixel_classifier(numpy_class=args['number_class']
                                      , dim=args['dim'][-1])

        # cnn classifier
        # classifier = CNN_Classifier(7040, 2) # in_dim, output_dim  #*******************************************************************careful


        classifier = nn.DataParallel(classifier).cuda()

        checkpoint = torch.load(os.path.join(checkpoint_path, 'model_' + str(MODEL_NUMBER) + '.pth'))

        classifier.load_state_dict(checkpoint['model_state_dict'])

        classifier.eval()
        classifier_list.append(classifier)

    # 4
    with torch.no_grad():
        latent_cache = []
        image_cache = []
        seg_cache = []
        entropy_calculate = []
        results = []
        np.random.seed(start_step)
        count_step = 0

        print("num_sample: ", num_sample)

        for epoch in range(20):

            for i in range(0, 1000):

                if i not in discard_id:

                    print('Epoch:', epoch, "Genearte", i, "Out of:", num_sample)


                    # a. prepare the latent and cond_vec

                    latent = truncated_noise_sample(batch_size=1, truncation=0.4)

                    # control the generated catogories of image
                    label = torch.tensor([i, ])
                    label = torch.eye(1000, dtype=torch.float)[label]

                    latent = torch.from_numpy(latent).type(torch.FloatTensor).cuda()
                    label = torch.tensor(label, dtype=torch.float).cuda()

                    # b. generated the image and features

                    image, features = model(latent, label,  0.4)


                    image =process_image(image)
                    image = image[0]

                    #  upsample the feature maps
                    features_dim = 0
                    for k in range(len(features)):
                        features_dim += features[k].shape[1]

                    features_upsample = torch.FloatTensor(1, features_dim, args['dim'][0], args['dim'][1])

                    start_channel_index = 0
                    for j in range(len(features)):
                        len_channel = features[j].shape[1]
                        features_upsample[:, start_channel_index:start_channel_index + len_channel] = upsamplers[j](
                            features[j])
                        start_channel_index += len_channel

                    features_upsample = features_upsample.permute(0, 2, 3, 1)
                    features_upsample = features_upsample.reshape(-1, features_dim)


                    # feed the feature into classifier

                    all_seg = []
                    all_entropy = []
                    mean_seg = None

                    seg_mode_ensemble = []
                    for MODEL_NUMBER in range(args['model_num']):
                        classifier = classifier_list[MODEL_NUMBER]

                        img_seg = classifier(features_upsample)

                        # cnn classifier
                        # img_seg = classifier(features_upsample.unsqueeze(0).reshape(1, 256, 256, 7040).transpose(1,3)).squeeze().reshape(2, -1).transpose(0, 1)

                        # revised by wzy, attention classifier
                        #img_seg = classifier(features_upsample.unsqueeze(0), 256, 256).squeeze()

                        img_seg = img_seg.squeeze()

                        entropy = Categorical(logits=img_seg).entropy()
                        all_entropy.append(entropy)

                        all_seg.append(img_seg)
                        if mean_seg is None:
                            mean_seg = img_seg
                        else:
                            mean_seg += img_seg

                        img_seg_final = oht_to_scalar(img_seg)
                        img_seg_final = img_seg_final.reshape(args['dim'][0], args['dim'][1], 1)
                        img_seg_final = img_seg_final.cpu().detach().numpy()

                        seg_mode_ensemble.append(img_seg_final)

                    mean_seg = mean_seg / len(all_seg)

                    full_entropy = Categorical(logits=mean_seg).entropy()

                    js = full_entropy - torch.mean(torch.stack(all_entropy), 0)

                    top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()
                    entropy_calculate.append(top_k)

                    img_seg_final = np.concatenate(seg_mode_ensemble, axis=-1)
                    img_seg_final = scipy.stats.mode(img_seg_final, 2)[0].reshape(args['dim'][0], args['dim'][1])
                    del (features_upsample)

                    # ******************* wzy  save the generated images and sod mask *************

                    img_path = os.path.join(results_path, 'image')
                    mask_path = os.path.join(results_path, 'mask')
                    if not os.path.exists(img_path):
                        os.makedirs(img_path)
                    if not os.path.exists(mask_path):
                        os.makedirs(mask_path)

                    mask_path = os.path.join(mask_path, str(count_step) + '.png')
                    img_path = os.path.join(img_path, str(count_step) + '.jpg')

                    img = Image.fromarray(image)
                    img_seg = Image.fromarray(img_seg_final.astype('uint8') * 255)

                    img.save(img_path)
                    img_seg.save(mask_path)
                    count_step += 1
                    # *********************************  end *******************************************





def train(args):


    # **************** revised by wzy ********************

    # *********  prepar for load data  *************

    # 1. load model

    config = BigGANConfig(output_dim=256)
    model = BigGAN(config)

    model.load_state_dict(torch.load(args['weight_path']))
    # model = nn.DataParallel(model).cuda()
    model.cuda()

    # 2.  define the upsampler for setting: 256 x256
    res = args['dim'][1]
    mode = args['upsample_mode']

    upsamplers = [
                  nn.Upsample(scale_factor=res / 4, mode=mode),
                  # nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  # nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  # nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),

                  # nn.Upsample(scale_factor=res / 64, mode=mode),
                  # nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  # nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode)
                  ]

    # 3. load the latent and conditional label
    latent_all = np.load(args['annotation_image_latent_path'])
    latent_all = torch.from_numpy(latent_all).cuda().float()

    conditional_label_all = np.load(args['conditional_vec_path'])
    conditional_label_all = torch.from_numpy(conditional_label_all).cuda().float()

    # 4. load annotated mask


    mask_list = []
    for i in range(0, 1000): ###########################*****************************************************************************



            name = 'mask_%0d.npy' % i
            im_frame = np.load(os.path.join(args['annotation_mask_path'], name))
            mask = np.array(im_frame)
            # mask = cv2.resize(np.squeeze(mask), dsize=(args['dim'][1], args['dim'][0]), interpolation=cv2.INTER_NEAREST)
            mask_list.append(mask)

    # delete small annotation error
    for i in range(len(mask_list)):  # clean up artifacts in the annotation, must do
        for target in range(1, 50):
            if (mask_list[i] == target).sum() < 30:
                mask_list[i][mask_list[i] == target] = 0

    all_mask = np.stack(mask_list)


    # *************** end  ***************************************************







    # **************** revised by wzy ********************
    max_label =args['number_class'] -1
    # *****************************  end  ************************
    print(" *********************** max_label " + str(max_label) + " ***********************")


    print(" *********************** Current number data " + str(len(latent_all)) + " ***********************")


    batch_size = args['batch_size'] * 64



    # print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")

    for MODEL_NUMBER in range(args['model_num']):

        gc.collect()

        classifier = pixel_classifier(numpy_class=(max_label + 1), dim=args['dim'][-1])

        #, transformer classifier
        # classifier = att_cls(7040, 8) # dim, head_num

        # cnn classifier
        # classifier = CNN_Classifier(7040, 2) # in_dim, output_dim


        # classifier.init_weights()

        classifier = nn.DataParallel(classifier).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()





        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        mean_acc_all =[]

        for epoch in range(5):
            mean_acc =[]

            # ************** load data  *******************


            # 5. saved the mask and the generated features

            for i in range(0, 1000):
                    gc.collect()


                    # 1. generate the feature maps
                    latent_input = latent_all[i]
                    label_input = conditional_label_all[i]
                    with torch.no_grad():
                        img, features = model(latent_input, label_input, truncation=0.4)


                    mask = all_mask[i:i + 1]
                    mask = mask.reshape(-1)

                    # 2 upsample the feature maps
                    features_dim = 0
                    for k in range(len(features)):
                        features_dim += features[k].shape[1]

                    features_upsample = torch.FloatTensor(1, features_dim, args['dim'][0], args['dim'][1])

                    start_channel_index = 0
                    for j in range(len(features)):
                        len_channel = features[j].shape[1]
                        features_upsample[:, start_channel_index:start_channel_index + len_channel] = upsamplers[j](
                            features[j])
                        start_channel_index += len_channel

                    features_upsample = features_upsample.permute(0, 2, 3, 1)
                    features_upsample = features_upsample.reshape(-1, args['dim'][-1])

                    traindata = trainData(features_upsample, mask)
                    train_loader = DataLoader(dataset=traindata, batch_size= batch_size, shuffle=True)




                    for X_batch, y_batch in train_loader:
                        # **********  added by wzy ************
                        X_batch = X_batch.reshape(-1, X_batch.shape[-1])
                        y_batch = y_batch.reshape(-1, X_batch.shape[0]).squeeze()
                        # ********  end *********

                        X_batch, y_batch = torch.tensor(X_batch).cuda(), torch.tensor(y_batch).cuda()
                        X_batch = X_batch.type(torch.float32)
                        y_batch = y_batch.type(torch.long)

                        optimizer.zero_grad()

                        y_pred = classifier(X_batch)

                        # revised by wzy, transformer classifier
                        # y_pred = classifier(X_batch.unsqueeze(0), 64, 64).squeeze()  #

                        # cnn classifier
                        # y_pred = classifier(X_batch.unsqueeze(0).reshape(1, 64 , 64, 7040).transpose(1, 3)).squeeze().reshape(2, -1).transpose(0, 1)

                        loss = criterion(y_pred, y_batch)
                        acc = multi_acc(y_pred, y_batch)
                        mean_acc.append(acc.cpu().numpy())

                        loss.backward()
                        optimizer.step()

                        iteration += 1
                        print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                        if iteration % 1000 == 0:

                            gc.collect()





            mean_acc = sum(mean_acc)/ ((args['dim'][0] * args['dim'][1]) / batch_size )
            mean_acc_all.append(mean_acc)

            # save model at very epoch
            if not os.path.exists(args['exp_dir']):
                os.makedirs(args['exp_dir'])

            model_path = os.path.join(args['exp_dir'],
                                      'model_epoch' + str(epoch) + '.pth')

            torch.save({'model_state_dict': classifier.state_dict()},
                       model_path)

        # save the mean acc in each epoch
        if not os.path.exists(args['exp_dir']):
            os.makedirs(args['exp_dir'])
        np.save(os.path.join(args['exp_dir'],'mean_acc_' + str(MODEL_NUMBER) + '.npy'), mean_acc_all)
        gc.collect()

        # save the final model
        model_path = os.path.join(args['exp_dir'],
                                  'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)
        gc.collect()


        gc.collect()
        torch.cuda.empty_cache()    # clear cache memory on GPU


if __name__ == '__main__':




    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, default='./experiments/biggan/all.json')
    parser.add_argument('--exp_dir', type=str,  default="../saved_model/biggan/model-1")
    parser.add_argument('--test', type=bool, default=False)

    parser.add_argument('--save_vis', type=bool, default=False)
    parser.add_argument('--start_step', type=int, default=0)

    parser.add_argument('--resume', type=str,  default="../saved_model/biggan/")  # the path
    parser.add_argument('--num_sample', type=int,  default=1000)

    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)

    if args.exp_dir != "":
        opts['exp_dir'] = args.exp_dir


    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))



    if args.test:
        test(opts, args.resume, args.num_sample, vis=args.save_vis, start_step=args.start_step)
    else:
        train(opts)

