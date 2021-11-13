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
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = self.layers[4](x)
        x = self.layers[5](x)
        x = self.layers[6](x)
        return x



        # return self.layers(x)



def prepare_stylegan(args):

    if args['stylegan_ver'] == "1":
        if args['category'] == "car":
            resolution = 512
            max_layer = 8
        elif  args['category'] == "face":
            resolution = 1024
            max_layer = 8
        elif args['category'] == "bedroom":
            resolution = 256
            max_layer = 7
        elif args['category'] == "cat":
            resolution = 256
            max_layer = 7
        else:
            assert "Not implementated!"

        avg_latent = np.load(args['average_latent'])
        avg_latent = torch.from_numpy(avg_latent).type(torch.FloatTensor).cuda()

        g_all = nn.Sequential(OrderedDict([
            ('g_mapping', G_mapping()),
            ('truncation', Truncation(avg_latent,max_layer=max_layer, threshold=0.7)),
            ('g_synthesis', G_synthesis( resolution=resolution))
        ]))

        g_all.load_state_dict(torch.load(args['stylegan_checkpoint'], map_location='cpu'))
        g_all.eval()
        g_all = nn.DataParallel(g_all).cuda()

    else:
        assert "Not implementated error"

    res  = args['dim'][1]
    mode = args['upsample_mode']
    upsamplers = [nn.Upsample(scale_factor=res / 4, mode=mode),
                  nn.Upsample(scale_factor=res / 4, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode)
                  ]

    if resolution > 256:
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))

    if resolution > 512:

        upsamplers.append(Interpolate(res, 'bilinear'))
        upsamplers.append(Interpolate(res, 'bilinear'))

    return g_all, avg_latent, upsamplers


def test(args, checkpoint_path, num_sample, start_step=0, vis=True):

    from utils.data_util import sod_palette as palette
    if not vis:
        result_path = os.path.join(checkpoint_path, 'samples200' )
    else:
        result_path = os.path.join(checkpoint_path, 'vis_%d'%num_sample)
    if os.path.exists(result_path):
        pass
    else:
        # os.system('mkdir -p %s' % (result_path))
        os.makedirs(result_path)
        print('Experiment folder created at: %s' % (result_path))


    g_all, avg_latent, upsamplers = prepare_stylegan(args)

    classifier_list = []
    for MODEL_NUMBER in range(args['model_num']):
        print('MODEL_NUMBER', MODEL_NUMBER)

        classifier = pixel_classifier(numpy_class=args['number_class']
                                      , dim=args['dim'][-1])
        classifier =  nn.DataParallel(classifier).cuda()

        checkpoint = torch.load(os.path.join(checkpoint_path, 'model_' + str(MODEL_NUMBER) + '.pth'))

        classifier.load_state_dict(checkpoint['model_state_dict'])


        classifier.eval()
        classifier_list.append(classifier)

    with torch.no_grad():
        latent_cache = []
        image_cache = []
        seg_cache = []
        entropy_calculate = []
        results = []
        np.random.seed(start_step)
        count_step = start_step



        print( "num_sample: ", num_sample)

        for i in range(num_sample):

            print("Genearte", i, "Out of:", num_sample)

            curr_result = {}

            latent = np.random.randn(1, 512)

            curr_result['latent'] = latent


            latent = torch.from_numpy(latent).type(torch.FloatTensor).cuda()
            latent_cache.append(latent)

            img, affine_layers = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                                     return_upsampled_layers=True)

            if args['dim'][0] != args['dim'][1]:
                img = img[:, 64:448][0]
            else:
                img = img[0]

            image_cache.append(img)
            if args['dim'][0] != args['dim'][1]:
                affine_layers = affine_layers[:, :, 64:448]
            affine_layers = affine_layers[0]

            affine_layers = affine_layers.reshape(args['dim'][-1], -1).transpose(1, 0)

            all_seg = []
            all_entropy = []
            mean_seg = None

            seg_mode_ensemble = []
            for MODEL_NUMBER in range(args['model_num']):
                classifier = classifier_list[MODEL_NUMBER]

                img_seg = classifier(affine_layers)

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
            del (affine_layers)

            #******************* wuzhenyu  save the generated images and sod mask *************

            image_label_name = os.path.join(result_path, 'cat_' + str(count_step) + '.png')
            image_name = os.path.join(result_path, 'cat_' + str(count_step) + '.jpg')

            img = Image.fromarray(img)
            img_seg = Image.fromarray(img_seg_final.astype('uint8')*255)

            img.save(image_name)
            img_seg.save(image_label_name)
            count_step += 1
            #*********************************  end *******************************************






def main(args):


    # 1 load the stylegan model
    g_all, avg_latent, upsamplers = prepare_stylegan(args)

    # 2. load the latent
    latent_all = np.load(args['annotation_image_latent_path'])
    latent_all = torch.from_numpy(latent_all).cuda()

    # 3. load annotated mask
    mask_list = []
    latent_all = latent_all[:args['max_training']]
    num_data = len(latent_all)

    for i in range(len(latent_all)):

        if i >= args['max_training']:
            break
        name = 'image_mask%0d.npy' % i

        im_frame = np.load(os.path.join(args['annotation_mask_path'], name))
        mask = np.array(im_frame)
        mask = cv2.resize(np.squeeze(mask), dsize=(args['dim'][1], args['dim'][0]), interpolation=cv2.INTER_NEAREST)

        mask_list.append(mask)

    # delete small annotation error
    for i in range(len(mask_list)):  # clean up artifacts in the annotation, must do
        for target in range(1, 50):
            if (mask_list[i] == target).sum() < 30:
                mask_list[i][mask_list[i] == target] = 0

    all_mask = np.stack(mask_list)



    # *****************************  end  ************************





    # **************** revised by wuzhenyu ********************
    max_label =args['number_class'] -1
    # *****************************  end  ************************
    print(" *********************** max_label " + str(max_label) + " ***********************")


    print(" *********************** Current number data " + str(num_data) + " ***********************")


    batch_size = args['batch_size'] *64



    print(" *********************** Current dataloader length " +  str(len(latent_all)) + " ***********************")

    for MODEL_NUMBER in range(args['model_num']):

        gc.collect()

        classifier = pixel_classifier(numpy_class=(max_label + 1), dim=args['dim'][-1])

        # classifier.init_weights()

        classifier = nn.DataParallel(classifier).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()


        iteration = 0

        stop_sign = 0
        mean_acc_all =[]

        for epoch in range(5):
            mean_acc =[]

            # 4. generate featurs

            for i in range(len(latent_all)):

                gc.collect()

                latent_input = latent_all[i].float()

                img, feature_maps = latent_to_image(g_all, upsamplers, latent_input.unsqueeze(0), dim=args['dim'][1],
                                                    return_upsampled_layers=True,
                                                    use_style_latents=args['annotation_data_from_w'])
                if args['dim'][0] != args['dim'][1]:
                    # only for car
                    feature_maps = feature_maps[:, :, 64:448]
                mask = all_mask[i:i + 1]
                feature_maps = feature_maps.permute(0, 2, 3, 1)

                feature_maps = feature_maps.reshape(-1, args['dim'][2])

                mask = mask.reshape(-1)

                traindata = trainData(feature_maps, mask)
                train_loader = DataLoader(dataset=traindata, batch_size= batch_size, shuffle=True)



                for X_batch, y_batch in train_loader:



                    # **********  added by wuzhenyu ************
                    X_batch = X_batch.reshape(-1, X_batch.shape[-1])
                    y_batch = y_batch.reshape(-1, X_batch.shape[0]).squeeze()
                    # ********  end *********

                    X_batch, y_batch = torch.tensor(X_batch).cuda(), torch.tensor(y_batch).cuda()
                    X_batch = X_batch.type(torch.float32)
                    y_batch = y_batch.type(torch.long)

                    optimizer.zero_grad()
                    y_pred = classifier(X_batch)
                    loss = criterion(y_pred, y_batch)
                    acc = multi_acc(y_pred, y_batch)
                    mean_acc.append(acc)

                    loss.backward()
                    optimizer.step()

                    iteration += 1
                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                    if iteration % 1000 == 0:

                        gc.collect()




            mean_acc = sum(mean_acc)/ ((args['dim'][0] * args['dim'][1]) / batch_size )
            mean_acc_all.append(mean_acc)

            if stop_sign == 1:
                break

        if not os.path.exists(args['exp_dir']):
            os.makedirs(args['exp_dir'])

        # np.save(os.path.join(args['exp_dir'],'mean_acc_' + str(MODEL_NUMBER) + '.npy'), mean_acc_all.cpu())
        gc.collect()
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

    parser.add_argument('--exp', type=str, default="./experiments/stylegan/cat_sod.json")
    parser.add_argument('--exp_dir', type=str,  default="")
    parser.add_argument('--test', type=bool, default=False)

    parser.add_argument('--save_vis', type=bool, default=False)
    parser.add_argument('--start_step', type=int, default=0)

    parser.add_argument('--resume', type=str,  default="../saved_model/stylegan/cat/model-1/")
    parser.add_argument('--num_sample', type=int,  default=200)

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
        main(opts)

