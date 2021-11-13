import os
import sys
import torch
from torch.hub import urlparse, get_dir, download_url_to_file
import pickle
import torchvision
import torch.nn as nn
from networks import ToRGBLayer
#






MODELS = {
    'ffhq': ('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl', None),
    'afhqwild': ('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl', None),
}


def tensor_to_image(t: torch.Tensor):
    t = t.cpu().detach()
    t = torch.clamp(t * 0.5 + 0.5, min=0, max=1)
    return torchvision.transforms.ToPILImage()(t)

def download_url(url, download_dir=None, filename=None):
    parts = urlparse(url)
    if download_dir is None:
        hub_dir = get_dir()
        download_dir = os.path.join(hub_dir, 'checkpoints')
    if filename is None:
        filename = os.path.basename(parts.path)
    cached_file = os.path.join(download_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file)
    return cached_file


class GeneratorWrapper(torch.nn.Module):
    """ A wrapper to put the GAN in a standard format. This wrapper takes
        w as input, rather than (z, c) """

    def __init__(self, G, num_classes=None):
        super().__init__()
        self.G = G  # NOTE! This takes in w, rather than z
        self.dim_z = G.synthesis.w_dim
        self.conditional = (num_classes is not None)
        self.num_classes = num_classes

        self.num_ws = G.synthesis.num_ws
        self.truncation_psi = 0.5
        self.truncation_cutoff = 8

    def forward(self, z):
        r"""The input `z` is expected to be `w`, not `z`, in the notation
            of the original StyleGAN 2 paper"""
        if len(z.shape) == 2:  # expand to 18 layers
            z = z.unsqueeze(1).repeat(1, self.num_ws, 1)

        # x0 = self.G.synthesis.b4(z)
        # x1 = self.G.synthesis.b8(x0)
        # x2 = self.G.synthesis.b16(x1)
        # x3 = self.G.synthesis.b32(x2)
        # x4 = self.G.synthesis.b64(x3)
        # x5 = self.G.synthesis.b128(x4)
        # x6 = self.G.synthesis.b256(x5)
        # x7 = self.G.synthesis.b512(x6)

        x7 = self.G.synthesis(z)

        return x7

    def sample_latent(self, batch_size, device='cpu'):
        z = torch.randn([batch_size, self.dim_z], device=device)
        c = None if self.conditional else None  # not implemented for conditional models
        w = self.G.mapping(z, c, truncation_psi=self.truncation_psi, truncation_cutoff=self.truncation_cutoff)
        return w


def add_utils_to_path():
    import sys
    from pathlib import Path
    util_path = str(Path(__file__).parent)
    if util_path not in sys.path:
        sys.path.append(util_path)
        print(f'Added {util_path} to path')


def make_stylegan2(model_name='ffhq') -> torch.nn.Module:
    """G takes as input an image in NCHW format with dtype float32, normalized 
    to the range [-1, +1]. Some models also take a conditioning class label, 
    which is passed as img = G(z, c)"""
    add_utils_to_path()  # we need dnnlib and torch_utils in the path
    if model_name in MODELS:
        url, num_classes = MODELS[model_name]
        cached_file = download_url(url)
    else:
        cached_file = model_name
        num_classes = model_name.split('/')[-1][:-4]
    assert cached_file.endswith('.pkl')
    with open(cached_file, 'rb') as f:
        G = pickle.load(f)['G_ema']
    G = GeneratorWrapper(G, num_classes=num_classes)

    return G.eval()




if __name__ == '__main__':

    import torch.nn.functional as F
    import cv2

    # Testing, afhqcat, afhqdog, afhqwild, brecahad, cifar10, ffhq, metfaces
    G = make_stylegan2(model_name='H:/mywork/pre-trained-model/stylegan2/metfaces.pkl').cuda()

    # G = make_stylegan2().cuda()

    print('Created G')
    print(f'Params: {sum(p.numel() for p in G.parameters()):_}')

    for i in range(0, 100):

        z = G.sample_latent(batch_size=1, device='cuda')  # -> torch.Size([1, 128])
        x = G(z=z)  # -> torch.Size([1, 3, 256, 256])
        x = torch.squeeze(x)

        img = tensor_to_image(x)

        img.save('H:/mywork/work6/datasetGAN_release-master/data/stylegan2/metfaces/img_{}.jpg'.format(i))
        print('img_{}.jpg'.format(i))



