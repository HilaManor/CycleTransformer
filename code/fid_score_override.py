"""
This code is almost a carbon copy of the original library pytorch-fid by mseitzer,
available here https://github.com/mseitzer/pytorch-fid
There was no way to pass additional transformation to the images for the 'get_activations' function, meaning all
images in the folders had to be of the same size - which they are not.
The code is given as is with the original documentation as to not change the original author's documentation wants.
The only change is the addition of the Resize((224,224)) transformation in 'get_activations'. 
"""

from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import torchvision.transforms as TF
import os
import torch
from tqdm import tqdm
import numpy as np
import pathlib
from torch.nn.functional import adaptive_avg_pool2d


def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = fid_score.ImagePathDataset(files, transforms=TF.Compose([TF.Resize((224, 224)), 
                                                             TF.ToTensor()]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in fid_score.IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)

    return m, s


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid_given_paths(real_ims_path, fake_ims_path, device, batch_size=40):
    num_avail_cpus = len(os.sched_getaffinity(0))
    num_workers = min(num_avail_cpus, 8)
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]  # Final average pooling features (pool3 features)
    model = InceptionV3([block_idx]).to(device)
    
    m1, s1 = compute_statistics_of_path(real_ims_path, model, batch_size,
                                        dims, device, num_workers)
    m2, s2 = compute_statistics_of_path(fake_ims_path, model, batch_size,
                                        dims, device, num_workers)
    fid_value = fid_score.calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value
    
    
