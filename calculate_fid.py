import numpy as np
import torch
from scipy import linalg
from torchvision.models import inception_v3
from utils import permute_labels
import torchvision.transforms as T
from torch import nn
from torch.nn import functional as F



transform = T.Compose([
            lambda x: x * 0.5 + 0.5,
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  
            lambda x: F.interpolate(x, size=(299, 299), mode='bilinear', 
                                    align_corners=False, recompute_scale_factor=False)
        ])


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(dataloader, model, classifier, attr):
    classifier.eval()
    model.eval()
    device = model.device
    
    real_act = []
    gen_act = []
    
    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)
            label_src = label[:, attr]
            label_trg = permute_labels(label_src)
           
            gen = model.generate(image, label_trg)
            image, gen = transform(image), transform(gen)

            real_act.append(classifier(image))
            gen_act.append(classifier(gen))
    real_act = torch.cat(real_act).cpu()
    gen_act = torch.cat(gen_act).cpu()
    
    mu1, sigma1 = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu2, sigma2 = gen_act.mean(axis=0), np.cov(gen_act, rowvar=False)  
    
    return mu1, sigma1, mu2, sigma2

@torch.no_grad()
def calculate_fid(dataloader, model, attr):
    device = model.device
    classifier = inception_v3(pretrained=True, progress=True)
    classifier.fc = nn.Identity()
    classifier.to(device)
    
    m1, s1, m2, s2 = calculate_activation_statistics(dataloader, model, classifier, attr)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value.item()
