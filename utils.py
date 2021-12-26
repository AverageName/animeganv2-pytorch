import torch
import torch.nn as nn

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(b * c * d)


def style_loss(style, fake):
    return nn.L1Loss(gram_matrix(style), gram_matrix(fake))


def percep_style_loss(vgg, real, anime_true, generated):
    real_feature_map = vgg(real)
    generated_feature_map = vgg(generated)
    anime_true_feature_map = vgg(anime_true)

    percep_loss = nn.L1Loss(real_feature_map, generated_feature_map)
    s_loss = style_loss(generated_feature_map, anime_true_feature_map)

    return percep_loss, s_loss


def rgb2yuv(image):
    image = (image + 1.0)/2.0

    y = image[:, 0, :, :] * 0.299 + image[:, 1, :, :] * 0.587 + image[:, 2, :, :] * 0.114
    u = -image[:, 0, :, :] * 0.14714119  - image[:, 1, :, :] * 0.28886916 + image[:, 2, :, :] * 0.43601035
    v = image[:, 0, :, :] * 0.61497538 - image[:, 1, :, :] * 0.51496512 - image[:, 2, :, :] * 0.10001026
    return torch.stack([y, u, v], axis=-1)


def color_loss(real, generated):
    real_yuv = rgb2yuv(real)
    generated_yuv = rgb2yuv(generated)

    return  nn.L1Loss(real_yuv[:, :, :, 0], generated_yuv[:, :, :, 0]) + \
            nn.HuberLoss(real_yuv[:, :, :, 1], generated_yuv[:,:,:,1]) + \
            nn.HuberLoss(real_yuv[:,:,:,2], generated_yuv[:,:,:,2])


def total_variation_loss(img, weight):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)