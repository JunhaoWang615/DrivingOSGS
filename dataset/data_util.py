import os

import numpy as np
import PIL.Image as pil

import torch.nn.functional as F
import torchvision.transforms as transforms

_DEL_KEYS= ['rgb', 'rgb_context', 'rgb_original', 'rgb_context_original', 'intrinsics', 'contexts', 'splitname', 'ego_pose'] 


def transform_mask_sample(sample, data_transform):
    """
    This function transforms masks to match input rgb images.
    """
    image_shape = data_transform.keywords['image_shape']
    # resize transform
    resize_transform = transforms.Resize(image_shape, interpolation=pil.ANTIALIAS)
    sample['mask'] = resize_transform(sample['mask'])
    # totensor transform
    tensor_transform = transforms.ToTensor()    
    sample['mask'] = tensor_transform(sample['mask'])
    return sample


def img_loader(path):
    """
    This function loads rgb image.
    """
    with open(path, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('RGB')


def mask_loader_scene(path, mask_idx, cam):
    """
    This function loads mask that correspondes to the scene and camera.
    """
    fname = os.path.join(path, str(mask_idx), '{}_mask.png'.format(cam.upper()))    
    with open(fname, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('L')


def align_dataset(sample, scales, contexts):
    """
    This function reorganize samples to match our trainer configuration.
    """
    K = sample['intrinsics']
    aug_images = sample['rgb']
    aug_contexts = sample['rgb_context']
    org_images = sample['rgb_original']
    org_contexts= sample['rgb_context_original']    
    ego_poses = sample['ego_pose']

    n_cam, _, w, h = aug_images.shape

    # initialize intrinsics
    resized_K = np.expand_dims(np.eye(4), 0).repeat(n_cam, axis=0)
    resized_K[:, :3, :3] = K

    # augment images and intrinsics in accordance with scales 
    for scale in scales:
        scale_int = int(scale)
        
        scaled_K = resized_K.copy()
        scaled_K[:,:2,:] /= (2**scale_int)  # 使用转换后的int计算
        
        sample[('K', scale_int)] = scaled_K.copy()  # 可选：统一key为int类型
        sample[('inv_K', scale_int)]= np.linalg.pinv(scaled_K).copy()

        # 核心修复2：将尺寸计算结果转为Python原生int，确保size参数类型正确
        target_size = (int(w // (2**scale_int)), int(h // (2**scale_int)))
        
        resized_org = F.interpolate(org_images, 
                                          size=target_size,  # 使用转换后的尺寸元组
                                          mode = 'bilinear',
                                          align_corners=False)
        resized_aug = F.interpolate(aug_images, 
                                          size=target_size,  # 统一使用转换后的尺寸
                                          mode = 'bilinear',
                                          align_corners=False)            
            
        sample[('color', 0, scale_int)] = resized_org
        sample[('color_aug', 0, scale_int)] = resized_aug

    # for context data
    for idx, frame in enumerate(contexts):
        sample[('color', frame, 0)] = org_contexts[idx]        
        sample[('color_aug',frame, 0)] = aug_contexts[idx]
        sample[('cam_T_cam', 0, frame)] = ego_poses[idx]
        
    # delete unused arrays
    for key in list(sample.keys()):
        if key in _DEL_KEYS:
            del sample[key]
    return sample