import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
import pickle
import cv2
import os
import sys

sys.path.append('..')
from utils.generic_loader import *
from utils.object_information import *

class mmWaveDataset(Dataset):
    def __init__(
        self,
        phase='train',
        augment = True,
        plot=False,
        los_only=None,
        **augment_parameters,
    ):
        super(mmWaveDataset, self).__init__()
        self.augment = augment
        self.add_noise = augment_parameters.get("add_noise", False)
        self.conditional_noise = augment_parameters.get("conditional_noise", False)
        self.dilate_mask = augment_parameters.get("dilate_mask", False)
        self.use_new_mask = augment_parameters.get("use_new_mask", False)
        self.use_multi_mask = augment_parameters.get("use_multi_mask", False)
        self.relative_norm = augment_parameters.get("relative_norm", False)
        self.apply_mask = augment_parameters.get("apply_mask", False)
        self.two_channel = augment_parameters.get("two_channel", False)
        self.num_channels = augment_parameters.get("num_channels", False)
        self.mask_only = augment_parameters.get("mask_only", False)
        self.use_diff = augment_parameters.get("use_diff", False)
        self.use_spec = augment_parameters.get("use_spec", False)
        self.use_edge = augment_parameters.get("use_edge", False)
        self.num_channels = augment_parameters.get("num_channels", False)
        self.gaussian_blur = augment_parameters.get("gaussian_blur", False)
        self.use_new_data = augment_parameters.get("use_new_data", False)
        self.fixed_exp_num = augment_parameters.get("fixed_exp_num", False)
        self.clip_vals = augment_parameters.get("clip_vals", False)


        # Classes:
        # 0: can
        # 1: Handle 
        # 2: Box
        # 3: Sphere
        # 4: Cup
        # 5: Clamp

        filename = 'test_split_1729816641.pkl' if self.use_new_data else 'test_split_1709602125.pkl'
        with open(filename, 'rb') as f:
            split = pickle.load(f)
        classes = split['classes']
        class_num = {
            'can': 0,
            'handle': 1,
            'box': 2,
            'sphere': 3,
            'cup': -1, 
            'clamp': -1,
        }
        obj_inds = split[phase]

        self.all_data = []
        self.all_labels = []
        self.all_is_sim = []
        self.all_names = []
        self.all_masks = []
        self.all_exp_nums = []
        
        all_sim_exts = []
        self.max_sim_weights = []
        if self.use_diff:
            all_sim_exts.append('_diffuse_match_10000_cluster')
            self.max_sim_weights.append(1)
        if self.use_spec:
            all_sim_exts.append('_specular_5_10000_cluster')
            self.max_sim_weights.append(1)
        if self.use_edge:
            all_sim_exts.append('_edges_20_10000_cluster')
            self.max_sim_weights.append(0.5)

        obj_info = ObjectInformation()
        exp_num = '1'
        failed = False
        for obj_key in obj_inds:
            if self.use_new_data:
                obj_id, name, is_sim, is_los, exp_num = obj_key
                if is_sim: exp_num = '1'
            else:
                obj_id, name, is_sim, is_los = obj_key
            if not is_sim and self.fixed_exp_num is not None and exp_num != self.fixed_exp_num: continue
            if los_only == 'los' and not is_los: continue
            if los_only == 'nlos' and is_los: continue
            print(f'Loading object {obj_id} {name} Sim: {is_sim} LOS: {is_los}. Exp: {exp_num}')

            avail_attr = ExperimentAttributes.SIM_AVAIL if is_sim else (ExperimentAttributes.LOS_AVAIL if is_los else ExperimentAttributes.NLOS_AVAIL)
            avail = obj_info.get_object_info(avail_attr, obj_id=obj_id, name=name, exp_num=exp_num)
            if not avail: continue

            crop = obj_info.get_object_info(ExperimentAttributes.CROP, obj_id=obj_id, name=name, exp_num=exp_num)
            crop_high = obj_info.get_object_info(ExperimentAttributes.CROP_HIGH, obj_id=obj_id, name=name, exp_num=exp_num)

            if classes[(obj_id, name)] == '' or class_num[classes[(obj_id,name)]] == -1: continue

            all_exts = ['_new_dataset'] if not is_sim else all_sim_exts
            all_images = []
            all_masks = []
            for ext in all_exts:
                image_77, masks = self._load_image(obj_id, name, is_sim, is_los, ext, exp_num, crop=crop, crop_high=crop_high) 
                if np.count_nonzero(image_77) == 0: failed = True # 
                all_images.append(torch.from_numpy(image_77))
                all_masks.append(torch.from_numpy(np.array(masks)))
            if failed: continue
            
            self.all_data.append(all_images)
            self.all_labels.append(class_num[classes[(obj_id, name)]]) 
            self.all_names.append(name)
            self.all_is_sim.append(is_sim)
            self.all_masks.append(all_masks)
            self.all_exp_nums.append(exp_num)

        self.norm_mean = 0.5
        self.norm_std = 0.5 if not self.relative_norm else 1.0
        if self.num_channels == 3:
            normalize = transforms.Normalize(mean=[self.norm_mean, self.norm_mean, self.norm_mean], std=[self.norm_std, self.norm_std, self.norm_std])
        else:
            normalize = transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
            normalize_nlos = transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        
        self.num_channels = 2 if self.two_channel else self.num_channels
        if augment:
            # Get to fit resnet shape and apply random augmentations
            transform_list = [transforms.ToPILImage(),
                            transforms.CenterCrop((224,224)),]
            if self.gaussian_blur: 
                transform_list.append(transforms.GaussianBlur(11, sigma=(0.1,4.0)))
            transform_list.extend([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(degrees=360),
                                    transforms.ToTensor(),
                                    transforms.RandomErasing(),
                                    transforms.RandomAffine(degrees=(0,360), translate=(0.1, 0.3), scale=(0.85,1.15)),
                                    ])
            if not self.relative_norm: 
                transform_list.append(normalize)
            self.transform = transforms.Compose(transform_list)
        else:
            # Only apply transformations to get to fit into resnet shape
            transform_list = [transforms.ToPILImage(),
                            transforms.CenterCrop((224,224)),]
            if self.gaussian_blur: 
                transform_list.append(transforms.GaussianBlur(11, sigma=(3.0,3.0)))
            transform_list.extend([transforms.ToTensor()])
            if not self.relative_norm: 
                transform_list.append(normalize)
            self.transform = transforms.Compose(transform_list)
            
    
        # Plot some examples
        if plot:
            for i in range(len(self.all_data)):
                data_transformed = self.__getitem__(i)[0]
                plt.title(f'{phase} Augment: {self.augment} Exp Num: {self.all_exp_nums[i]} {self.all_names[i]}')
                if self.two_channel:
                    plt.pcolormesh(transforms.ToPILImage()(torch.clamp(data_transformed[0], -1, 1)), cmap='jet')
                    plt.colorbar()
                    print(data_transformed[0])
                    plt.show() 
                    plt.pcolormesh(transforms.ToPILImage()(torch.clamp(data_transformed[1], -1, 1)), cmap='jet')
                    plt.colorbar()
                    print(data_transformed[0])
                    plt.show() 
                else:
                    # print(torch.max(data_transformed))
                    # print(torch.min(data_transformed))
                    # plt.pcolormesh(transforms.ToPILImage()(torch.clamp(data_transformed, -1, 1)), cmap='jet')
                    # plt.colorbar()
                    # # print(data_transformed[0])
                    # plt.show() 
                    data_transformed += 0.5
                    normalization = Normalize(vmin=0, vmax=255)
                    plt.pcolormesh(transforms.ToPILImage()(torch.clamp(data_transformed, -1, 1)), cmap='jet', norm=normalization)
                    plt.colorbar()
                    # print(data_transformed[0])
                    plt.show() 

    def get_sample_weights(self):
        weights = []
        for i in range(len(self.all_labels)):
            num_samples_per_class = len(np.where(np.array(self.all_labels) == self.all_labels[i])[0])
            weights.append(1.0/num_samples_per_class)
        return np.array(weights)
                
    def _load_image(self, obj_id, name, is_sim, is_los, ext, exp_num, crop=True, crop_high=False): 
        # Load image
        loader = GenericLoader(obj_id, name, is_sim=is_sim, is_los=is_los, exp_num=exp_num)
        if not is_sim:
            x_angle, y_angle, z_angle = loader.find_obj_angles()
        else:
            x_angle, y_angle, z_angle = (0, 0, 0) # TODO: Add more sim angles
        image_77, _, _ = loader.load_image_file(radar_type='77_ghz', x_angle=x_angle, y_angle=y_angle, z_angle=z_angle, background_subtraction=None, ext=ext, crop_high=crop_high, crop=crop)

        # Load mask(s)
        all_masks = []
        if is_sim or exp_num == '2':
            mask_ext = ext
        elif crop_high:
            mask_ext = f'{ext}_fix_high3'
        else:
            mask_ext = f'{ext}_fix3'
        if self.use_new_mask: mask_ext = f'{ext}_richard'
        elif self.use_multi_mask: mask_ext = f'{ext}_original'
        if self.use_multi_mask:
            total_num_masks = 10
            for mask_num in range(total_num_masks):
                full_mask_ext = f'{mask_ext}_{mask_num}'
                mask_77 = loader.load_radar_masks(radar_type='77_ghz',x_angle=x_angle, y_angle=y_angle, z_angle=z_angle, ext=full_mask_ext)['radar_mask']
                all_masks.append(mask_77)
        else:
            mask_77 = loader.load_radar_masks(radar_type='77_ghz',x_angle=x_angle, y_angle=y_angle, z_angle=z_angle, ext=mask_ext)['radar_mask']
            all_masks.append(mask_77)

        # Dilate mask(s) if appropriate
        if self.dilate_mask:
            kernel = np.ones((5, 5), np.uint8) 
            for i in range(len(all_masks)):
                all_masks[i] = cv2.dilate(all_masks[i].astype(np.uint8), kernel, iterations=3) 


        # Convert to 2D and normalize sim data
        image_77 = np.sum(np.abs(image_77), axis=2) / image_77.shape[2]
        if is_sim:
            image_77 /= (230*550*300)
        if exp_num == '2':
            image_77 *= 10
        return image_77, all_masks


    def __getitem__(self, idx):
        while True:
            data = self.all_data[idx][:]
            all_masks = self.all_masks[idx]
            for i in range(len(data)):
                masks = all_masks[i]
                num_masks = len(masks)
                if len(masks)==1:
                    mask = masks[0]
                else:
                    mask_ind = torch.randint(0,num_masks,(1,))[0]
                    mask = masks[mask_ind]
                
                if self.apply_mask: 
                    data[i][mask==0] = 0

                if self.two_channel:
                    data[i] = np.concatenate([data[i][:,:,np.newaxis], mask[:,:,np.newaxis]], axis=2)
                    data[i] = np.transpose(data[i], [2,0,1])
                elif self.mask_only:
                    data[i] = mask.type(torch.FloatTensor)#.astype(np.float32)
                elif self.num_channels == 3:
                    data[i] = np.concatenate([data[i][:,:,np.newaxis], data[i][:,:,np.newaxis], data[i][:,:,np.newaxis]], axis=2)
                    data[i] = np.transpose(data[i], [2,0,1])

            if self.all_is_sim[idx]:
                full_img = None
                total_weight = 0
                for i in range(len(self.max_sim_weights)):
                    weight = torch.rand(1)[0]*self.max_sim_weights[i]
                    weighted_data = weight * data[i]
                    if full_img is None: full_img = weighted_data
                    else: full_img += weighted_data
                    total_weight += weight
                data = full_img / total_weight
            else:
                data = data[0]

            num_nonzero = torch.count_nonzero(data)


            if self.add_noise:
                proceed = torch.rand(1)[0]
                if proceed > 0.75 or not self.conditional_noise: 
                    mean = torch.rand(1)[0]*0.1
                    noise = torch.normal(torch.max(data)*mean, 0.01*torch.max(data), data.shape)
                    noise = noise.expand(data.shape)
                    data += noise
                    data[data<0] = 0


            data = self.transform(data)

            if self.clip_vals:
                cutoff = torch.max(data) * 0.8
                data[data > cutoff] = cutoff

            new_nonzero = torch.count_nonzero(data+self.norm_mean/self.norm_std)
            if self.relative_norm:
                new_nonzero = torch.count_nonzero(data)

            if self.relative_norm: 
                # Scale -0.5 to 0.5
                data /= torch.max(data)
                data -= 0.5


            if new_nonzero > 0.5*num_nonzero:
                break

        return torch.as_tensor(data), torch.as_tensor(self.all_labels[idx])

    def __len__(self):
        return len(self.all_data)
