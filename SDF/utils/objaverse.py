
import os

import torch
from torch.utils import data

import numpy as np

import csv

import trimesh

class Objaverse(data.Dataset):
    def __init__(
        self, 
        split, 
        transform=None, 
        sdf_sampling=True, 
        sdf_size=4096, 
        surface_sampling=True, 
        surface_size=2048,
        dataset_folder='./data/',
        return_sdf=True,
        ):
        
        self.surface_size = surface_size

        self.transform = transform
        self.sdf_sampling = sdf_sampling
        self.sdf_size = sdf_size
        self.split = split

        self.surface_sampling = surface_sampling

        self.npz_folder = dataset_folder

        all_models = sorted([
            os.path.join(dataset_folder, m)
            for m in os.listdir(dataset_folder)
            if m.endswith('.npz')
        ])

        n = len(all_models)
        n_train = int(0.9 * n)
        if split == 'train':
            self.models = all_models[:n_train]
        else:
            self.models = all_models[n_train:]

        print(f"Objaverse [{split}]: {len(self.models)} / {n} shapes")
        
        self.return_sdf = return_sdf

    def __getitem__(self, idx):
        
        npz_path = self.models[idx]
        try:
            with np.load(npz_path) as data:
                vol_points = data['vol_points']
                vol_sdf = data['vol_sdf']
                near_points = data['near_points']
                near_sdf = data['near_sdf']
                surface = data['surface_points']
        except Exception as e:
            idx = np.random.randint(self.__len__())
            return self.__getitem__(idx)

        if self.surface_sampling:
            ind = np.random.default_rng().choice(surface.shape[0], self.surface_size, replace=False)
            surface = surface[ind]
            # surface_normals = surface_normals[ind]
            # surface_normals = trimesh.unitize(surface_normals)
        surface = torch.from_numpy(surface)

        # surface_normals = torch.from_numpy(surface_normals).float()

        if self.sdf_sampling:
            ### make sure balanced sampling, maybe not necessary when doing sdf regression
            pos_vol_id = vol_sdf<0

            if pos_vol_id.sum() > self.sdf_size//2:
                ind = np.random.default_rng().choice(pos_vol_id.sum(), self.sdf_size//2, replace=False)
                pos_vol_points = vol_points[pos_vol_id][ind]
                pos_vol_sdf = vol_sdf[pos_vol_id][ind]
            else:
                pos_vol_id = near_sdf<0
                if pos_vol_id.sum() > self.sdf_size//2:
                    ind = np.random.default_rng().choice(pos_vol_id.sum(), self.sdf_size//2, replace=False)
                    pos_vol_points = near_points[pos_vol_id][ind]
                    pos_vol_sdf = near_sdf[pos_vol_id][ind]
                else:
                    ind = np.random.default_rng().choice(vol_points.shape[0], self.sdf_size//2, replace=False)
                pos_vol_points = vol_points[ind]
                pos_vol_sdf = vol_sdf[ind]

            neg_vol_id = vol_sdf>=0

            ind = np.random.default_rng().choice(neg_vol_id.sum(), self.sdf_size//2, replace=False)
            neg_vol_points = vol_points[neg_vol_id][ind]
            neg_vol_sdf = vol_sdf[neg_vol_id][ind]

            vol_points = np.concatenate([pos_vol_points, neg_vol_points], axis=0)
            vol_sdf = np.concatenate([pos_vol_sdf, neg_vol_sdf], axis=0)
            ###

            ind = np.random.default_rng().choice(near_points.shape[0], self.sdf_size, replace=False)
            near_points = near_points[ind]
            near_sdf = near_sdf[ind]

        
        vol_points = torch.from_numpy(vol_points)
        vol_sdf = torch.from_numpy(vol_sdf).float()

        near_points = torch.from_numpy(near_points)
        near_sdf = torch.from_numpy(near_sdf).float()

        # Both splits return [vol_points, near_points] so that loss_vol can be
        # computed during validation.  Augmentation is only applied for train.
        points = torch.cat([vol_points, near_points], dim=0)
        sdf = torch.cat([vol_sdf, near_sdf], dim=0)

        if self.transform:
            surface, points = self.transform(surface, points)

        ## random rotation
        if self.split == 'train':
            perm = torch.randperm(3)
            points = points[:, perm]
            surface = surface[:, perm]

            negative = torch.randint(2, size=(3,)) * 2 - 1
            points *= negative[None]
            surface *= negative[None]

            roll = torch.randn(1)
            yaw = torch.randn(1)
            pitch = torch.randn(1)

            tensor_0 = torch.zeros(1)
            tensor_1 = torch.ones(1)

            RX = torch.stack([
                            torch.stack([tensor_1, tensor_0, tensor_0]),
                            torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                            torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)

            RY = torch.stack([
                            torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                            torch.stack([tensor_0, tensor_1, tensor_0]),
                            torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3,3)

            RZ = torch.stack([
                            torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                            torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                            torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)

            R = torch.mm(RZ, RY)
            R = torch.mm(R, RX)

            points = torch.mm(points, R).detach()
            surface = torch.mm(surface, R).detach()
            # surface_normals = torch.mm(surface_normals, R).detach()
              
        if self.return_sdf is False: # return occupancies (sign) instead
            sdf = (sdf<0).float()

        return points, sdf, surface, npz_path, npz_path#, surface_normals

    def __len__(self):
        return len(self.models)
