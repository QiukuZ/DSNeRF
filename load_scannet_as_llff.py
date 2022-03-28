# Add by QiukuZ
# add scannet dataset to DSNerf

from email.mime import base
from tkinter import image_names
import numpy as np
import os, imageio
import cv2

def load_scannet_data(basedir, index, with_depth=False, image_w=1296, image_h=968, depth_w=640, depth_h=480, factor=1):
    # load pose and 3x5 [R:t:hwf]
    pose_path = os.path.join(basedir, 'pose')
    image_path = os.path.join(basedir, 'color')
    bds = np.zeros([len(index), 2])
    poses = np.zeros([3, 5, len(index)])
    images = np.zeros([len(index), image_h, image_w, 3])
    intrinsic = np.loadtxt(os.path.join(basedir, "intrinsic/intrinsic_color.txt"))
    hwf = np.array([image_h, image_w, (intrinsic[0,0] + intrinsic[1,1]) / 2.0])
    for i, idx in enumerate(index):
        image_filename = os.path.join(image_path, f"{idx}.jpg")
        color = imageio.imread(image_filename)
        # if not factor == 1:
        images[i,:,:,:] = color.astype(np.float32)/255

    for i, idx in enumerate(index):
        pose = np.loadtxt(os.path.join(pose_path, f"{idx}.txt"))
        poses[:, :, i] = np.vstack([pose[:3,:4].transpose(), hwf]).transpose()
        bds[i,:] = [0.5, 5.0]

    # Important!!!! 
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # [-u, r, -t] -> [r, u, -t]
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = images.astype(np.float32)
    bds = bds.astype(np.float32)
    print("!!!")
    
    # 由于scannet depth和color尺寸并不相同
    # 其中depth w = 640, color w = 1296
    # 将深度认为是深度的一半分辨率
    depth_scale = 2
    depth_scale = int(depth_scale / factor)
    if with_depth:
        depth_gts = []
        x,y = np.meshgrid(np.linspace(0, depth_h-1, int(depth_h)), np.linspace(0, depth_w-1, int(depth_w)))
        x = x.reshape(1, -1)[0]
        y = y.reshape(1, -1)[0]
        depths_coord = np.vstack([x * depth_scale,y * depth_scale]).transpose()
        depth_path = os.path.join(basedir, 'depth')
        for i, idx in enumerate(index):
            depth = imageio.imread(os.path.join(depth_path, f"{idx}.png"))[x.astype(np.int32),y.astype(np.int32)]
            depth = depth / 1000.
            depth_mask = np.where(depth > 0.5)
            depth_n = depth[depth_mask]
            depth_gts.append({"depth":depth_n, "coord":depths_coord[depth_mask][:,::-1].astype(np.float32), "weight": np.ones_like(depth_n)})
        return images, poses, bds, depth_gts
    return images, poses, bds
