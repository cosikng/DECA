# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import numpy as np
from time import time
import argparse
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg


def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images
    testdata = datasets.TestData(args.image_path, iscrop=args.iscrop, face_detector=args.detector)
    expdata = datasets.TestData(args.exp_path, iscrop=args.iscrop, face_detector=args.detector)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config=deca_cfg, device=device)
    # identity reference
    i = 0
    name = testdata[i]['imagename']
    savepath = '{}/{}.jpg'.format(savefolder, name)
    images = testdata[i]['image'].to(device)[None, ...]
    with torch.no_grad():
        id_codedict = deca.encode(images)
        id_opdict, id_visdict = deca.decode(id_codedict)
    id_visdict = {x: id_visdict[x] for x in ['inputs', 'shape_detail_images']}

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置输出视频为mp4格式
    # cap_fps是帧率，根据自己需求设置帧率
    cap_fps = 30

    # size要和图片的size一样，但是通过img.shape得到图像的参数是（height，width，channel），
    # 可以实现在图片文件夹下查看图片属性，获得图片的分辨率
    size = (224, 224)  # size（width，height）
    # 设置输出视频的参数，如果是灰度图，可以加上 isColor = 0 这个参数
    # video = cv2.VideoWriter('results/result.avi',fourcc, cap_fps, size, isColor=0)
    video = cv2.VideoWriter('result2.mp4', fourcc, cap_fps, size)  # 设置保存视频的名称和路径，默认在根目录下
    # -- expression transfer
    # exp code from image
    for i in tqdm(range(len(expdata))):
        exp_images = expdata[i]['image'].to(device)[None, ...]
        with torch.no_grad():
            exp_codedict = deca.encode(exp_images)
            # transfer exp code
            id_codedict['pose'][:, 3:] = exp_codedict['pose'][:, 3:]
            id_codedict['exp'] = exp_codedict['exp']
            transfer_opdict, transfer_visdict = deca.decode(id_codedict)
            id_visdict['transferred_shape'] = transfer_visdict['shape_detail_images']
            to_save = dict()
            to_save['shape_detail_images'] = transfer_visdict['shape_detail_images']
            video.write(deca.visualize(to_save))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--image_path', default='TestSamples/examples/IMG_0392_inputs.jpg', type=str,
                        help='path to input image')
    parser.add_argument('-e', '--exp_path', default='TestSamples/2', type=str,
                        help='path to expression')
    parser.add_argument('-s', '--savefolder', default='TestSamples/animation_results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard')
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check detectos.py for details')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output')
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image')
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat')
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images')
    main(parser.parse_args())

    # main(parser.parse_args())
