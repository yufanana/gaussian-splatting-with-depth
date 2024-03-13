#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    z_density_path = os.path.join(model_path, name, "ours_{}".format(iteration), "z_density")
    color_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "color_npy")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(z_density_path, exist_ok=True)
    makedirs(color_npy_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_info = render(view, gaussians, pipeline, background)
        rendering = render_info["render"]
        z_density = render_info["z_density"]
        # print("Z density: {}".format(z_density))
        gt = view.original_image[0:3, :, :]
        original_id = view.image_name.split(".")[0]
        torchvision.utils.save_image(rendering, os.path.join(render_path, original_id + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, original_id + ".png"))
        # Save Density to NPY file
        np.save(os.path.join(z_density_path, original_id + ".npy"), z_density.cpu().numpy())
        render_numpy = rendering.cpu().numpy()
        render_numpy = np.clip(np.round(render_numpy * 255), 0, 255).astype(np.uint8) 
        render_numpy = np.transpose(render_numpy, (1, 2, 0))
        # np.save(os.path.join(color_npy_path, original_id + ".npy"), render_numpy)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)