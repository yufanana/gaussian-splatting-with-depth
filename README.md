# Gaussian Splatting with Depth

This repository contains the official authors implementation associated with the paper "Z-Splat: Z-Axis Gaussian Splatting for Camera-Sonar Fusion", which can be found [here](https://arxiv.org/abs/2404.04687).

## BibTeX

```
@misc{qu2024zsplatzaxisgaussiansplatting,
      title={Z-Splat: Z-Axis Gaussian Splatting for Camera-Sonar Fusion}, 
      author={Ziyuan Qu and Omkar Vengurlekar and Mohamad Qadri and Kevin Zhang and Michael Kaess and Christopher Metzler and Suren Jayasuriya and Adithya Pediredla},
      year={2024},
      eprint={2404.04687},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.04687}, 
}
```

## Cloning the Repository

The repository contains submodules, thus please check it out with

```bash
## HTTPS
git clone --recursive https://github.com/QuintonQu/gaussian-splatting-with-depth/tree/gs-depth-main
```

## Optimizer

The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used Visual Studio 2019 for Windows)
- CUDA SDK 11 for PyTorch extensions, install *after* Visual Studio (we used 11.8, **known issues with 11.6**)
- C++ Compiler and CUDA SDK must be compatible

### Setup

#### Local Setup

Our default, provided install method is based on Conda package and environment management:

```bash
SET DISUTILS_USE_SDK=1  # Windows only
conda env create -f environment.yml
conda activate zsplat
```

If there are issues with the submodules, install them manually.

```bash
pip install submodules/simple_knn
pip install submodules/diff-gaussian-rasterization
```

## Dataset

### Mitsuba

</details>

<details><summary>Folder structure</summary>

```
<location>
├── color
│   ├── 0000.npy
│   ├── 0001.npy
│   └── ...
├── depth
│   ├── 0000.npy
│   ├── 0001.npy
│   └── ...
└── camera
    ├── fov.npy
    └── to_worlds.npy
```

</details>

### COLMAP

<details><summary>Folder structure</summary>

```
<location>
├── input
│   ├── <image 0>
│   ├── <image 1>
│   └── ...
└── distorted
    ├── database.db
    └── sparse
          └── 0
              ├── cameras.bin
              ├── images.bin
              ├── points3D.bin
              ├── points3D.ply
              └── project.ini
```
</details>

## Interactive Viewers
We provide two interactive viewers for our method: remote and real-time. Our viewing solutions are based on the [SIBR](https://sibr.gitlabpages.inria.fr/) framework, developed by the GRAPHDECO group for several novel-view synthesis projects.

### Hardware Requirements
- OpenGL 4.5-ready GPU and drivers (or latest MESA software)
- 4 GB VRAM recommended
- CUDA-ready GPU with Compute Capability 7.0+ (only for Real-Time Viewer)

### Software Requirements
- Visual Studio or g++, **not Clang** (we used Visual Studio 2019 for Windows)
- CUDA SDK 11, install *after* Visual Studio (we used 11.8)
- CMake (recent version, we used 3.24)
- 7zip (only on Windows)

### Pre-built Windows Binaries
We provide pre-built binaries for Windows [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip). We recommend using them on Windows for an efficient setup, since the building of SIBR involves several external dependencies that must be downloaded and compiled on-the-fly.

### Installation from Source
If you cloned with submodules (e.g., using ```--recursive```), the source code for the viewers is found in ```SIBR_viewers```. The network viewer runs within the SIBR framework for Image-based Rendering applications.

#### Windows
CMake should take care of your dependencies.
```shell
cd SIBR_viewers
cmake -Bbuild .
cmake --build build --target install --config RelWithDebInfo
```
You may specify a different configuration, e.g. ```Debug``` if you need more control during development.

#### Ubuntu 22.04
You will need to install a few dependencies before running the project setup.
```shell
# Dependencies
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
# Project setup
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # add -G Ninja to build faster
cmake --build build -j24 --target install
``` 

#### Ubuntu 20.04
Backwards compatibility with Focal Fossa is not fully tested, but building SIBR with CMake should still work after invoking
```shell
git checkout fossa_compatibility
```

### Navigation in SIBR Viewers
The SIBR interface provides several methods of navigating the scene. By default, you will be started with an FPS navigator, which you can control with ```W, A, S, D, Q, E``` for camera translation and ```I, K, J, L, U, O``` for rotation. Alternatively, you may want to use a Trackball-style navigator (select from the floating menu). You can also snap to a camera from the data set with the ```Snap to``` button or find the closest camera with ```Snap to closest```. The floating menues also allow you to change the navigation speed. You can use the ```Scaling Modifier``` to control the size of the displayed Gaussians, or show the initial point cloud.