# MOFFUSION

[[`arXiv`]to-be-uploaded]
[[`Project Page`]to-be-uploaded]
[[`BibTex`]to-be-uploaded]

Code release for the paper "Multi-modal conditioning for metal-organic frameworks generation using 3D modeling techniques".

![Architecture_대지 1 사본 20](https://github.com/parkjunkil/MOFFUSION/assets/88761984/baeab366-33b3-44a2-9646-585f6b1106ae)

MOFFUSION is a multi-modal conditional diffusion model for MOF generation. Diverse modalities of data, including numeric, categorical, and text data, can be used for the conditional generation of MOFs.


# Data Preparation
```
mkdir data outputs # from root directory
cd data
mkdir sdfs splits logs
cd sdfs; mkdir resolution_32; cd ..

cd ../outputs
mkdir logs
```


# Installation
We recommend to build a [`conda`](https://www.anaconda.com/products/distribution) environment. You might need a different version of `cudatoolkit` depending on your GPU driver.
```
conda create -n moffusion python=3.9.18 -y && conda activate moffusion
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -y -c conda-forge cudatoolkit-dev  # this might take some time
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install h5py joblib termcolor scipy einops tqdm matplotlib opencv-python PyMCubes imageio trimesh omegaconf tensorboard notebook Pillow==9.5.0 py3Dmol ipywidgets transformers pormake
pip install -U scikit-learn
```




# Usage

## Training

```
PYTHONPATH=. sh launchers/train_vqvae_mof.sh
```

## Download the pretrained weight

First create a foler to save the pre-trained weights. Here we assume the folder is `./saved_ckpt`. Then download the pre-trained weights from the provided links and put them in the `./saved_ckpt` folder.
```
mkdir saved_ckpt

# VQVAE's checkpoint
wget https://uofi.box.com/shared/static/zdb9pm9wmxaupzclc7m8gzluj20ja0b6.pth -O saved_ckpt/vqvae-snet-all.pth

# SDFusion
wget https://uofi.box.com/shared/static/ueo01ctnlzobp2dmvd8iexy1bdsquuc1.pth -O saved_ckpt/sdfusion-snet-all.pth

# SDFusion: single-view reconstruction (img2shape)
wget https://uofi.box.com/shared/static/01hnf7pbewft4115qkvv9zhh22v4d8ma.pth -O saved_ckpt/sdfusion-img2shape.pth

# SDFusion: text-guided shape generation (txt2shape)
wget https://uofi.box.com/shared/static/vyqs6aex3rwbgxweyl3qh21c8p6vu33f.pth -O saved_ckpt/sdfusion-txt2shape.pth

# SDFusion: multi-modal conditional shape generation (partial shape + [ img {and/or} txt] -> shape)
wget https://uofi.box.com/shared/static/d95l3465arc0ffley5vwmz8bscaubmhc.pth -O saved_ckpt/sdfusion-mm2shape.pth

```

## Demo
Please check the provided jupyter notebooks for how to use the code. First open the jupyter notebook server.
```
jupyter notebook
```

Then, open one of the following notebooks for the task you want to perform.

1. Unconditional generation and shape completion: `demo_uncond_shape_comp.ipynb`
2. Single-view reconstruction (img2shape): `demo_img2shape.ipynb`
3. Text-guided shape generation (txt2shape): `demo_txt2shape.ipynb`
4. Multi-modal conditional shape generation (partial shape + [ img | txt ] ): `demo_mm2shape.ipynb`
5. (coming soon!) Text-guided Texturization: `demo_txt2tex.ipynb`

Note that the notebooks will automatically save the generated shapes in the `./demo_results` folder.

# How to train the SDFusion

## Preparing the data

* First, depending on your OS, you might need to install the required packages/binaries via `brew` or `apt-get` for computing the SDF given a mesh. If you cannot run the preprocessing files, please ctrl+c & ctrl+v the error message and search it on Google (usually there will be a one-line solution), or open an issue on this repo. We will try to update the README with the reported issues and their solutions under the [Issues and FAQ](#issue) section.

* ShapeNet
    1. Download the ShapeNetV1 dataset from the [official website](https://www.shapenet.org/). Then, extract the downloaded file and put the extracted folder in the `./data` folder. Here we assume the extracted folder is at `./data/ShapeNet/ShapeNetCore.v1`.
    2. Run the following command for preprocessing the SDF from mesh.
```
mkdir -p data/ShapeNet && cd data/ShapeNet
wget [url for downloading ShapeNetV1]
unzip ShapeNetCore.v1.zip
./launchers/unzip_snet_zipfiles.sh # unzip the zip files
cd preprocess
./launchers/launch_create_sdf_shapenet.sh
```

* BuildingNet
    1. Download the BuildingNet dataset from the [official website](https://buildingnet.org/). After you fill out [the form](https://docs.google.com/forms/d/e/1FAIpQLSevg7fWWMYYMd1vaOdDloUX_55VOQK7PqS1DlniFV7_vuoI0w/viewform), please download the v0 version of the dataset and uncompress it under `./data`. Here we assume the extracted folder is `./data/BuildingNet_dataset_v0_1`.
    2. Run the following command for preprocessing the SDF from mesh.
```
cd preprocess
./launchers/launch_create_sdf_building.sh
cd ../
```

* Pix3D
    - First download the Pix3D dataset from the [official website](http://pix3d.csail.mit.edu): 
```
wget http://pix3d.csail.mit.edu/data/pix3d.zip -P data
cd data
unzip pix3d.zip
cd ../
```
    - Then, run the following command for preprocessing the SDF from mesh.
```
cd preprocess
./launchers/launch_create_sdf_pix3d.sh
cd ../
```

* ShapeNetRendering
    - Run the following command for getting the rendering images, which is provided by the [3D-R2N2](http://3d-r2n2.stanford.edu/) paper.
```
wget ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz -P data/ShapeNet
cd data/ShapeNet && tar -xvf ShapeNetRendering.tgz
cd ../../
```

* text2shape
    - Run the following command for setting up the text2shape dataset.
```
mkdir -p data/ShapeNet/text2shape
wget http://text2shape.stanford.edu/dataset/captions.tablechair.csv -P data/ShapeNet/text2shape
cd preprocess
./launchers/create_snet-text_splits.sh
```

## Training
1. Train VQVAE
```
# ShapeNet
./launchers/train_vqvae_snet.sh

# BuildingNet
./launchers/train_vqvae-bnet.sh
```

After training, copy the trained VQVAE checkpoint to the `./saved_ckpt` folder. Let's say the name of the checkpoints are `vqvae-snet-all.ckpt` or `vqvae-bnet-all.ckpt`. This is necessary for training the Diffusion model. For SDFusion on various tasks, please see 2.~5. below.

2. Train SDFusion on ShapeNet and BuildingNet

```
# ShapeNet
./launchers/train_sdfusion_snet.sh

# BuildingNet
./launchers/train_sdfusion_bnet.sh
```

3. Train SDFusion for single-view reconstruction
```
./launchers/train_sdfusion_img2shape.sh
```

4. Train SDFusion for text-guided shape generation
```
# text2shape
./launchers/train_sdfusion_txt2shape.sh
```

5. Train SDFusion for multi-modality shape generation
```
./launchers/train_sdfusion_mm2shape.sh
```

6. Train the text-guided texturization
```
coming soon!
```

# <a name="citation"></a> Citation

If you find this code helpful, please consider citing:

1. Conference version
```BibTeX
@inproceedings{cheng2023sdfusion,
  author={Cheng, Yen-Chi and Lee, Hsin-Ying and Tuyakov, Sergey and Schwing, Alex and Gui, Liangyan},
  title={{SDFusion}: Multimodal 3D Shape Completion, Reconstruction, and Generation},
  booktitle={CVPR},
  year={2023},
}
```
2. arxiv version
```BibTeX
@article{cheng2022sdfusion,
  author = {Cheng, Yen-Chi and Lee, Hsin-Ying and Tuyakov, Sergey and Schwing, Alex and Gui, Liangyan},
  title = {{SDFusion}: Multimodal 3D Shape Completion, Reconstruction, and Generation},
  journal = {arXiv},
  year = {2022},
}
```

# <a name="issue"></a> Issues and FAQ
Coming soon!

# Acknowledgement
This code borrows heavely from [LDM](https://github.com/CompVis/latent-diffusion), [AutoSDF](https://github.com/yccyenchicheng/AutoSDF/), [CycleGAN](https://github.com/junyanz/CycleGAN), [stable dreamfusion](https://github.com/ashawkey/stable-dreamfusion), [DISN](https://github.com/laughtervv/DISN). We thank the authors for their great work. The followings packages are required to compute the SDF: [freeglut3](https://freeglut.sourceforge.net/), [tbb](https://www.ubuntuupdates.org/package/core/kinetic/universe/base/libtbb-dev).

This work is supported in part by NSF under Grants 2008387, 2045586, 2106825, MRI 1725729, and NIFA award 2020-67021-32799. Thanks to NVIDIA for providing a GPU for debugging.
