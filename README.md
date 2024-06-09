# MOFFUSION

[[`arXiv`]to-be-uploaded]
[[`Project Page`]to-be-uploaded]
[[`BibTex`]to-be-uploaded]

Code release for the paper "Multi-modal conditioning for metal-organic frameworks generation using 3D modeling techniques".

![Architecture_대지 1 사본 20_대지 1 사본 21](https://github.com/parkjunkil/MOFFUSION/assets/88761984/9002e6c7-9689-4d0e-8d62-ccd72fd7f980)


MOFFUSION is a multi-modal conditional diffusion model for MOF generation. MOFFUSION showed exceptional generation performance compared to baseline models in terms of structure validity and property statistics. Diverse modalities of data, including numeric, categorical, text, and their combinations, were successfully handled for the conditional generation of MOFs. Notably, signed distance functions (SDFs) were used for the input representation of MOFs, marking their first implementation in the generation of porous materials (below).


<p align="center"><img src=https://github.com/parkjunkil/MOFFUSION/assets/88761984/fdfa3198-0895-455b-9b86-cad24670a0d2>


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




# Demo

## Download the pretrained weight

First create a foler `./saved_ckpt` to save the pre-trained weights. Then download the pre-trained weights from the provided links and put them in the `./saved_ckpt` folder.
```
mkdir saved_ckpt  # skip if there already exists

# VQVAE's checkpoint
wget [] -O saved_ckpt/vqvae.pth

# MOF Constructor's checkpoint 
wget [] -O saved_ckpt/mof_constructor_topo.pth
wget [] -O saved_ckpt/mof_constructor_BB.pth

# MOFFUSION: Unconditional model (uncond)
wget [] -O saved_ckpt/moffusion_uncond.pth

# MOFFUSION: Conditional models (topo, H2, text, node&lcd, vf&Sa)
wget [] -O saved_ckpt/moffusion_topo.pth
wget [] -O saved_ckpt/moffusion_H2.pth
wget [] -O saved_ckpt/moffusion_text.pth
wget [] -O saved_ckpt/moffusion_node_lcd.pth # optional
wget [] -O saved_ckpt/moffusion_vf_sa.pth # optional
```

## Run Juypter Notebooks
Please check the provided jupyter notebooks for how to use the code. First open the jupyter notebook server.
```
jupyter notebook
```

Then, open one of the following notebooks for the task you want to perform.

1. Unconditional generation: `demo_uncond.ipynb`
2. Conditional generation on topology: `demo_topo.ipynb`
3. Conditional generation on text: `demo_text.ipynb`
4. Conditional generation on hydrogen working capacity: `demo_H2.ipynb`
5. Pore crafting: `demo_pore_crafting.ipynb`

Note that the notebooks will automatically save the generated shapes in the `./samples` folder.
For example, if you run `demo_topo.ipynb`, the generated outputs will be saved in `./samples/Demo_topo`.



# Train MOFFUSION

## Preparing the data
```
We are now figuring out how to share an SDF dataset for MOFFUSION training, which is quite large (i.e., 60 GB). Please stay tuned !!
```

## Training
1. Train VQVAE
```
# BuildingNet
./launchers/train_vqvae.sh
```

After training, copy the trained VQVAE checkpoint to the `./saved_ckpt` folder. Let's say the name of the checkpoints are `vqvae-snet-all.ckpt` or `vqvae-bnet-all.ckpt`. This is necessary for training the Diffusion model. For SDFusion on various tasks, please see 2.~5. below.

2. Train MOF-Constructor (Optional)
```
We encourage users to use saved MOF-Constructor checkpoint files without needing to re-trian them.
However, if you want to re-train them, you can easily do it as all models are available in the repository.
```

3. Train MOFFUSION (without unconditional)
```
./launchers/train_moffusion_uncond.sh
```

4. Train MOFFUSION conditioned on hydrogen working capacity
```
./launchers/train_moffusion_H2.sh
```

5. Train MOFFUSION conditioned on topology
```
./launchers/train_moffusion_topo.sh
```

6. Train MOFFUSION conditioned on text
```
./launchers/train_moffusion_text.sh
```

7. Train MOFFUSION for multi-condioning
```
upcoming!
```

# <a name="citation"></a> Citation

If you find this code helpful, please consider citing:

1. Journal version
```BibTeX
@inproceedings{cheng2023sdfusion,
  author={Park, Junkil and Lee, Youhan and Kim, Jihan},
  title={Multi-modal conditioning for metal-organic frameworks generation using 3D modeling techniques},
  booktitle={},
  year={2024},
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
This code borrows heavely from [SDFUSION](https://github.com/yccyenchicheng/SDFusion)). The followings packages are required to compute the SDF: [pymol](https://freeglut.sourceforge.net/](https://github.com/schrodinger/pymol-open-source)), [mesh-to-sdf](https://www.ubuntuupdates.org/package/core/kinetic/universe/base/libtbb-dev](https://github.com/marian42/mesh_to_sdf)).

This work is supported in part by ... .
