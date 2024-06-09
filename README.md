# MOFFUSION

[[`arXiv`]to-be-uploaded]
[[`Project Page`](https://parkjunkil.github.io/MOFFUSION/)]
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

First create a folder `./saved_ckpt` to save the pre-trained weights. Then download the pre-trained weights from the provided links and put them in the `./saved_ckpt` folder.
```
mkdir saved_ckpt  # skip if there already exists

# VQVAE's checkpoint
wget https://figshare.com/ndownloader/files/46925977 -O saved_ckpt/vqvae.pth

# MOF Constructor's checkpoint 
wget https://figshare.com/ndownloader/files/46925971 -O saved_ckpt/mof_constructor_topo.pth
wget https://figshare.com/ndownloader/files/46925974 -O saved_ckpt/mof_constructor_BB.pth

# MOFFUSION's checkpoint
## Unconditional model (uncond)
wget https://figshare.com/ndownloader/files/46931689 -O saved_ckpt/moffusion_uncond.pth

## Conditional models (topo, H2, text)
wget https://figshare.com/ndownloader/files/46926004 -O saved_ckpt/moffusion_topo.pth
wget https://figshare.com/ndownloader/files/46931701-O saved_ckpt/moffusion_H2.pth
wget https://figshare.com/ndownloader/files/46925995 -O saved_ckpt/moffusion_text.pth
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

We are now figuring out how to share an SDF dataset for MOFFUSION training, which is quite large (i.e., 60 GB). Please stay tuned!!
Once we share the dataset, please follow the procedure outlined below.

## Training
1. Train VQVAE
```
./launchers/train_vqvae.sh

#After training, copy the trained VQVAE checkpoint to the `./saved_ckpt` folder (or any other folders), and specify the path in the launcher file.
```

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

1. Journal version (To be uploaded)
```BibTeX
@inproceedings{,
  author={Park, Junkil and Lee, Youhan and Kim, Jihan},
  title={Multi-modal conditioning for metal-organic frameworks generation using 3D modeling techniques},
  booktitle={},
  year={2024},
}
```
2. arxiv version
```BibTeX
@article{,
  author={Park, Junkil and Lee, Youhan and Kim, Jihan},
  title={Multi-modal conditioning for metal-organic frameworks generation using 3D modeling techniques},
  booktitle={},
  year={2024},
}
```

# <a name="issue"></a> Issues and FAQ
Coming soon!

# Acknowledgement
This code borrows heavely from [SDFUSION](https://github.com/yccyenchicheng/SDFusion).
The followings packages are required to compute the SDF: [pymol](https://freeglut.sourceforge.net/](https://github.com/schrodinger/pymol-open-source)), [mesh-to-sdf](https://www.ubuntuupdates.org/package/core/kinetic/universe/base/libtbb-dev](https://github.com/marian42/mesh_to_sdf)).

Funding information to be uploaded.
