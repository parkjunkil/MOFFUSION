import glob
import os
from pathlib import Path

from itertools import tee

import trimesh
import numpy as np
import pymol
import pymeshlab 
from mesh_to_sdf import mesh_to_voxels, get_surface_point_cloud
#import torch
import os
import math

import numpy as np

import argparse

os.environ['PYOPENGL_PLATFORM'] = 'egl'


parser = argparse.ArgumentParser(description="Code for converting atomic file format")
parser.add_argument("-xyz", "--FileIn_xyz", action='store', dest="FileIn_xyz", type=str, help="enter input filename")
parser.add_argument("-cif", "--FileIn_cif", action='store', dest="FileIn_cif", type=str, help="enter output filename")
parser.add_argument("-out", "--FileOut", action='store', dest="FileOut", type=str, help="enter output filename")
args = parser.parse_args()



def normalize_cube(mesh, norm_length=60):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    #assert np.max(mesh.bounding_box.extents) <= 60, 'cell larger than set max value'

    vertices = mesh.vertices - mesh.bounding_box.centroid

    vertices *= 2 / norm_length

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def get_raster_points(voxel_resolution):
    points = np.meshgrid(
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution)
    )

    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)

    return points 


def fractional_to_cartesian(cell_params, fractional_coords):
    a, b, c, alpha, beta, gamma = cell_params
    alpha = math.pi *(alpha/180)
    beta = math.pi * (beta/180)
    gamma = math.pi * (gamma/180)

    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_gamma = math.cos(gamma)
    sin_gamma = math.sin(gamma)

    cell_matrix = np.array([
        [a, b * cos_gamma, c * cos_beta],
        [0, b * sin_gamma, c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma],
        [0, 0, c * math.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma) / sin_gamma]
    ])

    fractional_vector = np.array(fractional_coords)

    cartesian_coords = np.dot(cell_matrix, fractional_vector)
    
    return cartesian_coords

def get_raster_points_fractional(voxel_resolution, cell_params, max_length=60):

    factor = (2*voxel_resolution-1)/(2*voxel_resolution)
    
    points = np.meshgrid(
        np.linspace(- factor, factor, voxel_resolution*2),
        np.linspace(- factor, factor, voxel_resolution*2),
        np.linspace(- factor, factor, voxel_resolution*2)
    )
    
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)

    points = np.array([fractional_to_cartesian(cell_params,point)*2/max_length for point in points])
    
    return points



def get_larger_sdf(mesh_file, cell_params, voxel_resolution = 32):
    
    raster_points = get_raster_points_fractional(voxel_resolution, cell_params)
    
    mesh = trimesh.load(mesh_file)

    norm_mesh = normalize_cube(mesh)

    surface_point_cloud = get_surface_point_cloud(norm_mesh, bounding_radius = 3**0.5)

    sdf = surface_point_cloud.get_sdf_in_batches(raster_points)

    voxels = sdf.reshape((voxel_resolution*2, voxel_resolution*2, voxel_resolution*2))

    return voxels


def get_array(array, size=32*2):
    i,j,k = array
    return i%size, j%size, k%size

def get_replicate(index, size=32):
    i,j,k = index
    return get_array((i,j,k)), get_array((i+size,j,k)), get_array((i,j+size,k)), get_array((i,j,k+size)), get_array((i+size,j+size,k)), get_array((i+size,j,k+size)), get_array((i,j+size,k+size)), get_array((i+size,j+size,k+size))

def get_min(voxels, indices):
    min_val = np.inf
    for index in indices:
        val = voxels[int(index[0]), int(index[1]), int(index[2])]
        if min_val > val:
            min_val = val
    return min_val
            
def consider_pbc(voxels, voxel_resolution = 32):
    
    new_voxels = np.full((voxel_resolution, voxel_resolution, voxel_resolution), np.inf)

    for i in range(voxel_resolution):
        for j in range(voxel_resolution):
            for k in range(voxel_resolution):     
                
                new_voxels[i,j,k]  = get_min(voxels, get_replicate((i+voxel_resolution/2,j+voxel_resolution/2,k+voxel_resolution/2)))
                
    return new_voxels


from pymatgen.core import Structure
def get_cell_param(path_cif):
    structure = Structure.from_file(path_cif)

    lattice = structure.lattice
    return [lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma]


def periodic_sdf(path_xyz, path_cif, path_out, voxel_resolution = 32):

    xyz_id = (path_xyz.split('/')[-1]).split('.')[0]

    
    pymol.cmd.load(xyz_file)
    pymol.cmd.show('surface')
    pymol.cmd.set('surface_quality', '0')
    pymol.cmd.save(xyz_id+'.wrl')
    

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(xyz_id+'.wrl')
    ms.save_current_mesh(xyz_id+'.ply')

    cell_params = get_cell_param(path_cif)

    sdf_voxels = get_larger_sdf(xyz_id+'.ply', cell_params)

    periodic_sdf_voxels = consider_pbc(sdf_voxels)

    np.save(path_out, periodic_sdf_voxels)

#    os.system('rm '+xyz_id+'.wrl')
#    os.system('rm '+xyz_id+'.ply')

    return


         

if __name__== "__main__":
    
    xyz_file = args.FileIn_xyz      # Replace with your LAMMPS data file name
    cif_file = args.FileIn_cif  # Replace with your desired CIF output file name
    out_file = args.FileOut

    periodic_sdf(xyz_file, cif_file, out_file) 
