import sys
import numpy as np

from pormake import *

import py3Dmol
from ipywidgets import interact,fixed,IntSlider
import ipywidgets

def generate_mofs(candidate_file, save_dir, fail_dir):
    print("Loading topologies...")
    #topo_path = "pormake/data/topologies-20200107.npz"
    #topologies = np.load(topo_path, allow_pickle=True)["topologies"].item()

    print("Loading building blocks...")
    _bb_data = np.load("pormake/data/building_blocks-20200107.npz", allow_pickle=True)
    node_bbs = _bb_data["node_bbs"].tolist()
    edge_bbs = _bb_data["edge_bbs"].tolist()

    # inverse map.
    name2bb = {bb.name: bb for bb in node_bbs+edge_bbs}
    name2bb["E0"] = None


    # ===========================================================

    with open(candidate_file, "r") as f:
        mof_names = f.read().split()

    # Turn off logs.
    log.disable_print()
    log.disable_file_print()


    print("Start generation.")
    for name in mof_names:
        #print(name, end=" ")

        try:
            mof = name_to_mof(name, topologies, name2bb)

            if isinstance(mof, str):
                print(mof, ", skip.")
                continue

            min_cell_length = np.min(mof.atoms.cell.cellpar()[:3])
            if min_cell_length < 4.5:
                print(f"{name} Too small cell. Skip.")
                continue

            max_cell_length = np.max(mof.atoms.cell.cellpar()[:3])
            if max_cell_length < 60.0:
                mof.write_cif("{}/{}.cif".format(save_dir, name))
                print(f"{name} Success.")
            else:
                mof.write_cif("{}/{}.cif".format(fail_dir, name))
                print(f"{name} Success (large cell).")
        except Exception as e:
            continue
            #print("Fails.", e)
    

def count_normal_atoms(bb):
    if bb is None:
        return 0
    else:
        return np.sum(bb.atoms.get_chemical_symbols() != np.array("X"))

def calculate_n_atoms_of_mof(_topology, _node_bbs, _edge_bbs):
    nt_counts = {}
    for nt in _topology.unique_node_types:
        n_nt = np.sum(_topology.node_types == nt)
        nt_counts[nt] = n_nt

    et_counts = {}
    for et in _topology.unique_edge_types:
        n_et = np.sum(
            np.all(_topology.edge_types == et[np.newaxis, :], axis=1)
        )
        et_counts[tuple(et)] = n_et

    counts = 0
    for nt, bb in enumerate(_node_bbs):
        counts += nt_counts[nt] * count_normal_atoms(bb)

    for et, bb in _edge_bbs.items():
        counts += et_counts[et] * count_normal_atoms(bb)

    return counts

# MOF building function.
def name_to_mof(_mof_name, topologies, name2bb):
    tokens = _mof_name.split("+")
    _topo_name = tokens[0]

    _node_bb_names = []
    _edge_bb_names = []
    for bb in tokens[1:]:
        if bb.startswith("N"):
            _node_bb_names.append(bb)

        if bb.startswith("E"):
            _edge_bb_names.append(bb)

    _topology = topologies[_topo_name]
    _node_bbs = [name2bb[n] for n in _node_bb_names]
    _edge_bbs = {tuple(et): name2bb[n]
                 for et, n in zip(_topology.unique_edge_types, _edge_bb_names)}

    # Check # of atoms of target MOF.
    n_atoms = calculate_n_atoms_of_mof(_topology, _node_bbs, _edge_bbs)
    if n_atoms > 1500:
        return "Too Many Atoms"

    # Check COF.
    has_metal = False
    for _bb in _node_bbs+list(_edge_bbs.values()):
        if _bb is None:
            continue
        if _bb.has_metal:
            has_metal = True
    if not has_metal:
        return "COF"

    _builder = Builder()
    _mof = _builder.build_by_type(_topology, _node_bbs, _edge_bbs)

    return _mof

'''
def CifTo3DView(cif, size=(600,300), style="stick", opacity=0.5):
    with open(cif, 'r') as f:
        cif_data = f.read()

    assert style in ('line', 'stick', 'sphere', 'carton')
    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(cif_data, 'cif')
    viewer.addUnitCell()
    viewer.setStyle({style:{}})
    viewer.setBackgroundColor('white')
    viewer.zoomTo()
    return viewer

def mof_viewer(cif_files, idx):
    mof = cif_files[idx]
    return CifTo3DView(mof).show()
'''
