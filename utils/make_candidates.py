from typing import Dict, Any, Optional, Union
import pickle
import random
import numpy as np
import argparse
from tqdm import tqdm
from itertools import chain
from pathlib import Path
import pormake as pm
from pormake import Database, BuildingBlock, Topology


pm.log.disable_print()
pm.log.disable_file_print()   


# Helper function for counting atoms in each building block (i.e. node, edge)
def count_atoms_in_bb(bb:BuildingBlock):
    if bb is None:
        return 0
    else:
        return np.sum(bb.atoms.get_chemical_symbols() != np.array('X'))
    
# Helper function for counting atoms in hmof
def calculate_n_atoms_of_mof(topo:Topology, nodes:dict, edges:dict):
    nt_counts = {}
    for nt in topo.unique_node_types:
        n_nt = np.sum(topo.node_types == nt)
        nt_counts[nt] = n_nt

    et_counts = {}
    for et in topo.unique_edge_types:
        n_et = np.sum(np.all(topo.edge_types == et[np.newaxis, :], axis=1))
        et_counts[tuple(et)] = n_et

    counts = 0
    for nt, node in nodes.items():
        counts += nt_counts[nt] * count_atoms_in_bb(node)

    for et, edge in edges.items():
        counts += et_counts[et] * count_atoms_in_bb(edge)

    return counts

# Helper function for making name of hmof
def make_mof_name(topo:Topology, nodes:dict, edges:dict):
    en = lambda x: x.name if x else "E0"
    
    node_names = [bb.name for bb in nodes.values()]
    edge_names = [en(bb) for bb in edges.values()]
    name = "+".join(
        [topo.name] + node_names + edge_names
    )
    return name


def contain_metal(_node, _edge):
    for _bb in chain(_node.values(), _edge.values()):
        if _bb is None:
            continue
        if _bb.has_metal:
            return True
    return False


def make_candidate(
        pre_defined_list: Dict[str, Any], 
        bb_dir: Optional[Union[str,Path]] = None, 
        topo_dir: Optional[Union[str,Path]] = None,
        max_n_atoms: int = 1000, 
        target_n_mofs: int = 10,
        has_metal: bool = True,
    ):

    if isinstance(bb_dir, str):
        bb_dir = Path(bb_dir)
    if isinstance(topo_dir, str):
        topo_dir = Path(topo_dir)

    # Basic settings for accessing database of pormake
    db = Database(bb_dir=bb_dir, topo_dir=topo_dir)

    failed_topo = ['ibb', 'mmo', 'css', 'tfy-a', 'elv', 'tsn', 'lcz', 'xbn', 'dgo', 'ten', 'scu-h', 'zim', 'ild', 'cds-t', 'crt', 'jsm', 'rht-x', 'mab', 'ddi', 'mhq', 'nbo-x', 'tcb', 'zst', 'she-d', 'ffg-a', 'cdh', 'ast-d', 'ffj-a', 'ddy', 'llw-z', 'tpt', 'utx', 'fnx', 'roa', 'nia-d', 'dnf-a', 'lcw_component_3', 'baz-a', 'yzh', 'dia-x']

    # Topology info
    # 1. for specific topologies
    # topo = ['pcu', 'tbo']
    # 2. for all topologies
    topo = db._get_topology_list()
    topo = list(set(topo).difference(set(failed_topo)))

    # Node info
    # Last update (24.04)
    # NOTICE: The information of failed topologies is not included!
    # Dictionary format: {'topology name(str)':'list of RMSD-CALCULATED nodes for the given topology(list[list[], ... ])'}
    # Criterion: rmsd <= 0.3
    # Change the directory of topology_with_rmsd_calculated_node.pickle file if necessary
    # pre_defined_list = './data/topology_with_rmsd_calculated_node.pickle'

    try:
        if not Path(pre_defined_list).resolve().exists():
            raise Exception('Error: topology_with_rmsd_caculated_node.pickle does not exist!')
    except Exception as e:
        print(e)
        exit()
            
    node = dict()
    with open(pre_defined_list, 'rb') as a:
        node = pickle.load(a)

    # Edge info

    edge = [f for f in db._get_bb_list() if f.startswith('E') or f.startswith('L')] + ['E0']
    hmof_candidates = []

    # Generate hMOFs
    pbar = tqdm(total=target_n_mofs)
    while len(hmof_candidates) < target_n_mofs:
        # Choose random topology
        _topo_name = random.sample(topo, 1)[0]

        try:
            assert _topo_name in node
            _topo = db.get_topo(_topo_name)
        except:
            continue

        # Check node validation (Some nodes of topology is empty because of rmsd_value) 
        is_Valid = True
        for node_info in node[_topo_name]:
            if node_info == []:
                is_Valid = False
                break
        
        if not is_Valid:
            continue

        # Choose random node & edge for the given topology
        _node = {k: db.get_bb(random.sample(v, 1)[0]) for k, v in zip(_topo.unique_node_types, node[_topo_name])}
        _edge = {}
        for k in _topo.unique_edge_types:
            random_edge = random.sample(edge, 1)[0]
            if random_edge != 'E0':
                _edge[tuple(k)] = db.get_bb(random_edge)
            else:
                _edge[tuple(k)] = None

        # Check MOF
        if has_metal and not contain_metal(_node, _edge):
            continue

        # Check the number of atoms
        n_atoms = calculate_n_atoms_of_mof(_topo, _node, _edge)
        if n_atoms > max_n_atoms:
            continue
        
        mof_name = make_mof_name(_topo, _node, _edge)

        # Check duplication
        if mof_name in hmof_candidates:
            continue

        # Add hmof candidates
        hmof_candidates.append(mof_name)
        pbar.update(1)

    return hmof_candidates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='make candidates'
    )

    parser.add_argument('-b', '--bb-dir', '--building-block-dir', default=None)
    parser.add_argument('-t', '--topo-dir', '--topology-dir', default=None)
    parser.add_argument('-p', '--pre-defined-list', type=str, default='data/rmsd_calculated_node.pickle')
    parser.add_argument('-s', '--save', type=str, default='hmof_candidates.txt')
    parser.add_argument('-m', '--max-n-atoms', type=int, default=1000, help='How many atoms do hMOFs have at most?')
    parser.add_argument('-n', '--target-n-mofs', type=int, default=10, help='How many hMOFS do you want to generate?')
    parser.add_argument('-hm', '--has-metal', type=bool, default=True, help='If True, structure must have metal atoms (for MOF)')

    args = parser.parse_args()

    hmof_candidates = make_candidate(
        pre_defined_list=args.pre_defined_list, 
        bb_dir=args.bb_dir, 
        topo_dir=args.topo_dir,
        max_n_atoms=args.max_n_atoms,
        target_n_mofs=args.target_n_mofs,
        has_metal=args.has_metal,
    )

    # Save your hmof in a txt file
    # Change the directory if necessary
    with open(args.save, 'w') as f:
        for hmof in hmof_candidates:
            f.write(f'{hmof}\n')