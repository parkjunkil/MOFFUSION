from typing import Optional, Union, Dict, Any
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import pormake as pm
from pormake import Database, Locator


pm.log.disable_print()
pm.log.disable_file_print()   


def get_predifined_list(
        bb_dir: Optional[Union[str,Path]] = None, 
        topo_dir: Optional[Union[str,Path]] = None,
) -> Dict[str, Any]:
    # Basic settings for accessing database of pormake
    if isinstance(bb_dir, str):
        bb_dir = Path(bb_dir)
    if isinstance(topo_dir, str):
        topo_dir = Path(topo_dir)

    db = Database(bb_dir=bb_dir, topo_dir=topo_dir)
    # db.serialize()

    # Node information
    bbs = db._get_bb_list()
    node_bbs = [f for f in bbs if (f.startswith('N') or f.startswith('C'))]

    # Topology information
    topos = db._get_topology_list()
    failed_topo = ['ibb', 'mmo', 'css', 'tfy-a', 'elv', 'tsn', 'lcz', 'xbn', 'dgo', 'ten', 'scu-h', 'zim', 'ild', 'cds-t', 'crt', 'jsm', 'rht-x', 'mab', 'ddi', 'mhq', 'nbo-x', 'tcb', 'zst', 'she-d', 'ffg-a', 'cdh', 'ast-d', 'ffj-a', 'ddy', 'llw-z', 'tpt', 'utx', 'fnx', 'roa', 'nia-d', 'dnf-a', 'lcw_component_3', 'baz-a', 'yzh', 'dia-x']
    topos = list(set(topos).difference(set(failed_topo)))

    # Dictionary for the final result
    topo_dict = {}

    # Rmsd calculation (This step is time-consuming)
    for topo_name in tqdm(topos):
        # Obtain topology
        try:
            topo = db.get_topo(topo_name)
        except Exception:
            continue
        # Record rmsd_calculated node
        total = []

        for i, connection_point in enumerate(topo.unique_cn):
            # Obtain node candidates of proper coordination number
            node_candidates = []
            for node_bb in node_bbs:
                node = db.get_bb(node_bb)
                if node.n_connection_points == connection_point:
                    node_candidates.append(node_bb)

            # rmsd calculation
            loc = Locator()

            for node_bb in node_candidates:
                bb_xyz = db.get_bb(node_bb)
                rmsd = loc.calculate_rmsd(topo.unique_local_structures[i], bb_xyz)

                if rmsd > 0.3:
                    node_candidates.remove(node_bb)

            total.append(node_candidates)

        topo_dict[topo_name] = total

    return topo_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Pre-defined pormake building block list'
    )

    parser.add_argument('-b', '--bb-dir', '--building-block-dir', default=None)
    parser.add_argument('-t', '--topo-dir', '--topology-dir', default=None)
    parser.add_argument('-s', '--save', type=str, default='data/rmsd_calculated_node.pickle')

    args = parser.parse_args()

    topo_dict = get_predifined_list(bb_dir=args.bb_dir, topo_dir=args.topo_dir)

    with open(args.save, 'wb') as f:
        pickle.dump(topo_dict, f)









