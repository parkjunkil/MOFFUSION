from typing import Optional, Union
from pathlib import Path
import pormake as pm
import argparse


pm.log.disable_print()
pm.log.disable_file_print()   


def serialize(
        bb_dir: Optional[Union[str,Path]] = None, 
        topo_dir: Optional[Union[str,Path]] = None,
):
    if isinstance(topo_dir, str):
        topo_dir = Path(topo_dir)

    db = pm.Database(bb_dir=bb_dir, topo_dir=topo_dir)
    db.serialize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Serialize topology files in pormake'
    )

    parser.add_argument('-b', '--bb-dir', '--building-block-dir', default=None)
    parser.add_argument('-t', '--topo-dir', '--topology-dir', default=None)
    args = parser.parse_args()

    topo_dict = serialize(bb_dir = args.bb_dir, topo_dir=args.topo_dir)