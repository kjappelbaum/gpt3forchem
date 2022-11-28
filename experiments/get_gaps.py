from glob import glob 
import fire
from pathlib import Path
import os 
from typing import Optional, Union
from tqdm import tqdm
def get_homo_lumo_gap(file):
    with open(file) as f:
        lines = f.readlines()
    for line in lines[::-1]:
        if 'HOMO-LUMO' in line:
            return float(line.split()[-3])
    return None


def get_homo_lumo_gaps(files):
    gaps = []
    for file in tqdm(files):
        gaps.append(get_homo_lumo_gap(file))
    
    stems = [Path(file).parent for file in files]
    return dict(zip(stems, gaps))

def all_gaps_in_dir(dir, outfile: Optional[Union[str, Path]] = None):
    files = glob(os.path.join(dir, "*", "xtb.out"))
    gaps = get_homo_lumo_gaps(files)
    if outfile is None:
        outfile = os.path.join(f"{str(dir)}_gaps.txt")
    
    with open(outfile, "w") as f:
        for key, value in gaps.items():
            f.write(f"{key} {value}\n")


if __name__ == "__main__":
    fire.Fire(all_gaps_in_dir)