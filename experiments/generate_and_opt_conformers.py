from glob import glob
import fire
import subprocess 
from typing import Optional, Union
from pathlib import Path
import os 
from tqdm import tqdm

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --output={jobname}.out
#SBATCH --error={jobname}.err
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

givemeconformer "{conf}"
xtb conformers.sdf --opt tight > xtb.out
"""

def submit_one(infile, outdir: Optional[Union[str, Path]] = None, submit: bool= False, subsample: Optional[int] = None,):
    with open(infile) as f:
        lines = f.readlines()
    
    if subsample is not None:
        lines = lines[:subsample]

    if outdir is None:
        outdir = Path(infile).stem
    os.makedirs(outdir, exist_ok=True)

    for i, line in tqdm(enumerate(lines)):
        conf = line.strip()
        slurm_file = SLURM_TEMPLATE.format(conf=conf, jobname=f"conf_{i}")
        confdir = os.path.join(outdir, str(i))
        os.makedirs(confdir, exist_ok=True)
        with open(os.path.join(confdir, "submit.slurm"), "w") as f:
            f.write(slurm_file)
        if submit:
            subprocess.run(["sbatch", f"submit.slurm"], cwd=confdir)

def submit_opts(infile, outdir: Optional[Union[str, Path]] = None, submit: bool= False, subsample: Optional[int] = None, folder: Optional[str] = None):
    if folder is not None:
        all_files = glob(os.path.join(folder, "*.txt"))
        for f in all_files:
            submit_one(f, os.path.join(folder, Path(f).stem), submit, subsample)
            print(f"Submitted {os.path.basename(f)}")
    else:
        submit_one(infile, outdir, submit, subsample)

    
if __name__ == "__main__":
    fire.Fire(submit_opts)