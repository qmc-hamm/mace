from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
import numpy as np
import time

from mace.calculators import MACECalculator
import argparse

# Two Types of model fetching 
# 1. type - runs : params - run_id, device, test_file, output_file
# Example command run -  python calc_test.py --run_id=d17b475d94ce4279ba2ef6f6e494f3ef --device=cpu --test_file=qmc/testing.xyz --output_file=output.xyz

# 2. type - models : params - name, version, device, test_file, output_file
# Example command run - python calc_test.py --name=mace --version=1 --device=cpu --test_file=qmc/testing.xyz --output_file=output.xyz


parser = argparse.ArgumentParser()
parser.add_argument("--run_id", help="MLFlow Run ID")
parser.add_argument("--name", help="Model Name for run type - models")
parser.add_argument("--version", help="Model Version for run type - models")
parser.add_argument("--device", help="Device the user wants to run", choices=["cpu", "cuda", "mps"] ,required=True)
parser.add_argument("--test_file", help="Test file", required=True)
parser.add_argument("--output_file", help="Output File", required=True)
args = parser.parse_args()


model_uri = ""
if args.run_id:
        model_uri = f"runs:/{args.run_id}/model/{args.device}"
else:
        model_uri = f"models:/{args.name.upper()}-{args.device.upper()}/{args.version}"


calculator = MACECalculator(model_path=model_uri, device=args.device, default_dtype="float32")
init_conf = read(args.test_file, '0')
init_conf.set_calculator(calculator)

dyn = Langevin(init_conf, 0.5*units.fs, temperature_K=310, friction=5e-3)
def write_frame():
        dyn.atoms.write(args.output_file, append=True)
dyn.attach(write_frame, interval=50)
dyn.run(100)
print("MD finished!")