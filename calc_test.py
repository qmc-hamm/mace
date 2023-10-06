from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
import numpy as np
import time

from mace.calculators import MACECalculator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", help="MLFlow Run ID", required=True)
parser.add_argument("--device", help="Device the user wants to run", required=True)
parser.add_argument("--test_file", help="Test file", required=True)
parser.add_argument("--output_file", help="Output File", required=True)
args = parser.parse_args()

model_uri = f"runs:/{args.run_id}/model/"


calculator = MACECalculator(model_path=model_uri + args.device, device=args.device, default_dtype="float32")
init_conf = read(args.test_file, '0')
init_conf.set_calculator(calculator)

dyn = Langevin(init_conf, 0.5*units.fs, temperature_K=310, friction=5e-3)
def write_frame():
        dyn.atoms.write(args.output_file, append=True)
dyn.attach(write_frame, interval=50)
dyn.run(100)
print("MD finished!")