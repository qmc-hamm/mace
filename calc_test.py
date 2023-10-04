from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
import numpy as np
import time

from mace.calculators import MACECalculator

run_id = "d17b475d94ce4279ba2ef6f6e494f3ef"
model_uri = f"runs:/{run_id}/model/"
device = "cpu"

calculator = MACECalculator(model_path=model_uri + device, device=device)
init_conf = read('BOTNet-datasets/dataset_3BPA/test_300K.xyz', '0')
init_conf.set_calculator(calculator)

dyn = Langevin(init_conf, 0.5*units.fs, temperature_K=310, friction=5e-3)
def write_frame():
        dyn.atoms.write('md_3bpa.xyz', append=True)
dyn.attach(write_frame, interval=50)
dyn.run(100)
print("MD finished!")