from __future__ import annotations
import argparse, subprocess, sys
from PINNmizer.io import load_mizer_inputs
from experiments.species_ensemble_pinn.known_state import KnownStateProvider

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--input-dir",required=True); p.add_argument("--known-state-csv",required=True)
    p.add_argument("--biology-label",required=True,choices=["detailed","trait"]); p.add_argument("extra",nargs=argparse.REMAINDER)
    a=p.parse_args(argv); params,n_init,_=load_mizer_inputs(a.input_dir); KnownStateProvider(a.known_state_csv,params,n_init,mode="dynamic-known")
    for idx in range(len(params.species)):
        subprocess.run([sys.executable,"-m","experiments.species_ensemble_pinn.train_species","--input-dir",a.input_dir,
            "--known-state-csv",a.known_state_csv,"--biology-label",a.biology_label,"--species-idx",str(idx),*a.extra],check=True)
if __name__ == "__main__": main()
