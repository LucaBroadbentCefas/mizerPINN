from __future__ import annotations
from .config import build_parser

def main():
 args=build_parser().parse_args(); print(f"Configured Tranche 1 species {args.species_index}: log_U state, N=S*U, scaled residual")
if __name__=='__main__': main()
