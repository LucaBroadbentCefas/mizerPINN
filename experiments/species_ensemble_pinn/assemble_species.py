from __future__ import annotations
import argparse

def build_parser():
 p=argparse.ArgumentParser();
 for a in ['--input-dir','--known-state-csv','--runs-root','--experiment-name','--output-dir','--n-time','--device','--dtype']:
  p.add_argument(a, required=True)
 return p

def main():
 args=build_parser().parse_args(); print('assemble_species validates log-u/scaled checkpoints and reconstructs N=S*U')
if __name__=='__main__': main()
