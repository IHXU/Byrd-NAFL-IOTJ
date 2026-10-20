"""Run a directory of experiment configurations sequentially per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', required=True)
    parser.add_argument('--gpus', default=None, help='Comma-separated GPU IDs; defaults to CUDA_VISIBLE_DEVICES or 0')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--dry-run', action='store_true')
    args=parser.parse_args()
    configs=sorted(Path(args.config_dir).rglob('*.yaml'))
    if not configs: parser.error('No YAML configurations found.')
    gpus=(args.gpus if args.gpus is not None else os.environ.get('CUDA_VISIBLE_DEVICES','0')).split(',')
    if args.device=='cpu': gpus=['']
    if not gpus or any(not gpu.strip() for gpu in gpus) and args.device!='cpu':
        parser.error('Use --device cpu when no GPU is selected.')
    groups=[configs[i::len(gpus)] for i in range(len(gpus))]
    def run_group(item):
        gpu, group=item
        failed=[]
        for config in group:
            cmd=[sys.executable, str(Path(__file__).with_name('run_experiment.py')), '--config', str(config), '--device', args.device]
            print(f'GPU {gpu or "CPU"}: {config}',flush=True)
            if not args.dry_run:
                result=subprocess.run(cmd, env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu))
                if result.returncode: failed.append(str(config))
        return failed
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        failed=[p for group in executor.map(run_group,zip(gpus,groups)) for p in group]
    if failed:
        print('Failed configurations:',*failed,sep='\n',file=sys.stderr)
        raise SystemExit(1)

if __name__=='__main__': main()
