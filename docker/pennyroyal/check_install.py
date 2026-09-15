"""CPU-only image check; model/GPU validation is a separate release gate."""

import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path

import nixl
import torch


def main():
    root = Path('/opt/pennyroyal')
    expected = {
        'torch': '2.13.0+cu130',
        'torchvision': '0.28.0+cu130',
        'torchaudio': '2.11.0+cu130',
        'flashinfer-python': '0.6.17',
        'sglang-kernel': '0.4.6.post1+cu130',
        'triton': '3.7.1',
        'nixl': '1.4.0',
        'nixl-cu13': '1.4.0',
    }
    installed = {name: metadata.version(name) for name in expected}
    if installed != expected or torch.version.cuda != '13.0':
        raise RuntimeError(f'Unexpected CUDA package set: {installed}, CUDA={torch.version.cuda}')
    sys.path.insert(0, str(root / '.ple-nvme'))
    import sglang_ssd_stream._io  # noqa: F401

    for name in ('serve-flash-next-frspec.sh', 'serve-flash-next.sh', 'serve-qwen38-27b-dflash2.sh'):
        subprocess.run(['bash', '-n', str(root / 'configs/pennyroyal' / name)], check=True)
    agent = nixl.nixl_agent(
        'penny-container-check',
        nixl.nixl_agent_config(enable_prog_thread=False, enable_listen_thread=False, backends=['POSIX']),
    )
    if not {'FILE_SEG', 'DRAM_SEG'} <= set(agent.get_backend_mem_types('POSIX')):
        raise RuntimeError('NIXL POSIX plugin is missing FILE/DRAM support')
    source = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty:
        raise RuntimeError(f'Image has modified tracked source: {dirty}')
    print(json.dumps({'source': source, 'packages': installed, 'posix_plugin': 'available', 'gpu_tested': False}, indent=2))


if __name__ == '__main__':
    main()
