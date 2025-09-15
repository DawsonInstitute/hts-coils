"""
Script: update_notebook_kernels.py
- Walks notebooks/ and updates top-level metadata to set kernelspec to 'hts-py311'
- Makes a .bak copy before editing
- Safe idempotent behavior
"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
KERNEL_SPEC = {
    "kernelspec": {
        "name": "hts-py311",
        "display_name": "HTS Coils (py3.11)",
        "language": "python"
    },
    "language_info": {
        "name": "python",
        "version": "3.11"
    }
}

def update_notebook(nb_path: Path) -> bool:
    # read
    text = nb_path.read_text(encoding='utf-8')
    nb = json.loads(text)
    metadata = nb.get('metadata', {})
    changed = False
    # Check kernelspec
    ks = metadata.get('kernelspec')
    if ks != KERNEL_SPEC['kernelspec']:
        metadata['kernelspec'] = KERNEL_SPEC['kernelspec']
        changed = True
    # Check language_info
    li = metadata.get('language_info')
    if li != KERNEL_SPEC['language_info']:
        metadata['language_info'] = KERNEL_SPEC['language_info']
        changed = True
    if changed:
        # Overwrite file in-place (repository has version control; backups not needed)
        nb['metadata'] = metadata
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return changed


def main():
    nb_files = sorted(NOTEBOOKS_DIR.glob('*.ipynb'))
    changed_files = []
    for nb in nb_files:
        try:
            if update_notebook(nb):
                changed_files.append(str(nb.name))
        except Exception as e:
            print(f"Failed to update {nb}: {e}")
    print("Updated notebooks:", changed_files)

if __name__ == '__main__':
    main()
