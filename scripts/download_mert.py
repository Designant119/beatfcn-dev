"""按锁定 revision 下载 MERT-v2 权重，并逐文件校验 sha256。

这套流程对权重是强绑定的：m-a-p/MERT-v2-FullSong @7001f76，四个文件缺一不可，
任何一个 sha256 对不上，模型就不加载。所以不能用随便一个 MERT 版本，也不能
只下 pytorch_model.bin。

  python scripts/download_mert.py --out /data/weights/mert
"""
import argparse
import hashlib
import sys
from pathlib import Path

REPO = 'm-a-p/MERT-v2-FullSong'
REVISION = '7001f7657276238629e61c7746ce70735e0604c0'
FILES = {
    'config.json': 'f2e194895f58be3ddba327255db129ff0e3bee550cc0ecf08e4d22d79ce3bca3',
    'model.safetensors': 'e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d',
    'configuration_mert2.py': '77b53ec9d7ee31a599d744fb006e812c7eeaf7390deb46e2f460cf8c17b00bd6',
    'modeling_mert2.py': 'b1a3174e5649c4b26b0c90d8626f0adacfbbba111a58ed3bb72ad651945a2f5c',
}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='权重落地目录')
    ap.add_argument('--mirror', default='', help='可选 HF 镜像，例如 https://hf-mirror.com')
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return print('先 pip install huggingface_hub')

    for name, expected in FILES.items():
        target = out / name
        if target.is_file() and sha_file(target) == expected:
            print(f'已存在且校验通过 {name}')
            continue
        path = hf_hub_download(REPO, name, revision=REVISION, local_dir=str(out),
                               endpoint=args.mirror or None)
        got = sha_file(path)
        if got != expected:
            return print(f'{name} 校验失败：{got} != {expected}')
        print(f'下载并校验通过 {name}（{Path(path).stat().st_size >> 20} MB）')

    print(f'\n权重就绪：{out}')
    print('用法：--mert-source', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
