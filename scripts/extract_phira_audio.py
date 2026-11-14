"""把谱面 zip 里的音轨按 sha256 摊平到一个目录，供 batch_extract_features 读取。

batch_extract_features 的 --audio-dir 约定：目录里每个文件按 `<audio_sha256>.<ext>` 命名，
它用 p.name.split('.')[0] 取指纹。而谱面 zip 里音频名是随机的（还有不带扩展名的），
所以需要这一层：读 zip -> 算 sha256 -> 同指纹只写一份。

zip 一律只读，不修改。

  python -m scripts.extract_phira_audio --zips /data/phira_charts --output /data/phira_audio
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUDIO_EXT = ('.mp3', '.ogg', '.wav', '.flac', '.m4a', '.aac', '.opus')
CONFIG_EXT = ('.json', '.yml', '.yaml', '.txt', '.md')
IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')


def _music_name(z):
    for entry in z.namelist():
        if not entry.lower().endswith(('.yml', '.yaml')):
            continue
        try:
            text = z.read(entry).decode('utf-8', 'replace')
        except Exception:                        # noqa: BLE001
            continue
        for line in text.splitlines():
            s = line.strip()
            if s.lower().startswith('music:'):
                return s.split(':', 1)[1].strip().strip('\'"')
    return None


def pick_audio(z):
    """返回 (成员名, 扩展名)。优先 info.yml 的 music，其次扩展名，最后取最大的非配置非图片文件。"""
    names = {n.lower().rsplit('/', 1)[-1]: n for n in z.namelist()}
    raw = _music_name(z)
    if raw:
        hit = names.get(raw.lower().rsplit('/', 1)[-1])
        if hit:
            return hit, Path(hit).suffix.lower()
    hits = [i for i in z.infolist() if i.filename.lower().endswith(AUDIO_EXT)]
    if hits:
        big = max(hits, key=lambda i: i.file_size)
        return big.filename, Path(big.filename).suffix.lower()
    others = [i for i in z.infolist() if not i.filename.lower().endswith(CONFIG_EXT + IMAGE_EXT)]
    if others:
        big = max(others, key=lambda i: i.file_size)
        return big.filename, Path(big.filename).suffix.lower()
    return None, ''


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--zips', required=True, help='谱面 zip 目录')
    ap.add_argument('--output', required=True, help='摊平后的音频目录')
    ap.add_argument('--every', type=int, default=200)
    args = ap.parse_args()

    zips = Path(args.zips)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    index_path = out / 'audio_index.json'
    index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.is_file() else {}
    known = {v['sha256'] for v in index.values() if isinstance(v, dict) and v.get('sha256')}

    files = sorted(zips.glob('*.zip'))
    written = skipped = failed = 0
    for i, zp in enumerate(files, 1):
        try:
            with zipfile.ZipFile(zp) as z:
                member, ext = pick_audio(z)
                if member is None:
                    failed += 1
                    index[zp.stem] = {'error': 'no_audio'}
                    continue
                data = z.read(member)
        except Exception as e:                   # noqa: BLE001
            failed += 1
            index[zp.stem] = {'error': f'{type(e).__name__}: {e}'[:160]}
            continue
        sha = hashlib.sha256(data).hexdigest()
        if sha in known:
            skipped += 1
        else:
            (out / f'{sha}{ext or ".bin"}').write_bytes(data)
            known.add(sha)
            written += 1
        index[zp.stem] = {'member': member, 'sha256': sha, 'bytes': len(data), 'ext': ext}
        if args.every and i % args.every == 0:
            print(f'[{i}/{len(files)}] 新写 {written} 已有 {skipped} 失败 {failed}', flush=True)

    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n摊平完成：新写 {written} 份，已存在跳过 {skipped}，失败 {failed}')
    print(f'输出：{out}（索引 {index_path}）')
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
