import hashlib
import os
import re
import shutil
import sys
import tempfile

import requests
import tqdm

from .config import MODEL_DIR
from .log import get_logger


def _compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _filename_from_url(url: str, default: str = '') -> str:
    m = re.search(r'/([^/?]+)[^/]*$', url)
    return m.group(1) if m else default


def _download_with_progress(url: str, path: str):
    headers = {}
    downloaded_size = 0
    if os.path.isfile(path):
        downloaded_size = os.path.getsize(path)
        headers['Range'] = 'bytes=%d-' % downloaded_size
        headers['Accept-Encoding'] = 'deflate'

    r = requests.get(url, stream=True, allow_redirects=True, headers=headers)
    if downloaded_size and r.headers.get('Accept-Ranges') != 'bytes':
        r = requests.get(url, stream=True, allow_redirects=True)
        downloaded_size = 0
    total = int(r.headers.get('content-length', 0))
    chunk_size = 1024

    if r.ok:
        with tqdm.tqdm(
            desc=os.path.basename(path),
            initial=downloaded_size,
            total=total + downloaded_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar:
            with open(path, 'ab' if downloaded_size else 'wb') as f:
                is_tty = sys.stdout.isatty()
                count = 0
                for data in r.iter_content(chunk_size=chunk_size):
                    size = f.write(data)
                    bar.update(size)
                    count += 1
                    if not is_tty and count % 1000 == 0:
                        print(bar)
    else:
        raise RuntimeError(f'Download failed: "{url}" (HTTP {r.status_code})')


class WeightManager:
    _MODEL_SUB_DIR = ''
    _MODEL_MAPPING = {}

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        os.makedirs(self.model_dir, exist_ok=True)
        self._loaded = False
        self._downloaded = self._check_downloaded()

    @property
    def model_dir(self):
        return os.path.join(MODEL_DIR, self._MODEL_SUB_DIR)

    def _get_file_path(self, *args) -> str:
        return os.path.join(self.model_dir, *args)

    def is_loaded(self) -> bool:
        return self._loaded

    def is_downloaded(self) -> bool:
        return self._downloaded

    def _check_downloaded(self) -> bool:
        for map_key in self._MODEL_MAPPING:
            if not self._check_downloaded_map(map_key):
                return False
        return True

    def _check_downloaded_map(self, map_key: str) -> bool:
        mapping = self._MODEL_MAPPING[map_key]
        if 'file' in mapping:
            path = mapping['file']
            if os.path.basename(path) in ('.', ''):
                path = os.path.join(path, _filename_from_url(mapping['url'], map_key))
        else:
            path = _filename_from_url(mapping['url'], map_key)
        if not os.path.exists(self._get_file_path(path)):
            return False
        return True

    async def download(self, force=False):
        if force or not self.is_downloaded():
            await self._download()
            self._downloaded = True

    async def _download(self):
        print(f'\nDownloading models into {self.model_dir}\n')
        for map_key, mapping in self._MODEL_MAPPING.items():
            if self._check_downloaded_map(map_key):
                print(f' -- Skipping {map_key} (already downloaded)')
                continue

            download_path = self._get_file_path(mapping.get('file', '.'))
            if os.path.basename(download_path) in ('', '.'):
                os.makedirs(download_path, exist_ok=True)
                download_path = os.path.join(download_path, _filename_from_url(mapping['url'], map_key))
            download_path += '.part'

            print(f' -- Downloading: "{mapping["url"]}"')
            _download_with_progress(mapping['url'], download_path)

            if 'hash' in mapping:
                print(f' -- Verifying: "{download_path}"')
                digest = _compute_sha256(download_path).lower()
                expected = mapping['hash'].lower()
                if digest != expected:
                    raise RuntimeError(
                        f'Hash mismatch: {digest} != {expected}'
                    )
                print(' -- Verifying: OK!')

            final_path = download_path[:-5]
            shutil.move(download_path, final_path)
            print()

    async def load(self, device: str, *args, **kwargs):
        if not self.is_downloaded():
            await self.download()
        if not self.is_loaded():
            await self._load(device=device, *args, **kwargs)
            self._loaded = True

    async def unload(self):
        if self.is_loaded():
            await self._unload()
            self._loaded = False

    async def infer(self, *args, **kwargs):
        if not self.is_loaded():
            raise RuntimeError(f'{self.__class__.__name__}: Model not loaded.')
        return await self._infer(*args, **kwargs)

    async def _load(self, device: str, *args, **kwargs):
        raise NotImplementedError

    async def _unload(self):
        raise NotImplementedError

    async def _infer(self, *args, **kwargs):
        raise NotImplementedError
