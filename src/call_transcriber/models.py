"""Downloading the optional model bundles.

Whisper fetches its own weights on first use. The sherpa-onnx models -- Parakeet
for speech, and the segmentation and embedding pair for working out who is
talking -- do not, so they are fetched here.

Kept out of install.bat deliberately: these are opt-in, they are large, and
Windows has no tar. Python has both the downloader and the extractor already.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


@dataclass(frozen=True)
class Bundle:
    key: str
    what: str
    url: str
    folder: str
    # A file that exists once the bundle is in place, so a re-run is a no-op
    # rather than a re-download.
    marker: str
    size_mb: int

    def target(self, root: Path) -> Path:
        return root / "models" / self.folder

    def installed(self, root: Path) -> bool:
        return (self.target(root) / self.marker).exists()


BUNDLES: tuple[Bundle, ...] = (
    Bundle(
        key="parakeet",
        what="Parakeet TDT speech recognition (int8)",
        url=f"{RELEASES}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
        folder="parakeet",
        marker="encoder.int8.onnx",
        size_mb=640,
    ),
    Bundle(
        key="segmentation",
        what="Speaker segmentation (who is talking, and when)",
        url=f"{RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        folder="diarize-segmentation",
        marker="model.onnx",
        size_mb=6,
    ),
    Bundle(
        key="embedding",
        what="Speaker embedding (deciding which voice is which)",
        url=f"{RELEASES}/speaker-recongition-models/nemo_en_titanet_small.onnx",
        folder="diarize-embedding",
        marker="nemo_en_titanet_small.onnx",
        size_mb=38,
    ),
)

BY_KEY = {b.key: b for b in BUNDLES}
DIARIZATION_KEYS = ("segmentation", "embedding")


def install(
    bundle: Bundle, root: Path, force: bool = False, archive: Path | None = None
) -> Path:
    """Fetch and unpack one bundle. Returns the folder it lives in.

    `archive` installs from a file already on disk instead of downloading,
    which is the way through a network that intercepts TLS or blocks GitHub.
    """
    target = bundle.target(root)
    if bundle.installed(root) and not force and archive is None:
        log.info("%s is already installed", bundle.what)
        return target

    target.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        if archive is not None:
            if not archive.exists():
                raise RuntimeError(f"{archive} does not exist")
            log.info("installing %s from %s", bundle.what, archive)
            download = archive
        else:
            log.info("downloading %s (~%d MB)", bundle.what, bundle.size_mb)
            download = Path(tmp) / Path(bundle.url).name
            _fetch(bundle.url, download, bundle)

        if download.suffix == ".onnx":
            shutil.copy2(download, target / download.name)
        else:
            _unpack(download, target)

    if not bundle.installed(root):
        raise RuntimeError(
            f"{bundle.what} downloaded but {bundle.marker} is not in {target}. "
            f"The archive layout may have changed upstream."
        )
    log.info("%s ready in %s", bundle.what, target)
    return target


def _fetch(url: str, destination: Path, bundle: Bundle) -> None:
    """Download with requests, which carries its own CA bundle.

    urllib defers to the operating system's certificate store, and on Windows
    that frequently lacks the chain these hosts present -- the download then
    fails with CERTIFICATE_VERIFY_FAILED even though nothing is wrong.
    requests ships certifi and does not have that problem.
    """
    try:
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            step = max(1, total // 20) if total else 0
            next_mark = step

            with open(destination, "wb") as handle:
                for chunk in response.iter_content(1 << 20):
                    handle.write(chunk)
                    done += len(chunk)
                    if step and done >= next_mark:
                        print(f"    {done / 1_048_576:6.0f} MB of {total / 1_048_576:.0f}")
                        next_mark += step
    except requests.exceptions.SSLError as exc:
        raise RuntimeError(_manual_instructions(bundle, f"the TLS handshake failed ({exc})")) from exc
    except requests.RequestException as exc:
        raise RuntimeError(_manual_instructions(bundle, str(exc))) from exc


def _manual_instructions(bundle: Bundle, reason: str) -> str:
    """When the download cannot be made to work, say how to do it by hand."""
    return (
        f"{reason}\n\n"
        f"  This is the download failing, not the model being unavailable. A\n"
        f"  company network that inspects HTTPS traffic will do this.\n\n"
        f"  To do it by hand instead:\n"
        f"    1. Open this in a browser and save the file:\n"
        f"         {bundle.url}\n"
        f"    2. Then run, with the path to what you downloaded:\n"
        f"         run.bat models --{bundle.key} --file \"C:\\path\\to\\{Path(bundle.url).name}\""
    )


def _unpack(archive: Path, target: Path) -> None:
    """Extract, flattening the single top-level folder these archives use."""
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        with tarfile.open(archive, "r:*") as tar:
            _safe_extract(tar, staging)

        entries = list(staging.iterdir())
        source = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging
        for item in source.iterdir():
            destination = target / item.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), str(destination))


def _safe_extract(tar: tarfile.TarFile, target: Path) -> None:
    """Refuse members that would write outside the target directory."""
    resolved = target.resolve()
    for member in tar.getmembers():
        destination = (resolved / member.name).resolve()
        if not str(destination).startswith(str(resolved)):
            raise RuntimeError(f"archive member escapes the target folder: {member.name}")
    tar.extractall(target)  # noqa: S202 - members validated above


def status(root: Path) -> list[tuple[Bundle, bool]]:
    return [(bundle, bundle.installed(root)) for bundle in BUNDLES]
