"""Updating the program in place.

The alternative was: open a browser, download a ZIP, extract it over the
folder, remember to pick "replace the files in the destination", then run
install.bat. Five steps to change some Python, done by hand, every time. This
is the same thing done properly.

Nothing belonging to the installation is in the archive to begin with --
config.toml, output/, models/ and .venv/ are all excluded from the repository,
so there is no version of this that can overwrite a setting or a recording.
The check below enforces that rather than trusting it.
"""

from __future__ import annotations

import filecmp
import logging
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

log = logging.getLogger(__name__)

ARCHIVE_URL = (
    "https://github.com/instarandope/phone-call-transcriber/archive/refs/heads/main.zip"
)

# Anything that is yours rather than the project's. None of it is in the
# archive, so this is a backstop against that ever changing by accident: a
# stray config.toml committed upstream would otherwise replace your settings
# on the next update, silently, which is the exact failure this project has
# already been bitten by once.
NEVER_REPLACE = frozenset({"config.toml", ".venv", "output", "models"})


class UpdateError(RuntimeError):
    """Raised when the update cannot be completed."""


@dataclass
class Result:
    changed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)

    @property
    def anything(self) -> bool:
        return bool(self.changed or self.added)

    @property
    def needs_install(self) -> bool:
        """Did a dependency change? Then the virtual environment is stale."""
        return "requirements.txt" in self.changed


def fetch(url: str = ARCHIVE_URL, timeout: tuple[int, int] = (30, 300)) -> bytes:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:
        raise UpdateError(
            f"could not download the update ({exc}).\n"
            f"  Check the internet connection, or download it by hand from\n"
            f"    {url}"
        ) from exc


def apply(archive: bytes, root: Path) -> Result:
    """Unpack the archive over `root`, leaving anything of yours alone."""
    result = Result()
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        try:
            with zipfile.ZipFile(_as_file(archive, staging)) as bundle:
                _safe_extract(bundle, staging)
        except zipfile.BadZipFile as exc:
            raise UpdateError(f"the download is not a valid archive ({exc})") from exc

        source = _project_root(staging)
        for item in sorted(source.rglob("*")):
            if item.is_dir():
                continue
            relative = item.relative_to(source)
            if _is_protected(relative):
                result.protected.append(str(relative))
                continue

            destination = root / relative
            if not destination.exists():
                result.added.append(str(relative))
            elif filecmp.cmp(item, destination, shallow=False):
                continue  # byte for byte identical; leave the mtime alone
            else:
                result.changed.append(str(relative))

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)

    return result


def _as_file(archive: bytes, staging: Path) -> Path:
    path = staging / "download.zip"
    path.write_bytes(archive)
    return path


def _project_root(staging: Path) -> Path:
    """GitHub wraps the tree in one folder named for the branch."""
    entries = [p for p in staging.iterdir() if p.is_dir()]
    if len(entries) == 1:
        return entries[0]
    raise UpdateError("the archive does not look like this project")


def _is_protected(relative: Path) -> bool:
    return bool(NEVER_REPLACE.intersection(relative.parts))


def _safe_extract(bundle: zipfile.ZipFile, target: Path) -> None:
    """Refuse members that would write outside the staging directory."""
    resolved = target.resolve()
    for member in bundle.namelist():
        destination = (resolved / member).resolve()
        if not str(destination).startswith(str(resolved)):
            raise UpdateError(f"archive member escapes the folder: {member}")
    bundle.extractall(target)  # noqa: S202 - members validated above
