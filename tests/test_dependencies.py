"""Every third-party import must be declared.

Adding a feature and forgetting its dependency produces the worst kind of
failure: it works on the machine it was built on, installs cleanly everywhere
else, and only breaks when someone reaches the new feature. That is exactly how
sherpa-onnx shipped undeclared.
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "call_transcriber"

# Import name -> the name it is installed under, where they differ.
DISTRIBUTION = {
    "webrtcvad": "webrtcvad-wheels",
    "faster_whisper": "faster-whisper",
    "sherpa_onnx": "sherpa-onnx",
    "PIL": "Pillow",
    "yaml": "PyYAML",
}

# Imported for their side effects on packaging, not used directly.
IGNORE = {"call_transcriber"}

# Arrives with a declared package rather than on a line of its own, and must
# not get one. onnxruntime is sherpa-onnx's own dependency, and sherpa-onnx is
# compiled against a particular build of it -- asking pip to resolve
# onnxruntime separately invites a different build than the one the compiled
# extension expects. What matters is that the package it comes with is
# declared, which is what this asserts.
VIA = {"onnxruntime": "sherpa-onnx", "ctranslate2": "faster-whisper"}


def imported_packages() -> set[str]:
    found: set[str] = set()
    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import: our own code.
                if node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
    return {
        name for name in found
        if name not in sys.stdlib_module_names and name not in IGNORE
    }


def declared() -> set[str]:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    names = set()
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(line.split(">=")[0].split("==")[0].split("[")[0].strip().lower())
    return names


def test_every_third_party_import_is_in_requirements():
    have = declared()
    missing = sorted(
        name for name in imported_packages()
        if DISTRIBUTION.get(name, name).lower() not in have
        and VIA.get(name, "").lower() not in have
    )
    assert not missing, (
        f"imported but not declared in requirements.txt: {missing}. "
        f"install.bat would not install them, so the feature that needs them "
        f"fails only once someone reaches it."
    )


def test_a_package_that_arrives_with_another_still_needs_that_other_one():
    """onnxruntime is exempt from its own line only because sherpa-onnx brings
    it. Drop sherpa-onnx and the exemption has to stop applying."""
    have = declared()
    for name, provider in VIA.items():
        assert provider.lower() in have, (
            f"{name} is imported and only allowed because {provider} supplies it, "
            f"but {provider} is no longer declared"
        )


def test_the_optional_engines_are_declared():
    """These are opt-in features, but the library still has to be installed."""
    assert "sherpa-onnx" in declared()


def test_requirements_and_pyproject_agree():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for name in declared():
        # pystray and Pillow live under an extra in pyproject.
        if name in {"pystray", "pillow"}:
            continue
        assert name in pyproject.lower(), f"{name} is in requirements.txt but not pyproject.toml"


@pytest.mark.parametrize("module", sorted(imported_packages()))
def test_each_import_maps_to_something_installable(module):
    have = declared()
    assert (
        DISTRIBUTION.get(module, module).lower() in have
        or VIA.get(module, "").lower() in have
    ), f"nothing in requirements.txt installs {module}"
