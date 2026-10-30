"""The stable UUID query path must be provided by the image's own SDK."""

import subprocess
import sys
from pathlib import Path


def test_image_links_sdk_tool_before_switching_user():
    root = Path(__file__).resolve().parents[4]
    dockerfile = (root / "docker/Dockerfile").read_text()
    start = dockerfile.index(
        'RUN AMD_SMI="$(readlink -f "${ROCM_PATH}/bin/amd-smi")"'
    )
    end = dockerfile.index("\n\n", start)
    block = dockerfile[start:end]
    assert 'test -x "${AMD_SMI}"' in block
    assert 'ln -s "${AMD_SMI}" /usr/bin/amd-smi' in block
    assert 'test "$(readlink -f /usr/bin/amd-smi)" = "${AMD_SMI}"' in block
    assert start < dockerfile.index("USER ${HOST_UID}:${HOST_GID}")


def test_stable_tool_can_find_its_sdk_relative_cli_modules(tmp_path):
    root = Path(__file__).resolve().parents[4]
    dockerfile = (root / "docker/Dockerfile").read_text()
    start = dockerfile.index('RUN AMD_SMI="$(readlink')
    block = dockerfile[start : dockerfile.index("\n\n", start)][4:]
    sdk = tmp_path / "sdk"
    (sdk / "bin").mkdir(parents=True)
    cli = sdk / "libexec/amdsmi_cli"
    cli.mkdir(parents=True)
    (cli / "amdsmi_init.py").write_text("VERSION = 'fixture-sdk-version'\n")
    tool = sdk / "bin/amd-smi"
    tool.write_text(
        "import os, sys\n"
        "sys.path.append(os.path.dirname(os.path.abspath(__file__))"
        " + '/../libexec/amdsmi_cli')\n"
        "from amdsmi_init import VERSION\n"
        "print(VERSION)\n"
    )
    tool.chmod(0o755)
    usr = tmp_path / "usr"
    (usr / "bin").mkdir(parents=True)
    # Execute the real recipe against isolated CPU-only filesystem paths.
    block = block.replace("${ROCM_PATH}", str(sdk)).replace("/usr/", f"{usr}/")
    subprocess.run(["/bin/sh", "-ec", block], check=True, timeout=5)
    stable = usr / "bin/amd-smi"
    result = subprocess.run(
        [sys.executable, str(stable)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fixture-sdk-version"
    assert stable.resolve() == tool
    assert (usr / "libexec/amdsmi_cli").resolve() == cli
