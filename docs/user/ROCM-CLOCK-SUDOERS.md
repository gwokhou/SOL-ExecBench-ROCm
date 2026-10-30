# Host AMD-SMI sudoers rule

SOL owns the generator at
[`scripts/setup_rocm_clock_sudoers.py`](../../scripts/setup_rocm_clock_sudoers.py),
its [tests](../../tests/sol_execbench/core/platform/test_setup_rocm_clock_sudoers.py),
and the reviewed [local rule](../../scripts/sudoers/sol-execbench-amd-smi).
No scheduler checkout or service is needed to render or install the rule.

The generator defaults to `/usr/bin/amd-smi` independently of PATH ordering.
UUID-scoped host setup, AUTO restoration and performance readback share this
default and the same selector. `SOL_EXECBENCH_HOST_AMD_SMI` explicitly sets an
alternative absolute host invocation path for both runtime and generator/check;
empty and relative values fail instead of falling back to PATH. The generator's
`--amd-smi` option takes precedence for that invocation only. When preparing a
different host rule with this option, configure runtime with the same path.
The checked-in local instance remains `/usr/bin/amd-smi`; using another path
requires its own reviewed rule and is not authorized by the local instance.
Legacy all-device and standalone PMC tool discovery are unchanged. The host
override is not forwarded to Docker containers.
An explicitly supplied path retains its spelling. `Path.resolve()` is used
only to validate root control of the executable and its link chain. The CLI
and renderer require a verified GPU UUID; neither generates all-device rules.
The generic generator accepts a user and tool path, but the checked-in local
instance grants **only guohao**, **only UUID
`a3007590-0000-1000-800f-a29c1cca1511`**, and only these exact commands:

```text
/usr/bin/amd-smi set -l STABLE_PEAK -g a3007590-0000-1000-800f-a29c1cca1511
/usr/bin/amd-smi set -l AUTO -g a3007590-0000-1000-800f-a29c1cca1511
```

The instance SHA256 is
`c626246847555aea85568c0541e5460425429e3aec7043786630aeea05499e55`.
It contains no privileged `version`, wildcard arguments, other GPU, or
`NOPASSWD: ALL` grant. The host column `ALL` does not grant arbitrary commands.
The existing Docker policy and legacy all-device runtime paths are unchanged;
this rule does not authorize them. In particular, the deprecated standalone
gfx1200 profiler wrapper still needs its separately reviewed legacy policy.

## Render and inspect without installing

From this SOL checkout, using the existing Python 3.14 interpreter:

```sh
PYTHONPATH=src python3 -B scripts/setup_rocm_clock_sudoers.py --mode print \
  --user guohao --gpu-uuid a3007590-0000-1000-800f-a29c1cca1511
/usr/sbin/visudo -cf scripts/sudoers/sol-execbench-amd-smi
```

Rendering prints text; it does not inspect or change hardware. Candidate-only
`visudo -cf` validates syntax, not the installed policy or live authentication.
The generator's default `check` mode performs only sudo command-list queries.
`allowed_authentication_unverified` is deliberately not reported as verified
coverage. Do not run `verify-live` as an installation check: it invokes both
setters and requires separate authorization and an idle target GPU.

## Local interactive installation

System administration authorization is required. This source handoff does not
install a rule or authorize a workload. Use a local interactive terminal for
sudo authentication; never send a password through chat. The last recorded
attempt stopped before installation because `sudo -n /usr/sbin/visudo -c`
required interactive authentication. Recheck the current state before acting.

Keep project workloads paused. First inspect `namei -l /usr/bin/amd-smi`:
the executable, link chain and parent directories must remain root-controlled,
with no non-root write access to files/directories. Symlink mode 0777 itself
does not grant permission to replace a link in a root-controlled directory.
Do not change PATH or substitute a resolved version-directory path in the rule.

From this SOL checkout, set the paths and verify the exact reviewed content:

```sh
sol_rule="$PWD/scripts/sudoers/sol-execbench-amd-smi"
sol_stage=/etc/sudoers.d/.sol-execbench-amd-smi.staged-20261027
sol_dest=/etc/sudoers.d/sol-execbench-amd-smi
sha256sum "$sol_rule"
/usr/sbin/visudo -cf "$sol_rule"
```

Proceed only if the hash matches the full value above and syntax passes.
The following sequence stops on any failure. Both destination and staging path
must be absent; an existing file must be reviewed, never silently overwritten.
Install the reviewed static file directly so no project Python environment is
run as root:

```sh
sudo visudo -c &&
sudo test ! -e "$sol_dest" && sudo test ! -L "$sol_dest" &&
sudo test ! -e "$sol_stage" && sudo test ! -L "$sol_stage" &&
sudo install -o root -g root -m 0440 "$sol_rule" "$sol_stage" &&
sudo visudo -cf "$sol_stage" &&
sudo mv -T "$sol_stage" "$sol_dest" &&
sudo visudo -c
```

The dot-prefixed staging file is not an active sudoers include. Do not move,
delete or edit any other rules. After installation, use `sudo -n -ll` to inspect
the exact two entries and their `!authenticate` tag. Listing success is not
setter success. Installation does not clear a native owner, change admission
state or permit automatic recovery or task retries.

## Rollback

If final policy validation fails, or rollback is separately selected, verify
that the destination still has the exact reviewed hash. A different hash
requires manual review. Only the file installed here is moved out of the parsed
directory; other rules remain untouched:

```sh
sudo sha256sum "$sol_dest"
# Continue only for the exact c6262468...99e55 hash printed in full above.
sol_backup=/var/backups/sol-execbench-sudoers/stable-link-20261027
sudo install -d -o root -g root -m 0700 "$sol_backup" &&
sudo test ! -e "$sol_backup/sol-execbench-amd-smi" &&
sudo test ! -L "$sol_backup/sol-execbench-amd-smi" &&
sudo mv "$sol_dest" "$sol_backup/sol-execbench-amd-smi" &&
sudo visudo -c
```

If installation failed before rename, inspect the staging file and preserve it
outside `/etc/sudoers.d` instead. Do not replace an existing backup. No previous
policy file was overwritten, so rollback does not reconstruct old permissions.
