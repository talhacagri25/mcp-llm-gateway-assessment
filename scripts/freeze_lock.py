"""Write the exact dependency closure of this repository to stdout (requirements.lock).

Walks the installed metadata from the direct dependencies in requirements.txt,
so the lock reflects what the verification environment actually resolved,
transitive packages included, and nothing unrelated that happens to be installed.
"""

from __future__ import annotations

import sys
from importlib import metadata

from packaging.requirements import Requirement

HEADER = """\
# Exact versions of every runtime and development dependency, transitive ones
# included, as installed in the environment used to verify this repository
# (Python 3.12, Linux). requirements.txt carries the direct pins and the
# rationale; regenerate this file with `make lock` and audit it with `make audit`.
"""


def direct_requirements(path: str = "requirements.txt") -> list[str]:
    names = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(Requirement(line).name)
    return names


def closure(roots: list[str]) -> dict[str, tuple[str, str]]:
    seen: dict[str, tuple[str, str]] = {}
    stack = list(roots)
    while stack:
        name = stack.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            print(f"warning: {name} is not installed", file=sys.stderr)
            continue
        seen[key] = (dist.metadata["Name"], dist.version)
        for spec in dist.requires or []:
            requirement = Requirement(spec)
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
            stack.append(requirement.name)
    return seen


def main() -> int:
    sys.stdout.write(HEADER)
    for _key, (name, version) in sorted(closure(direct_requirements()).items()):
        sys.stdout.write(f"{name}=={version}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
