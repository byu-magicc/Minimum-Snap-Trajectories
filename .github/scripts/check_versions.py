"""Check that the version numbers in the repository agree with each other.

Always:          pyproject.toml `version` == CITATION.cff `version`.
On a release:    (TAG set, e.g. by the Wheels workflow from the GitHub release)
                 TAG == "v" + version, CHANGELOG.md has a "## [version] - YYYY-MM-DD" section,
                 and that date == CITATION.cff `date-released`.

Run from the repository root:  python .github/scripts/check_versions.py
Exits non-zero, listing every mismatch, so a release cannot publish inconsistent metadata.
"""

import os
import re
import sys
import tomllib


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    errors = []

    with open("pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]

    citation = read("CITATION.cff")
    # top-level keys only (no indentation), so preferred-citation fields are ignored
    m = re.search(r'^version:\s*"?([^"\s]+)"?\s*$', citation, re.M)
    cff_version = m.group(1) if m else None
    m = re.search(r'^date-released:\s*"?([0-9-]+)"?\s*$', citation, re.M)
    cff_date = m.group(1) if m else None

    print(f"pyproject.toml version : {version}")
    print(f"CITATION.cff version   : {cff_version}")
    if cff_version != version:
        errors.append(f"CITATION.cff version ({cff_version}) != pyproject.toml version ({version})")

    tag = os.environ.get("TAG", "").strip()
    if tag:
        print(f"release tag            : {tag}")
        if tag != f"v{version}":
            errors.append(f"release tag ({tag}) != 'v' + pyproject.toml version (v{version})")

        m = re.search(r"^## \[" + re.escape(version) + r"\] - (\d{4}-\d{2}-\d{2})\s*$",
                      read("CHANGELOG.md"), re.M)
        changelog_date = m.group(1) if m else None
        print(f"CHANGELOG.md section   : {'## [' + version + '] - ' + changelog_date if m else 'missing'}")
        print(f"CITATION.cff released  : {cff_date}")
        if not m:
            errors.append(f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' section")
        elif cff_date != changelog_date:
            errors.append(f"CITATION.cff date-released ({cff_date}) != CHANGELOG.md date ({changelog_date})")

    if errors:
        print("\nVersion mismatch:")
        for e in errors:
            print(f"  - {e}")
        print("\nSee RELEASING.md: the version and date must match in all of these places.")
        sys.exit(1)
    print("\nAll version numbers agree.")


if __name__ == "__main__":
    main()
