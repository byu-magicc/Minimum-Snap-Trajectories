# Releasing

Steps for cutting a release. Versions follow [Semantic Versioning](https://semver.org): while
below 1.0.0, bump the minor version (0.1.0 → 0.2.0) for new features or breaking changes and
the patch version (0.1.0 → 0.1.1) for fixes only.

## 1. Prepare (in a pull request)

1. In [`CHANGELOG.md`](CHANGELOG.md), rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`
   and add a fresh, empty `## [Unreleased]` above it. Update the comparison links at the
   bottom:
   ```
   [Unreleased]: https://github.com/byu-magicc/Minimum-Snap-Trajectories/compare/vX.Y.Z...HEAD
   [X.Y.Z]: https://github.com/byu-magicc/Minimum-Snap-Trajectories/releases/tag/vX.Y.Z
   ```
2. Set the version in **both** places, which must match:
   - `pyproject.toml`: `version = "X.Y.Z"`
   - `CITATION.cff`: `version: X.Y.Z` and `date-released: "YYYY-MM-DD"`
3. Merge the pull request once CI is green.

## 2. Tag and publish

From an up-to-date `main`:

```bash
git switch main && git pull
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

Then create the GitHub release from the tag, using that version's changelog section as the
notes:

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --notes-file <(awk '/^## \[X.Y.Z\]/{f=1;next} /^## \[/||/^\[[^]]*\]: /{f=0} f' CHANGELOG.md)
```

## 3. After the release

- **Zenodo:** once the repository is connected to Zenodo (GitHub integration, one-time setup),
  every GitHub release is archived automatically with a DOI. JOSS requires such an archive DOI
  at acceptance.
- **PyPI:** publishing the GitHub release runs `.github/workflows/wheels.yml`, which builds
  and tests every wheel plus the sdist, then uploads them (requires the one-time setup below).
  Check the release appears at https://pypi.org/project/min-snap/.
- Announce the release where relevant, and close or update the issues it resolves.

## One-time setup: trusted publishing (PyPI and TestPyPI)

Trusted publishing lets the GitHub workflow upload without any stored password or API token.
Do this once per index, logged in to [pypi.org](https://pypi.org) and
[test.pypi.org](https://test.pypi.org) (separate accounts).

1. Account → **Publishing** → **Add a new pending publisher**, with:

   | Field | Value |
   |---|---|
   | PyPI project name | `min-snap` |
   | Owner | `byu-magicc` |
   | Repository name | `Minimum-Snap-Trajectories` |
   | Workflow name | `wheels.yml` |
   | Environment name | `pypi` (on PyPI) / `testpypi` (on TestPyPI) |

   "Pending" because the project does not exist yet; the first upload creates it.
2. On GitHub, **Settings → Environments**: the `pypi` and `testpypi` environments are created
   on first use. For `pypi`, consider adding yourself as a **required reviewer**, so every
   upload waits for a click.
3. Dry run: **Actions → Wheels → Run workflow**, tick *Publish the built distributions to
   TestPyPI*. Then check it installs:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ min-snap
   ```
   (The extra index supplies numpy, which is not on TestPyPI.)

Versions on PyPI and TestPyPI can never be re-uploaded, even after deletion. If a TestPyPI dry
run needs repeating, bump the version (e.g. `0.1.0.dev1`) rather than reusing one.
