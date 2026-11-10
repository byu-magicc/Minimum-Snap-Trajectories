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
- **PyPI:** once the wheel-building workflow and trusted publishing exist (#17), pushing the
  tag builds and uploads the wheels. Check the release appears on PyPI.
- Announce the release where relevant, and close or update the issues it resolves.
