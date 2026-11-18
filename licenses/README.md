# Third-party licenses

COAST itself is licensed under the BSD 3-Clause License ([`../LICENSE`](../LICENSE)).

The Python wheels on PyPI also contain compiled code from the projects below, so their license
texts and notices are included here and in each wheel's `.dist-info/licenses/` directory.
The source distribution contains COAST's own source plus the vendored nanoflann header; the
other projects are downloaded when it is built.

| Component | Version | License | Files | Source |
|---|---|---|---|---|
| OSQP (QP solver) | 1.0.0 | Apache-2.0 | `OSQP-LICENSE`, `OSQP-NOTICE` | <https://github.com/osqp/osqp> |
| QDLDL (linear-system solver used by OSQP) | 0.1.8 | Apache-2.0 | `QDLDL-LICENSE` | <https://github.com/osqp/qdldl> |
| AMD (SuiteSparse ordering, bundled with OSQP) | as bundled with OSQP 1.0.0 | BSD-3-Clause | `AMD-LICENSE` | <https://github.com/DrTimothyAldenDavis/SuiteSparse> |
| Eigen (header-only linear algebra) | 3.4.0 | MPL-2.0; some files BSD-3-Clause or Apache-2.0 | `EIGEN-COPYING.*` | <https://gitlab.com/libeigen/eigen> |
| pybind11 (Python bindings, header-only) | 2.12 or newer (3.x for current builds) | BSD-3-Clause | `PYBIND11-LICENSE` | <https://github.com/pybind/pybind11> |
| nanoflann (k-d tree, vendored as `cpp/deps/nanoflann.hpp`) | as vendored | BSD-2-Clause | `NANOFLANN-LICENSE` | <https://github.com/jlblancoc/nanoflann> |
| Microsoft Visual C++ runtime `msvcp140.dll` (copied into `coast_planner.libs/` of the Windows wheels only) | from the build image | Microsoft Visual Studio redistributable terms | none | <https://visualstudio.microsoft.com/license-terms/> |

**Eigen.** The Eigen code in the wheels is unmodified; its source is available at the address
above, as the Mozilla Public License 2.0 requires for distribution in executable form. COAST uses
only Eigen's `Dense` and `Sparse` modules, which are MPL-2.0 (with a few BSD-3-Clause and
Apache-2.0 files); in Eigen 3.4.0 the LGPL-licensed code is confined to `unsupported/` modules
that COAST does not use, so Eigen's GPL, LGPL and MINPACK license files are not needed. COAST is
also compiled with `EIGEN_MPL2_ONLY`, which makes Eigen raise a compile error if any of the code
it marks as non-MPL2 is ever included.

**C++ library users.** Building the C++ library from source and linking it into your own program
compiles in the same OSQP, QDLDL, AMD, Eigen and nanoflann code, so ship these notices with
binaries you distribute. pybind11 and the MSVC runtime apply only to the Python wheels.

When updating a component's version in `cpp/cmake/MinSnapDeps.cmake` (or the vendored nanoflann
header), update its row here and replace its license file if the text changed.
