# JLD2 `EOFError` in `S1-Example-Core-BuildMinVar-RA` — Root Cause and Fix

## Symptom

Running the `S1-Example-Core-BuildMinVariancePortfolio-RA` notebook fails in a cell that calls `load_results(...)` (`code/src/Files.jl:151`) with:

```
EOFError: read end of file
[...]
[12] load_results(path::String)
    @ eCornellAIFinance ~/Downloads/eCornell-AI-finance-lectures-main/code/src/Files.jl:151
```

## Root Cause

All `.jld2` files across the repo are **Git LFS pointer stubs** (~130 bytes each), not the real HDF5 binaries.

Confirmed by inspecting one of the files:

```
version https://git-lfs.github.com/spec/v1
oid sha256:3931097e32b2369de066651ad91dd4a4075e24947a36f4f438577d66c14b8227
size 9277
```

`file` reports every `.jld2` as `ASCII text` (a real JLD2 file would be binary HDF5). Because the pointer stub has no HDF5 payload, JLD2 reads past the end of the file while parsing the header → `EOFError`.

Scope of the issue:
- `lectures/session-1/data/*.jld2` — all 6 files are LFS pointers (129–133 bytes).
- `code/src/data/*.jld2` — all 7 files are LFS pointers (130–133 bytes). `CLAUDE.md` states this directory should be ~270 MB; it is currently ~200 KB (only the `.png` figures have real content).

Since every `.jld2` is affected, you would hit the same error on every other example notebook too — not just `BuildMinVar-RA`.

## Why This Happened

The repo was downloaded as a ZIP from GitHub (the `-main` suffix on the directory name is the giveaway — that's GitHub's default ZIP export naming). **GitHub's "Download ZIP" button does not resolve Git LFS objects** — it bundles the pointer files verbatim.

## Fix

Re-fetch the repo via a real `git clone` with LFS enabled:

```bash
# 1. Install git-lfs if you don't have it
brew install git-lfs
git lfs install

# 2. Clone fresh (replace URL with the actual repo URL you downloaded from)
cd ~/Downloads
git clone <repo-url> eCornell-AI-finance-lectures
cd eCornell-AI-finance-lectures
git lfs pull    # usually automatic, but run explicitly to be sure
```

Then work from the cloned directory instead of the `-main` ZIP extract.

## Verification

After cloning, sanity-check with:

```bash
ls -lh code/src/data/
ls -lh lectures/session-1/data/
```

The `.jld2` files should be multi-MB (not 130 B). `code/src/data/` as a whole should be ~270 MB per `CLAUDE.md`.
