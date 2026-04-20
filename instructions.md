# Setup Instructions

After cloning the repo or unzipping the download, each session's Julia environment needs a `Manifest.toml`. You do not generate this file manually — each session's `Include.jl` does it automatically on first run.

From `lectures/session-1/Include.jl`:

```julia
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    using Pkg;
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end
```

## Option 1: Let the notebook do it

1. `cd lectures/session-1` (or whichever session you want to run)
2. Launch Jupyter and open the notebook
3. Run the first cell (`include("Include.jl")`)

The first run resolves and instantiates the environment, writes `Manifest.toml`, and installs all dependencies. Subsequent runs skip this step. Same pattern in sessions 2–4.

## Option 2: Instantiate from the shell first

```bash
cd lectures/session-1
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Repeat for `session-2`, `session-3`, `session-4` as needed.

## Caveat

The local package `code/` is a path dependency (`../../code` relative to each session). The repo layout must be intact — this will not work if you copy a single session folder out of the repo.
