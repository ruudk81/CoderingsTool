# R Project Environment Setup with renv

## Overview

This project uses `renv` for package management — similar to Python's `venv` + `requirements.txt`. Each project has its own isolated library, making it reproducible across machines.

## Quick Reference

| Python | R |
|--------|---|
| `requirements.txt` | `renv.lock` |
| `.venv/` | `renv/library/` |
| `pip install -r requirements.txt` | `renv::restore()` |
| `pip freeze > requirements.txt` | `renv::snapshot()` |

## Files

| File | Purpose | Git? |
|------|---------|------|
| `renv.lock` | Exact package versions (the lockfile) | ✅ Yes |
| `setup.R` | Generic setup script — works on any machine | ✅ Yes |
| `install_packages.R` | Project-specific package list (optional after lockfile exists) | ✅ Yes |
| `renv/` | Project library folder | ❌ No (gitignored) |
| `.Rprofile` | Auto-activates renv on R startup | ✅ Yes |

## Usage

### On a new machine (project already set up)

```r
# Open R in the project folder, then:
source("setup.R")
```

That's it. The script handles everything.

### First-time setup for a new project

1. **Create `install_packages.R`** with the packages your project needs:

```r
packages <- c(
  "dplyr",
  "ggplot2",
  # add your packages here
)

install.packages(packages)
```

2. **Initialize renv:**

```r
install.packages("renv")
renv::init()
```

3. **Restart R** (quit and reopen in the same folder)

4. **Install your packages:**

```r
source("install_packages.R")
```

5. **Create the lockfile:**

```r
renv::snapshot()
```

6. **Copy `setup.R`** from another project (it's generic)

7. **Commit to git:** `renv.lock`, `setup.R`, `install_packages.R`, `.Rprofile`

## Common Commands

| Command | What it does |
|---------|--------------|
| `renv::status()` | Check if packages are in sync |
| `renv::snapshot()` | Update lockfile with current packages |
| `renv::restore()` | Install packages from lockfile |
| `renv::install("package")` | Install a new package |
| `.libPaths()` | Check where packages are installed |

## How It Works

1. When R starts in a project folder, `.Rprofile` runs automatically
2. `.Rprofile` activates renv, which redirects `.libPaths()` to the project's `renv/library/`
3. Packages are installed per-project, not globally
4. `renv.lock` records exact versions for reproducibility

## Adding a New Package

```r
install.packages("newpackage")
renv::snapshot()  # update the lockfile
```

Then commit the updated `renv.lock`.

## Troubleshooting

**renv not activating on startup?**
- Make sure you're starting R from the project folder
- Check that `.Rprofile` exists and contains renv activation code

**Packages installing to global library?**
- Run `renv::activate()` then restart R
- Check `.libPaths()` — first path should be inside your project

**"Package not found" errors?**
- Run `renv::status()` to see what's missing
- Run `renv::restore()` to install from lockfile

## Platform Notes

- renv installs platform-specific binaries (macOS packages won't work on Windows)
- The lockfile is cross-platform — `renv::restore()` fetches correct binaries for each OS
- RDCOMClient (Windows COM automation) only works on Windows
