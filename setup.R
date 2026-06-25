# =============================================================================
# PROJECT SETUP SCRIPT
# =============================================================================
# Drop this file into any R project folder and run it to set up the environment.
# Works across machines (Windows, macOS, Linux).
#
# Usage: Open R in the project folder, then run:
#   source("setup.R")
#
# What it does:
#   1. Installs renv if not available
#   2. Initializes renv if not already set up
#   3. Restores packages from lockfile if they're missing
# =============================================================================

cat("=== R Project Setup ===\n\n")

# Step 1: Check if renv is available
if (!requireNamespace("renv", quietly = TRUE)) {
  cat("Installing renv...\n")
  install.packages("renv", repos = "https://cloud.r-project.org")
}

# Step 2: Check if project has renv initialized
if (!file.exists("renv.lock")) {
  cat("No renv.lock found. Initializing renv for this project...\n")
  renv::init()
  cat("\nrenv initialized. Add your packages, then run renv::snapshot()\n")
  cat("Restart R to activate the project environment.\n")
} else {
  # Step 3: Activate if not already active
  if (!renv:::renv_project_initialized(getwd())) {
    cat("Activating renv...\n")
    renv::activate()
  }
  
  # Step 4: Check if packages need to be restored
  status <- renv::status()

  # Check if there are any synchronization issues
  needs_restore <- !isTRUE(attr(status, "synchronized"))

  if (needs_restore) {
    cat("\nSome packages may need to be installed. Running renv::restore()...\n\n")
    renv::restore()
  } else {
    cat("All packages are installed and in sync.\n")
  }
}

cat("\n=== Setup complete ===\n")
cat("Library path:", .libPaths()[1], "\n")
