# =============================================================================
# INSTALL PACKAGES FOR RENV
# =============================================================================
# Run this once after renv::init() to install all project dependencies.
# These will be installed as macOS/Darwin-native packages in the project's
# renv library folder.
#
# Usage:
#   1. Start R from this project folder
#   2. Run: renv::init()
#   3. Restart R
#   4. Run: source("install_packages.R")
#   5. Run: renv::snapshot()

packages <- c(
  # Data import
  "foreign",
  
 # Data manipulation
  "dplyr",
  "stringr",
  
  # Visualization
  "ggplot2",
  "ggrepel",
  "scales",
  "showtext",
  "treemapify",
  
  # Excel export
  "openxlsx",
  
  # Statistical analysis
  "poLCA",
  
  # Infrastructure (used by VS Code R extension and other packages)
  "jsonlite",
  "rlang"
)

cat("Installing", length(packages), "packages for macOS...\n\n")

# Install all packages
install.packages(packages)

# Verify installation
cat("\n=== Verification ===\n")
for (pkg in packages) {
  if (pkg %in% installed.packages()[, "Package"]) {
    cat("✓", pkg, "\n")
  } else {
    cat("✗", pkg, "- FAILED\n")
  }
}

cat("\nDone! Now run renv::snapshot() to lock these versions.\n")
