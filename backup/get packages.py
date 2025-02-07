#!/usr/bin/env python3
"""
Script to analyze all imports in a Python project and compare with conda environment
"""

import os
import ast
import sys
from pathlib import Path
from collections import defaultdict, Counter
import subprocess
import pkg_resources
import importlib.util

def find_python_files(directory):
    """Find all Python files in directory recursively"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip common directories that don't contain user code
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def extract_imports_from_file(file_path):
    """Extract all imports from a Python file"""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError as e:
                print(f"Syntax error in {file_path}: {e}")
                return imports
                
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])  # Get top-level module
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])  # Get top-level module
                    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return imports

def get_conda_packages():
    """Get list of packages in current conda environment"""
    try:
        result = subprocess.run(['conda', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            packages = {}
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    version = parts[1]
                    packages[name] = version
            return packages
        else:
            print("Failed to run 'conda list'")
            return {}
    except FileNotFoundError:
        print("Conda not found. Make sure conda is installed and in PATH.")
        return {}

def get_pip_packages():
    """Get list of packages installed via pip"""
    try:
        import pkg_resources
        packages = {}
        for dist in pkg_resources.working_set:
            packages[dist.project_name.lower()] = dist.version
        return packages
    except Exception as e:
        print(f"Error getting pip packages: {e}")
        return {}

def categorize_imports(imports):
    """Categorize imports into standard library, third-party, and local"""
    stdlib_modules = set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set()
    
    # Common standard library modules for older Python versions
    common_stdlib = {
        'os', 'sys', 'json', 'csv', 'sqlite3', 'datetime', 'time', 'random',
        'collections', 'functools', 'itertools', 'typing', 'pathlib', 're',
        'asyncio', 'subprocess', 'logging', 'tempfile', 'shutil', 'hashlib',
        'contextlib', 'warnings'
    }
    stdlib_modules.update(common_stdlib)
    
    categorized = {
        'standard_library': set(),
        'third_party': set(),
        'local': set(),
        'unknown': set()
    }
    
    for imp in imports:
        if imp in stdlib_modules:
            categorized['standard_library'].add(imp)
        elif imp.startswith('.') or imp in ['models', 'config', 'prompts', 'utils', 'cache_manager']:
            categorized['local'].add(imp)
        else:
            # Try to check if it's importable
            try:
                spec = importlib.util.find_spec(imp)
                if spec is not None:
                    categorized['third_party'].add(imp)
                else:
                    categorized['unknown'].add(imp)
            except (ImportError, ModuleNotFoundError, ValueError):
                categorized['unknown'].add(imp)
    
    return categorized

def map_import_to_package(import_name):
    """Map import names to their conda/pip package names"""
    # Common mappings where import name != package name
    mapping = {
        'cv2': 'opencv',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn',
        'yaml': 'pyyaml',
        'bs4': 'beautifulsoup4',
        'requests': 'requests',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'scipy': 'scipy',
        'spacy': 'spacy',
        'openai': 'openai',
        'tiktoken': 'tiktoken',
        'instructor': 'instructor',
        'pydantic': 'pydantic',
        'streamlit': 'streamlit',
        'pyreadstat': 'pyreadstat',
        'nest_asyncio': 'nest-asyncio',
        'langchain_core': 'langchain-core',
        'langchain_openai': 'langchain-openai',
        'umap': 'umap-learn',
        'hdbscan': 'hdbscan',
        'tqdm': 'tqdm'
    }
    return mapping.get(import_name, import_name)

def analyze_project_dependencies(project_path):
    """Main function to analyze project dependencies"""
    print(f"Analyzing dependencies in: {project_path}")
    print("=" * 50)
    
    # Find all Python files
    python_files = find_python_files(project_path)
    print(f"Found {len(python_files)} Python files")
    
    # Extract all imports
    all_imports = set()
    imports_by_file = {}
    
    for file_path in python_files:
        file_imports = extract_imports_from_file(file_path)
        all_imports.update(file_imports)
        if file_imports:
            rel_path = os.path.relpath(file_path, project_path)
            imports_by_file[rel_path] = file_imports
    
    print(f"Total unique imports found: {len(all_imports)}")
    
    # Categorize imports
    categorized = categorize_imports(all_imports)
    
    print("\n" + "=" * 50)
    print("CATEGORIZED IMPORTS:")
    print("=" * 50)
    
    for category, imports in categorized.items():
        if imports:
            print(f"\n{category.upper().replace('_', ' ')} ({len(imports)}):")
            for imp in sorted(imports):
                print(f"  - {imp}")
    
    # Get current environment packages
    print("\n" + "=" * 50)
    print("CURRENT ENVIRONMENT ANALYSIS:")
    print("=" * 50)
    
    conda_packages = get_conda_packages()
    pip_packages = get_pip_packages()
    
    print(f"Conda packages: {len(conda_packages)}")
    print(f"Pip packages: {len(pip_packages)}")
    
    # Check which third-party imports are installed
    print("\n" + "=" * 30)
    print("THIRD-PARTY PACKAGE STATUS:")
    print("=" * 30)
    
    missing_packages = []
    installed_packages = []
    
    for imp in sorted(categorized['third_party']):
        package_name = map_import_to_package(imp)
        
        # Check in conda packages (exact match and fuzzy)
        conda_found = package_name in conda_packages
        conda_fuzzy = any(package_name in pkg for pkg in conda_packages.keys())
        
        # Check in pip packages
        pip_found = package_name.lower() in pip_packages
        pip_fuzzy = any(package_name.lower() in pkg.lower() for pkg in pip_packages.keys())
        
        if conda_found:
            version = conda_packages[package_name]
            print(f"✓ {imp} -> {package_name} (conda: {version})")
            installed_packages.append((imp, package_name, 'conda', version))
        elif pip_found:
            version = pip_packages[package_name.lower()]
            print(f"✓ {imp} -> {package_name} (pip: {version})")
            installed_packages.append((imp, package_name, 'pip', version))
        elif conda_fuzzy or pip_fuzzy:
            fuzzy_matches = [pkg for pkg in conda_packages.keys() if package_name in pkg]
            fuzzy_matches.extend([pkg for pkg in pip_packages.keys() if package_name.lower() in pkg.lower()])
            print(f"? {imp} -> {package_name} (possible matches: {fuzzy_matches})")
        else:
            print(f"✗ {imp} -> {package_name} (NOT FOUND)")
            missing_packages.append((imp, package_name))
    
    # Summary
    print("\n" + "=" * 30)
    print("SUMMARY:")
    print("=" * 30)
    print(f"Total third-party imports: {len(categorized['third_party'])}")
    print(f"Installed packages: {len(installed_packages)}")
    print(f"Missing packages: {len(missing_packages)}")
    
    if missing_packages:
        print("\nMISSING PACKAGES:")
        for imp, pkg in missing_packages:
            print(f"  - {pkg} (for import {imp})")
    
    # Generate requirements
    print("\n" + "=" * 30)
    print("REQUIREMENTS EXPORT:")
    print("=" * 30)
    
    print("\nConda packages in use:")
    conda_reqs = []
    for imp, pkg, source, version in installed_packages:
        if source == 'conda':
            conda_reqs.append(f"  - {pkg}={version}")
    
    if conda_reqs:
        for req in sorted(set(conda_reqs)):
            print(req)
    
    print("\nPip packages in use:")
    pip_reqs = []
    for imp, pkg, source, version in installed_packages:
        if source == 'pip':
            pip_reqs.append(f"    - {pkg}=={version}")
    
    if pip_reqs:
        for req in sorted(set(pip_reqs)):
            print(req)
    
    return {
        'all_imports': all_imports,
        'categorized': categorized,
        'installed_packages': installed_packages,
        'missing_packages': missing_packages,
        'conda_packages': conda_packages,
        'pip_packages': pip_packages
    }

if __name__ == "__main__":
    # Use current directory or provide path as argument
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."
    analyze_project_dependencies(project_path)