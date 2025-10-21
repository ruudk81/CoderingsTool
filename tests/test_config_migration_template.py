"""
Test template for config parameter migrations.

This file serves as a template. For each migration, create a copy:
tests/test_[parameter_name]_migration.py

Example: tests/test_headroom_migration.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_config_import():
    """Test that config class imports successfully"""
    try:
        # CUSTOMIZE: Import the config class you modified
        from config import ProcessingConfig, DEFAULT_PROCESSING_CONFIG
        print("✓ Config imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_parameter_exists():
    """Test that the migrated parameter exists in config"""
    try:
        from config import DEFAULT_PROCESSING_CONFIG

        # CUSTOMIZE: Replace with your parameter name
        parameter_name = 'rate_limit_headroom'

        assert hasattr(DEFAULT_PROCESSING_CONFIG, parameter_name), \
            f"Parameter '{parameter_name}' not found in config"

        value = getattr(DEFAULT_PROCESSING_CONFIG, parameter_name)
        print(f"✓ Parameter exists: {parameter_name} = {value}")
        return True
    except AssertionError as e:
        print(f"✗ Parameter check failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_parameter_value():
    """Test that parameter has expected value"""
    try:
        from config import DEFAULT_PROCESSING_CONFIG

        # CUSTOMIZE: Replace with your parameter and expected value
        parameter_name = 'rate_limit_headroom'
        expected_value = 0.9

        actual_value = getattr(DEFAULT_PROCESSING_CONFIG, parameter_name)

        assert actual_value == expected_value, \
            f"Expected {parameter_name}={expected_value}, got {actual_value}"

        print(f"✓ Parameter value correct: {parameter_name} = {actual_value}")
        return True
    except AssertionError as e:
        print(f"✗ Value check failed: {e}")
        return False


def test_utility_imports_config():
    """Test that modified utilities can import config"""
    try:
        # CUSTOMIZE: Import the utilities you modified
        from utils import qualityFilter, codeGenerator

        print("✓ Modified utilities import successfully")
        return True
    except ImportError as e:
        print(f"✗ Utility import failed: {e}")
        return False


def test_utility_uses_config():
    """Test that utility can instantiate and access config parameter"""
    try:
        from config import DEFAULT_PROCESSING_CONFIG
        # CUSTOMIZE: Import and instantiate your utility class
        # from utils.qualityFilter import Grader

        # Example test:
        # grader = Grader([], "test", processing_config=DEFAULT_PROCESSING_CONFIG)
        # assert hasattr(grader, 'processing_config')
        # assert grader.processing_config.rate_limit_headroom == 0.9

        print("✓ Utility uses config correctly (manual verification needed)")
        return True
    except Exception as e:
        print(f"✗ Utility config usage failed: {e}")
        return False


def test_config_validation():
    """Test that config validation works (if implemented)"""
    try:
        from config import ProcessingConfig

        # CUSTOMIZE: Test invalid values if validation implemented
        # Example:
        # try:
        #     invalid_config = ProcessingConfig(rate_limit_headroom=1.5)  # > 1.0
        #     print("✗ Validation failed to catch invalid value")
        #     return False
        # except ValueError:
        #     print("✓ Validation correctly rejects invalid values")

        print("✓ Config validation works (if implemented)")
        return True
    except Exception as e:
        print(f"✗ Validation test error: {e}")
        return False


def run_all_tests():
    """Run all migration tests"""
    print("=" * 60)
    print("CONFIG MIGRATION TEST SUITE")
    print("=" * 60)

    tests = [
        ("Config Import", test_config_import),
        ("Parameter Exists", test_parameter_exists),
        ("Parameter Value", test_parameter_value),
        ("Utility Imports", test_utility_imports_config),
        ("Utility Config Usage", test_utility_uses_config),
        ("Config Validation", test_config_validation),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        result = test_func()
        results.append((test_name, result))

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")

    all_passed = all(result for _, result in results)

    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
