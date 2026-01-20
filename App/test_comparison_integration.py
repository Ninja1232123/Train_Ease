#!/usr/bin/env python3
"""
Quick integration test for Model Comparison feature.
Tests that routes are registered and basic functionality works.
"""

import logging
import sys

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_comparison")

def test_manager_import():
    """Test that ComparisonManager can be imported."""
    try:
        from comparison_manager import ComparisonManager
        logger.info("✓ ComparisonManager imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import ComparisonManager: {e}")
        return False

def test_routes_import():
    """Test that comparison routes can be imported."""
    try:
        from comparison_routes import add_comparison_routes
        logger.info("✓ comparison_routes imported successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to import comparison_routes: {e}")
        return False

def test_manager_initialization():
    """Test that ComparisonManager can be initialized."""
    try:
        from comparison_manager import ComparisonManager
        manager = ComparisonManager(logger)
        logger.info("✓ ComparisonManager initialized successfully")

        # Test status method
        status = manager.get_comparison_status()
        assert status['model_a_loaded'] == False
        assert status['model_b_loaded'] == False
        logger.info("✓ get_comparison_status() works correctly")

        # Cleanup
        manager.cleanup()
        logger.info("✓ cleanup() works correctly")

        return True
    except Exception as e:
        logger.error(f"✗ Failed to initialize ComparisonManager: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_integration():
    """Test that the comparison routes integrate with Flask app."""
    try:
        # Import Flask app components
        from flask import Flask
        from comparison_manager import ComparisonManager
        from comparison_routes import add_comparison_routes

        # Create minimal Flask app
        app = Flask(__name__)
        manager = ComparisonManager(logger)

        # Register routes
        add_comparison_routes(app, manager, logger)

        # Check that routes were registered
        routes = [str(rule) for rule in app.url_map.iter_rules()]

        expected_routes = [
            '/api/comparison/load-model-a',
            '/api/comparison/load-model-b',
            '/api/comparison/generate',
            '/api/comparison/status',
            '/api/comparison/unload',
            '/api/comparison/unload-model-a',
            '/api/comparison/unload-model-b',
            '/api/comparison/save'
        ]

        for route in expected_routes:
            if route in routes:
                logger.info(f"✓ Route registered: {route}")
            else:
                logger.error(f"✗ Route NOT registered: {route}")
                return False

        # Cleanup
        manager.cleanup()

        logger.info("✓ All routes registered successfully")
        return True

    except Exception as e:
        logger.error(f"✗ App integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Model Comparison Integration Test")
    print("=" * 60)
    print()

    tests = [
        ("Import ComparisonManager", test_manager_import),
        ("Import comparison_routes", test_routes_import),
        ("Initialize ComparisonManager", test_manager_initialization),
        ("Flask App Integration", test_app_integration),
    ]

    results = []
    for name, test_func in tests:
        print(f"Running: {name}")
        result = test_func()
        results.append((name, result))
        print()

    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Model Comparison MVP is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
