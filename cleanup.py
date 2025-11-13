#!/usr/bin/env python3
"""
Cleanup script for Financial Analyst Advisor.

This script cleans up temporary files, caches, and development artifacts.
"""

import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_python_cache():
    """Remove Python cache files and directories."""
    logger.info("Cleaning Python cache...")
    
    # Find and remove __pycache__ directories
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                cache_path = Path(root) / dir_name
                logger.info(f"Removing: {cache_path}")
                shutil.rmtree(cache_path)
    
    # Remove .pyc files
    for root, dirs, files in os.walk("."):
        for file_name in files:
            if file_name.endswith(('.pyc', '.pyo', '.pyd')):
                file_path = Path(root) / file_name
                logger.info(f"Removing: {file_path}")
                file_path.unlink()


def clean_logs():
    """Remove log files."""
    logger.info("Cleaning log files...")
    
    log_patterns = ["*.log", "logs/", ".wandb/", "wandb/"]
    
    for pattern in log_patterns:
        for path in Path(".").glob(pattern):
            if path.exists():
                if path.is_file():
                    logger.info(f"Removing log file: {path}")
                    path.unlink()
                elif path.is_dir():
                    logger.info(f"Removing log directory: {path}")
                    shutil.rmtree(path)


def clean_temp_files():
    """Remove temporary files."""
    logger.info("Cleaning temporary files...")
    
    temp_patterns = [
        "tmp/", "temp/", "cache/",
        "*.tmp", "*.temp",
        ".DS_Store", "Thumbs.db",
        "*.swp", "*.swo", "*~"
    ]
    
    for pattern in temp_patterns:
        for path in Path(".").rglob(pattern):
            if path.exists():
                try:
                    if path.is_file():
                        logger.info(f"Removing temp file: {path}")
                        path.unlink()
                    elif path.is_dir():
                        logger.info(f"Removing temp directory: {path}")
                        shutil.rmtree(path)
                except Exception as e:
                    logger.warning(f"Could not remove {path}: {e}")


def clean_test_artifacts():
    """Remove test artifacts."""
    logger.info("Cleaning test artifacts...")
    
    test_patterns = [
        ".pytest_cache/", ".coverage", "htmlcov/", 
        ".tox/", ".hypothesis/", "coverage.xml"
    ]
    
    for pattern in test_patterns:
        for path in Path(".").glob(pattern):
            if path.exists():
                if path.is_file():
                    logger.info(f"Removing test file: {path}")
                    path.unlink()
                elif path.is_dir():
                    logger.info(f"Removing test directory: {path}")
                    shutil.rmtree(path)


def main():
    """Run cleanup operations."""
    logger.info("Starting cleanup...")
    
    try:
        clean_python_cache()
        clean_logs()
        clean_temp_files()
        clean_test_artifacts()
        
        logger.info("✅ Cleanup completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")


if __name__ == "__main__":
    main()