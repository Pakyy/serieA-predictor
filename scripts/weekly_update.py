#!/usr/bin/env python3
"""
Weekly data update script
Runs via GitHub Actions every Monday
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Run the pipeline with scraping enabled
from data.pipeline_complete import main

if __name__ == "__main__":
    print("🔄 Starting weekly update...")
    
    # Run pipeline: scrape + process + train + upload
    main(skip_scrape=False, skip_upload=False)
    
    print("✅ Weekly update complete!")