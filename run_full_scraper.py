#!/usr/bin/env python3
"""
Run the full Cultur Space scraper with embeddings
"""

import os
import sys
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main entry point for full scraper"""
    print("Starting Cultur Space Full Scraper with Embeddings")
    print("=" * 50)

    # Supabase: use env vars (e.g. GitHub Actions secrets) or fallback to defaults
    SUPABASE_URL = os.environ.get(
        "SUPABASE_URL", "https://yqawmzggcgpeyaaynrjk.supabase.co"
    )
    SUPABASE_KEY = os.environ.get(
        "SUPABASE_KEY",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4",
    )

    try:
        from culturspace_scraper import CulturSpaceScraper

        # Initialize and run scraper
        scraper = CulturSpaceScraper(SUPABASE_URL, SUPABASE_KEY)
        scraper.run()

        print("\nScraping completed successfully!")
        print("Check scraper.log for detailed logs")
        print("Check Supabase database for the scraped products")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running scraper: {e}")
        print("Check scraper.log for more details")
        sys.exit(1)

if __name__ == "__main__":
    main()