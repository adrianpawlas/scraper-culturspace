#!/usr/bin/env python3
"""
Test script for Cultur Space scraper
Tests with a few products to ensure everything works correctly.
"""

import sys
import logging
from culturspace_scraper import CulturSpaceScraper

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_scraper():
    """Test the scraper with a few products"""
    # Supabase configuration
    SUPABASE_URL = "https://yqawmzggcgpeyaaynrjk.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4"

    # Test URLs from the user's examples
    test_urls = [
        "https://culturspace.com/collections/shop-all/products/istanbul-track-jacket-navy",
        "https://culturspace.com/collections/shop-all/products/istanbul-track-jacket-white",
        "https://culturspace.com/collections/shop-all/products/reverse-pleat-shorts"
    ]

    logger = logging.getLogger(__name__)
    logger.info("Starting Cultur Space scraper test")

    scraper = CulturSpaceScraper(SUPABASE_URL, SUPABASE_KEY)

    try:
        successful_tests = 0

        for i, url in enumerate(test_urls, 1):
            logger.info(f"Testing product {i}/{len(test_urls)}: {url}")

            try:
                product_data = scraper.scrape_product_page(url)
                if product_data:
                    logger.info(f"[OK] Successfully scraped: {product_data.title}")
                    logger.info(f"  - Price: {product_data.price}")
                    logger.info(f"  - Sale: {product_data.sale}")
                    logger.info(f"  - Image URL: {product_data.image_url}")
                    logger.info(f"  - Image embedding dimension: {len(product_data.image_embedding) if product_data.image_embedding else 0}")
                    logger.info(f"  - Info embedding dimension: {len(product_data.info_embedding) if product_data.info_embedding else 0}")

                    # Test saving to Supabase
                    if scraper.save_to_supabase(product_data):
                        logger.info("  [OK] Successfully saved to Supabase")
                        successful_tests += 1
                    else:
                        logger.error("  [FAIL] Failed to save to Supabase")
                else:
                    logger.error("  [FAIL] Failed to scrape product data")

            except Exception as e:
                logger.error(f"  [FAIL] Error testing {url}: {e}")

        logger.info(f"Test completed. {successful_tests}/{len(test_urls)} products processed successfully")

        if successful_tests == len(test_urls):
            logger.info("All tests passed! Ready for full scrape.")
            return True
        else:
            logger.warning("%d tests failed. Please check the logs.", len(test_urls) - successful_tests)
            return False

    finally:
        scraper._cleanup_selenium()

if __name__ == "__main__":
    success = test_scraper()
    sys.exit(0 if success else 1)