#!/usr/bin/env python3
"""
Quick test script for Cultur Space scraper - without embeddings
"""

import requests
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_test():
    """Quick test without heavy ML models"""
    logger.info("Starting quick test...")

    # Test basic URL access
    url = "https://culturspace.com/collections/shop-all/products/istanbul-track-jacket-navy"
    logger.info(f"Testing URL: {url}")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        logger.info("✓ URL accessible")

        # Parse HTML
        soup = BeautifulSoup(response.content, 'lxml')

        # Test title extraction
        title_elem = soup.find('h1')
        if title_elem:
            title = title_elem.get_text(strip=True)
            logger.info(f"✓ Title found: {title}")
        else:
            logger.warning("✗ Title not found")

        # Test image extraction
        img_elem = soup.find('img', class_=re.compile(r'product.*image'))
        if img_elem:
            src = img_elem.get('src') or img_elem.get('data-src')
            if src:
                logger.info(f"✓ Image found: {src[:50]}...")
            else:
                logger.warning("✗ Image src not found")
        else:
            logger.warning("✗ Image element not found")

        # Test price extraction
        price_text = soup.find(text=re.compile(r'€\d+'))
        if price_text:
            logger.info(f"✓ Price pattern found: {price_text.strip()}")
        else:
            logger.warning("✗ Price pattern not found")

        logger.info("Quick test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        return False

if __name__ == "__main__":
    import re
    success = quick_test()
    print(f"Test result: {'PASS' if success else 'FAIL'}")