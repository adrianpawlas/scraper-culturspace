#!/usr/bin/env python3
"""
Simplified Cultur Space Scraper for testing - without embeddings
"""

import os
import json
import time
import requests
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urljoin
from pathlib import Path

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProductData:
    """Data structure for product information"""
    product_url: str
    image_url: str
    title: str
    description: Optional[str]
    price: str
    sale: Optional[str]
    metadata: Optional[str]

class SimpleCulturSpaceScraper:
    """Simplified scraper class for testing"""

    BASE_URL = "https://culturspace.com"
    SHOP_ALL_URL = "https://culturspace.com/collections/shop-all"
    SOURCE = "scraper"
    BRAND = "Cultur Space"
    GENDER = "MAN"
    SECOND_HAND = False

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        # Initialize Selenium
        self.driver = None

    def _setup_selenium(self):
        """Setup Selenium WebDriver"""
        if self.driver is None:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")

            from selenium.webdriver.chrome.service import Service
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("Selenium WebDriver initialized")

    def _cleanup_selenium(self):
        """Clean up Selenium WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None

    def get_product_urls(self) -> List[str]:
        """Extract all product URLs from the shop pages"""
        product_urls = []
        pages = [self.SHOP_ALL_URL, f"{self.SHOP_ALL_URL}?page=2"]

        for page_url in pages:
            logger.info(f"Scraping product URLs from: {page_url}")
            try:
                response = self.session.get(page_url)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'lxml')

                # Find all product links
                product_links = soup.find_all('a', href=re.compile(r'/collections/shop-all/products/'))

                for link in product_links:
                    href = link.get('href')
                    if href and '/products/' in href:
                        full_url = urljoin(self.BASE_URL, href)
                        if full_url not in product_urls:
                            product_urls.append(full_url)

                logger.info(f"Found {len(product_links)} products on {page_url}")

            except Exception as e:
                logger.error(f"Error scraping {page_url}: {e}")

        logger.info(f"Total unique product URLs found: {len(product_urls)}")
        return product_urls

    def scrape_product_page(self, url: str) -> Optional[ProductData]:
        """Scrape individual product page for details"""
        try:
            logger.info(f"Scraping product: {url}")

            # Use Selenium for JavaScript-heavy content
            self._setup_selenium()
            self.driver.get(url)

            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h1"))
            )

            # Get the page source and parse with BeautifulSoup
            soup = BeautifulSoup(self.driver.page_source, 'lxml')

            # Extract product information
            title = self._extract_title(soup)
            if not title:
                logger.warning(f"Could not extract title from {url}")
                return None

            image_url = self._extract_image_url(soup)
            description = self._extract_description(soup)
            price_info = self._extract_price_info(soup)
            metadata = self._extract_metadata(soup, title, description, price_info)

            product_data = ProductData(
                product_url=url,
                image_url=image_url or "",
                title=title,
                description=description,
                price=price_info['price'] or "",
                sale=price_info['sale'],
                metadata=metadata
            )

            logger.info(f"Successfully scraped: {title}")
            return product_data

        except Exception as e:
            logger.error(f"Error scraping product {url}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract product title"""
        try:
            title_elem = soup.find('h1')
            if title_elem:
                title = title_elem.get_text(strip=True)
                return ' '.join(title.split())
            return None
        except Exception as e:
            logger.error(f"Error extracting title: {e}")
            return None

    def _extract_image_url(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract product image URL"""
        try:
            # Look for images in JavaScript data first
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'ImageURL' in script.string:
                    # Extract from JavaScript object
                    image_match = re.search(r'ImageURL:\s*["\']([^"\']+)["\']', script.string)
                    if image_match:
                        return image_match.group(1)

            # Fallback to looking for img tags
            img_tags = soup.find_all('img')
            for img in img_tags:
                src = img.get('src') or img.get('data-src')
                if src and ('product' in src.lower() or 'grande' in src):
                    if src.startswith('//'):
                        src = 'https:' + src
                    elif src.startswith('/'):
                        src = urljoin(self.BASE_URL, src)
                    return src

            return None
        except Exception as e:
            logger.error(f"Error extracting image URL: {e}")
            return None

    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract product description"""
        try:
            # Look for description in the page content
            text_content = soup.get_text()
            # Look for common description patterns
            desc_patterns = [
                r'Description\s*(.*?)(?=Product details|$)',
                r'Our\s+[^.]*?\..*?(?=\n\n|\n\s*Product|\n\s*Made)'
            ]

            for pattern in desc_patterns:
                match = re.search(pattern, text_content, re.DOTALL | re.IGNORECASE)
                if match:
                    desc = match.group(1).strip()
                    if len(desc) > 20:
                        return desc

            return None
        except Exception as e:
            logger.error(f"Error extracting description: {e}")
            return None

    def _extract_price_info(self, soup: BeautifulSoup) -> Dict[str, Optional[str]]:
        """Extract price and sale information"""
        try:
            price_info = {'price': None, 'sale': None}

            # Look in JavaScript data first
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'Price' in script.string:
                    # Extract price from JavaScript
                    price_match = re.search(r'Price:\s*["\']([^"\']+)["\']', script.string)
                    if price_match:
                        price_info['price'] = price_match.group(1)

                    sale_match = re.search(r'CompareAtPrice:\s*["\']([^"\']+)["\']', script.string)
                    if sale_match and sale_match.group(1) != '"€0,00"':
                        price_info['sale'] = sale_match.group(1)

            return price_info
        except Exception as e:
            logger.error(f"Error extracting price info: {e}")
            return {'price': None, 'sale': None}

    def _extract_metadata(self, soup: BeautifulSoup, title: str, description: Optional[str], price_info: Dict) -> Optional[str]:
        """Extract additional metadata"""
        try:
            metadata = {
                'title': title,
                'description': description,
                'price': price_info.get('price'),
                'sale': price_info.get('sale'),
                'scraped_at': time.time()
            }

            return json.dumps(metadata, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return None

    def test_run(self, limit: int = 3):
        """Test run with limited products"""
        logger.info(f"Starting test run with limit: {limit}")

        try:
            # Get product URLs
            product_urls = self.get_product_urls()
            logger.info(f"Found {len(product_urls)} total products")

            # Limit for testing
            test_urls = product_urls[:limit]
            successful = 0

            for i, url in enumerate(test_urls, 1):
                logger.info(f"Testing product {i}/{len(test_urls)}: {url}")

                product_data = self.scrape_product_page(url)
                if product_data:
                    logger.info(f"SUCCESS: {product_data.title}")
                    logger.info(f"  Price: {product_data.price}")
                    logger.info(f"  Sale: {product_data.sale}")
                    logger.info(f"  Image: {product_data.image_url[:50] if product_data.image_url else 'None'}...")
                    successful += 1
                else:
                    logger.error(f"FAILED: {url}")

                # Small delay
                time.sleep(2)

            logger.info(f"Test completed: {successful}/{len(test_urls)} successful")
            return successful == len(test_urls)

        finally:
            self._cleanup_selenium()

def main():
    """Main entry point"""
    scraper = SimpleCulturSpaceScraper()
    success = scraper.test_run(limit=2)  # Test with just 2 products
    print(f"Test result: {'PASS' if success else 'FAIL'}")

if __name__ == "__main__":
    main()