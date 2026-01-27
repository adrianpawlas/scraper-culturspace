#!/usr/bin/env python3
"""
Cultur Space Fashion Scraper
Scrapes all products from Cultur Space, generates image embeddings, and stores in Supabase.
"""

import os
import json
import time
import hashlib
import requests
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from pathlib import Path
import tempfile

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import numpy as np
try:
    from supabase import create_client, Client
except ImportError:
    try:
        from supabase_py import create_client, Client
    except ImportError:
        # Fallback for older versions
        import supabase
        create_client = supabase.create_client
        Client = supabase.Client
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
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
    embedding: List[float]
    size: Optional[str] = None

class CulturSpaceScraper:
    """Main scraper class for Cultur Space"""

    BASE_URL = "https://culturspace.com"
    SHOP_ALL_URL = "https://culturspace.com/collections/shop-all"
    SOURCE = "scraper"
    BRAND = "Cultur Space"
    GENDER = "MAN"
    SECOND_HAND = False

    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        # Initialize Selenium for JavaScript-heavy pages
        self.driver = None

        # Initialize the image embedding model
        self.processor = None
        self.model = None
        self._init_embedding_model()

        logger.info("CulturSpace Scraper initialized")

    def _init_embedding_model(self):
        """Initialize the SigLIP model for image embeddings"""
        try:
            logger.info("Loading SigLIP model...")
            model_name = "google/siglip-base-patch16-384"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Using CUDA for embeddings")
            else:
                logger.info("Using CPU for embeddings")

            self.model.eval()
            logger.info("SigLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _setup_selenium(self):
        """Setup Selenium WebDriver"""
        if self.driver is None:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

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
            if not image_url:
                logger.warning(f"Could not extract image URL from {url}")
                return None

            description = self._extract_description(soup)
            price_info = self._extract_price_info(soup)
            sizes = self._extract_sizes(soup)
            metadata = self._extract_metadata(soup, title, description, price_info)

            # Generate embedding from image
            embedding = self._generate_image_embedding(image_url)

            if not embedding:
                logger.warning(f"Could not generate embedding for {url}")
                return None

            product_data = ProductData(
                product_url=url,
                image_url=image_url,
                title=title,
                description=description,
                price=price_info['price'],
                sale=price_info['sale'],
                metadata=metadata,
                embedding=embedding
            )

            # Add size information to the object for database storage
            product_data.size = ','.join(sizes) if sizes else None

            logger.info(f"Successfully scraped: {title}")
            return product_data

        except Exception as e:
            logger.error(f"Error scraping product {url}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract product title"""
        try:
            # Look for h1 tag with product title
            title_elem = soup.find('h1')
            if title_elem:
                title = title_elem.get_text(strip=True)
                # Remove any extra whitespace and normalize
                return ' '.join(title.split())

            # Alternative: look for title in specific product title class
            title_elem = soup.find('h1', class_=re.compile(r'product-title|title'))
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
            # Prefer ImageURL from JavaScript (Klaviyo/analytics) - matches live site
            for script in soup.find_all('script'):
                if script.string and 'ImageURL' in script.string:
                    m = re.search(r'ImageURL:\s*["\']([^"\']+)["\']', script.string)
                    if m:
                        url = m.group(1)
                        if url.startswith('//'):
                            url = 'https:' + url
                        elif url.startswith('/'):
                            url = urljoin(self.BASE_URL, url)
                        return url

            # Look for product images in various possible locations
            img_selectors = [
                'img[data-image]',
                '.product-gallery img',
                '.product-images img',
                '.product-image img',
                'img[alt*="product"]',
                'img.product__image',
                '.image-element img'
            ]

            for selector in img_selectors:
                img_elem = soup.select_one(selector)
                if img_elem:
                    # Try data-image, data-src, or src attributes
                    src = (img_elem.get('data-image') or
                          img_elem.get('data-src') or
                          img_elem.get('src'))
                    if src:
                        # Convert relative URLs to absolute
                        if src.startswith('//'):
                            src = 'https:' + src
                        elif src.startswith('/'):
                            src = urljoin(self.BASE_URL, src)
                        return src

            # Fallback: look for any img tag with reasonable size
            img_tags = soup.find_all('img')
            for img in img_tags:
                src = img.get('src') or img.get('data-src')
                if src:
                    # Skip tiny images, icons, etc.
                    width = img.get('width') or img.get('data-width')
                    height = img.get('height') or img.get('data-height')
                    if width and height:
                        try:
                            if int(width) > 200 and int(height) > 200:
                                if src.startswith('//'):
                                    src = 'https:' + src
                                elif src.startswith('/'):
                                    src = urljoin(self.BASE_URL, src)
                                return src
                        except (ValueError, TypeError):
                            pass

                    # If no dimensions, check if it looks like a product image
                    alt_text = img.get('alt', '').lower()
                    if any(keyword in alt_text for keyword in ['product', 'jacket', 'shirt', 'pants', 'tee']):
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
            # Look for description in various possible locations
            desc_selectors = [
                '.product-description',
                '.description',
                '[data-description]',
                '.product-details',
                '.product-info'
            ]

            for selector in desc_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem:
                    description = desc_elem.get_text(strip=True)
                    if description and len(description) > 10:  # Ensure it's substantial
                        return description

            # Look for description in structured data
            script_tags = soup.find_all('script', {'type': 'application/ld+json'})
            for script in script_tags:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and 'description' in data:
                        return data['description']
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'description' in item:
                                return item['description']
                except (json.JSONDecodeError, TypeError):
                    continue

            # Look for meta description
            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc:
                content = meta_desc.get('content')
                if content and len(content) > 10:
                    return content

            # Look for any paragraph that might contain description
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text and len(text) > 20 and not any(skip in text.lower() for skip in ['shipping', 'returns', 'contact']):
                    return text

            return None
        except Exception as e:
            logger.error(f"Error extracting description: {e}")
            return None

    def _extract_price_info(self, soup: BeautifulSoup) -> Dict[str, Optional[str]]:
        """Extract price and sale information in multiple currencies"""
        try:
            price_info = {'price': None, 'sale': None}

            # Extract the base EUR price first
            eur_price = None
            eur_sale = None

            # Look for price elements - try multiple selectors
            price_selectors = [
                'span.price',
                '.product-price',
                '[data-price]',
                '.price-item'
            ]

            for selector in price_selectors:
                price_elem = soup.select_one(selector)
                if price_elem:
                    price_text = price_elem.get_text(strip=True)
                    # Extract EUR price
                    eur_match = re.search(r'€(\d+(?:[,.]\d+)*)', price_text)
                    if eur_match:
                        eur_price = float(eur_match.group(1).replace(',', '.'))
                        break

            # Look for sale/discounted price
            sale_selectors = [
                'span.sale',
                '.sale-price',
                '.discounted-price',
                '[data-sale-price]'
            ]

            for selector in sale_selectors:
                sale_elem = soup.select_one(selector)
                if sale_elem:
                    sale_text = sale_elem.get_text(strip=True)
                    eur_sale_match = re.search(r'€(\d+(?:[,.]\d+)*)', sale_text)
                    if eur_sale_match:
                        eur_sale = float(eur_sale_match.group(1).replace(',', '.'))
                        break

            # Fallback: look in JavaScript (Klaviyo/analytics) for Price, ImageURL
            if eur_price is None or eur_sale is None:
                for script in soup.find_all('script'):
                    if script.string and 'Price' in script.string:
                        pm = re.search(r'Price:\s*["\']([^"\']+)["\']', script.string)
                        if pm and eur_price is None:
                            s = pm.group(1)
                            m = re.search(r'([\d,.]+)', s)
                            if m:
                                eur_price = float(m.group(1).replace(',', '.'))
                        sm = re.search(r'CompareAtPrice:\s*["\']([^"\']+)["\']', script.string)
                        if sm and sm.group(1) and '0' not in sm.group(1):
                            s = sm.group(1)
                            m = re.search(r'([\d,.]+)', s)
                            if m:
                                eur_sale = float(m.group(1).replace(',', '.'))
                        break

            # If we found prices, convert to multiple currencies
            if eur_price is not None:
                price_info['price'] = self._eur_to_multi_currency(eur_price)
            if eur_sale is not None and eur_sale > 0:
                price_info['sale'] = self._eur_to_multi_currency(eur_sale)

            return price_info
        except Exception as e:
            logger.error(f"Error extracting price info: {e}")
            return {'price': None, 'sale': None}

    def _eur_to_multi_currency(self, eur_amount: float) -> str:
        """Convert EUR amount to multiple currencies (20.90USD,450CZK,75PLN,...)."""
        try:
            currencies = {
                'EUR': eur_amount,
                'USD': eur_amount * 1.08,
                'GBP': eur_amount * 0.85,
                'PLN': eur_amount * 4.30,
                'CZK': eur_amount * 24.50,
                'SEK': eur_amount * 11.50,
                'DKK': eur_amount * 7.45,
                'NOK': eur_amount * 11.80,
                'CHF': eur_amount * 0.95,
                'CAD': eur_amount * 1.45,
                'AUD': eur_amount * 1.65,
            }

            formatted_prices = []
            for currency, amount in currencies.items():
                if currency == 'EUR':
                    formatted_amount = f"{amount:.2f}".replace('.', ',')
                else:
                    formatted_amount = f"{amount:.2f}"
                formatted_prices.append(f"{formatted_amount}{currency}")

            return ','.join(formatted_prices)
        except Exception as e:
            logger.error(f"Error converting currencies: {e}")
            return f"{eur_amount:.2f}EUR"

    def _extract_metadata(self, soup: BeautifulSoup, title: str, description: Optional[str], price_info: Dict) -> Optional[str]:
        """Extract additional metadata"""
        try:
            metadata = {
                'title': title,
                'description': description,
                'price': price_info.get('price'),
                'sale': price_info.get('sale'),
                'scraped_at': time.time(),
                'scraped_timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            }

            # Extract available sizes
            sizes = self._extract_sizes(soup)
            if sizes:
                metadata['sizes'] = sizes

            # Extract product details/features
            details = self._extract_product_details(soup)
            if details:
                metadata['product_details'] = details

            # Extract material information
            materials = self._extract_materials(soup)
            if materials:
                metadata['materials'] = materials

            # Extract care instructions if available
            care = self._extract_care_instructions(soup)
            if care:
                metadata['care_instructions'] = care

            return json.dumps(metadata, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return None

    def _extract_sizes(self, soup: BeautifulSoup) -> Optional[List[str]]:
        """Extract available sizes"""
        try:
            sizes = []

            # Look for size selectors/buttons
            size_selectors = [
                '.size-selector',
                '.product-size',
                '[data-size]',
                '.variant-selector'
            ]

            for selector in size_selectors:
                size_elems = soup.select(selector)
                for elem in size_elems:
                    size_text = elem.get_text(strip=True)
                    if size_text and len(size_text) <= 5:  # Reasonable size length
                        sizes.append(size_text)

            # Alternative: look for size options in select dropdown
            size_select = soup.find('select', {'name': re.compile(r'size|variant')})
            if size_select:
                options = size_select.find_all('option')
                for option in options:
                    value = option.get('value')
                    text = option.get_text(strip=True)
                    if value and text and value != '' and text not in ['Select Size', 'Choose Size']:
                        sizes.append(text)

            return list(set(sizes)) if sizes else None
        except Exception as e:
            logger.error(f"Error extracting sizes: {e}")
            return None

    def _extract_product_details(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract product details/features"""
        try:
            details_selectors = [
                '.product-details',
                '.product-features',
                '.details',
                '[data-details]'
            ]

            for selector in details_selectors:
                details_elem = soup.select_one(selector)
                if details_elem:
                    details = details_elem.get_text(strip=True)
                    if details and len(details) > 10:
                        return details

            return None
        except Exception as e:
            logger.error(f"Error extracting product details: {e}")
            return None

    def _extract_materials(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract material information"""
        try:
            # Look for material information in description or details
            text_content = soup.get_text().lower()

            # Common material keywords
            material_keywords = ['polyester', 'cotton', 'elastane', 'spandex', 'nylon', 'wool', 'leather']

            materials_found = []
            for keyword in material_keywords:
                if keyword in text_content:
                    # Extract surrounding context
                    start = text_content.find(keyword)
                    if start != -1:
                        end = min(start + 50, len(text_content))
                        context = text_content[start:end]
                        materials_found.append(context.strip())

            return ' '.join(materials_found) if materials_found else None
        except Exception as e:
            logger.error(f"Error extracting materials: {e}")
            return None

    def _extract_care_instructions(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract care instructions"""
        try:
            # Look for care/washing instructions
            care_keywords = ['care', 'washing', 'wash', 'dry clean', 'machine wash']

            text_content = soup.get_text().lower()
            for keyword in care_keywords:
                if keyword in text_content:
                    # Find the paragraph or section containing care info
                    paragraphs = soup.find_all(['p', 'div', 'span'])
                    for para in paragraphs:
                        text = para.get_text(strip=True).lower()
                        if keyword in text and len(text) > 10:
                            return para.get_text(strip=True)

            return None
        except Exception as e:
            logger.error(f"Error extracting care instructions: {e}")
            return None

    def _generate_image_embedding(self, image_url: str) -> Optional[List[float]]:
        """Generate 768-dimensional embedding from image URL"""
        try:
            # Download image
            response = self.session.get(image_url, timeout=30)
            response.raise_for_status()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name

            try:
                # Load and process image
                image = Image.open(temp_path).convert('RGB')

                # Process image
                inputs = self.processor(images=image, return_tensors="pt")

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Generate embedding (SigLIP vision encoder outputs 768-dim)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # SiglipModel / SiglipVisionModel: pooler_output or image_embeds or last_hidden_state
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embedding = outputs.pooler_output.squeeze().cpu().numpy()
                    elif hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
                        embedding = outputs.image_embeds.squeeze().cpu().numpy()
                    elif hasattr(outputs, 'last_hidden_state'):
                        embedding = outputs.last_hidden_state[:, 0].squeeze().cpu().numpy()
                    else:
                        logger.warning("Could not get embedding from model outputs")
                        return None

                # Ensure 768 dimensions
                if len(embedding.shape) > 1:
                    embedding = embedding.ravel()
                if embedding.shape[0] != 768:
                    logger.warning(f"Embedding dimension mismatch: {embedding.shape[0]}, expected 768")
                    return None

                # Normalize embedding
                embedding = embedding.astype(np.float64) / np.linalg.norm(embedding)

                return embedding.tolist()

            finally:
                # Clean up temp file
                os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Error generating embedding for {image_url}: {e}")
            return None

    def save_to_supabase(self, product_data: ProductData) -> bool:
        """Save product data to Supabase"""
        try:
            # Stable unique ID from source + product_url
            raw = f"{self.SOURCE}|{product_data.product_url}"
            product_id = f"cultur_space_{hashlib.md5(raw.encode()).hexdigest()[:16]}"

            data = {
                'id': product_id,
                'source': self.SOURCE,
                'product_url': product_data.product_url,
                'image_url': product_data.image_url,
                'brand': self.BRAND,
                'title': product_data.title,
                'description': product_data.description,
                'gender': self.GENDER,
                'second_hand': self.SECOND_HAND,
                'embedding': product_data.embedding,
                'price': product_data.price,
                'sale': product_data.sale,
                'metadata': product_data.metadata,
                'size': product_data.size,
            }

            # Insert or update
            result = self.supabase.table('products').upsert(
                data,
                on_conflict='source,product_url'
            ).execute()

            logger.info(f"Successfully saved product: {product_data.title}")
            return True

        except Exception as e:
            logger.error(f"Error saving to Supabase: {e}")
            return False

    def run(self):
        """Main execution method"""
        logger.info("Starting Cultur Space scraper")

        try:
            # Get all product URLs
            product_urls = self.get_product_urls()
            logger.info(f"Found {len(product_urls)} products to scrape")

            # Scrape each product
            successful = 0
            failed = 0

            for url in tqdm(product_urls, desc="Scraping products"):
                try:
                    product_data = self.scrape_product_page(url)
                    if product_data:
                        if self.save_to_supabase(product_data):
                            successful += 1
                        else:
                            failed += 1
                    else:
                        failed += 1

                    # Small delay to be respectful
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Failed to process {url}: {e}")
                    failed += 1

            logger.info(f"Scraping completed. Successful: {successful}, Failed: {failed}")

        finally:
            self._cleanup_selenium()

def main():
    """Main entry point"""
    # Supabase configuration
    SUPABASE_URL = "https://yqawmzggcgpeyaaynrjk.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4"

    # Initialize and run scraper
    scraper = CulturSpaceScraper(SUPABASE_URL, SUPABASE_KEY)
    scraper.run()

if __name__ == "__main__":
    main()