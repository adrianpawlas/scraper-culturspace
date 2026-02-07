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
from transformers import SiglipImageProcessor, SiglipModel, SiglipTokenizer
from PIL import Image
import numpy as np
_supabase_client = None

def _get_supabase_client(url: str, key: str):
    """Use Supabase client if available; else None (caller uses REST fallback)."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception:
        return None

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
    image_embedding: List[float]
    info_embedding: List[float]
    size: Optional[str] = None
    category: Optional[str] = None
    additional_images: Optional[str] = None  # JSON array of URLs

class CulturSpaceScraper:
    """Main scraper class for Cultur Space"""

    BASE_URL = "https://culturspace.com"
    SHOP_ALL_URL = "https://culturspace.com/collections/shop-all"
    SOURCE = "scraper"
    BRAND = "Cultur Space"
    GENDER = "MAN"
    SECOND_HAND = False

    def __init__(self, supabase_url: str, supabase_key: str):
        self._supabase_url = supabase_url.rstrip("/")
        self._supabase_key = supabase_key
        self._supabase = _get_supabase_client(supabase_url, supabase_key)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        # Initialize Selenium for JavaScript-heavy pages
        self.driver = None

        # Initialize the image/text embedding model (SigLIP)
        self.processor = None
        self.model = None
        self.tokenizer = None
        self._init_embedding_model()

        logger.info("CulturSpace Scraper initialized")

    def _init_embedding_model(self):
        """Initialize the SigLIP model for image and text embeddings (768-dim)"""
        try:
            logger.info("Loading SigLIP model...")
            model_name = "google/siglip-base-patch16-384"
            # Use image processor + full model to avoid AutoProcessor tokenizer/processor_config 404 path
            self.processor = SiglipImageProcessor.from_pretrained(model_name)
            self.model = SiglipModel.from_pretrained(model_name)
            self.tokenizer = SiglipTokenizer.from_pretrained(model_name)

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

            image_url, additional_urls = self._extract_images(soup)
            if not image_url:
                logger.warning(f"Could not extract image URL from {url}")
                return None

            description = self._extract_description(soup)
            price_info = self._extract_price_info(soup, page_url=url)
            sizes = self._extract_sizes(soup)
            category = self._extract_category(soup)
            metadata = self._extract_metadata(soup, title, description, price_info)

            # Generate image embedding from main image only
            image_embedding = self._generate_image_embedding(image_url)
            if not image_embedding:
                logger.warning(f"Could not generate image embedding for {url}")
                return None

            # Build info text for text embedding (title, description, category, price, etc.)
            info_parts = [title, description or "", category or "", price_info.get('price') or "", price_info.get('sale') or ""]
            if sizes:
                info_parts.append(" ".join(sizes))
            info_parts.append(self.BRAND)
            info_text = " ".join(str(p).strip() for p in info_parts if p)
            info_embedding = self._generate_text_embedding(info_text)
            if not info_embedding:
                logger.warning(f"Could not generate info embedding for {url}")
                return None

            additional_images_json = json.dumps(additional_urls) if additional_urls else None

            product_data = ProductData(
                product_url=url,
                image_url=image_url,
                title=title,
                description=description,
                price=price_info.get('price') or '',
                sale=price_info.get('sale'),
                metadata=metadata,
                image_embedding=image_embedding,
                info_embedding=info_embedding,
                size=','.join(sizes) if sizes else None,
                category=category,
                additional_images=additional_images_json,
            )

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

    def _normalize_image_src(self, src: str) -> str:
        """Convert relative image URL to absolute."""
        if not src:
            return src
        if src.startswith('//'):
            return 'https:' + src
        if src.startswith('/'):
            return urljoin(self.BASE_URL, src)
        return src

    def _extract_images(self, soup: BeautifulSoup) -> Tuple[Optional[str], List[str]]:
        """Extract main product image URL and list of additional image URLs. Returns (main_url, additional_urls)."""
        try:
            all_urls: List[str] = []
            seen: set = set()

            def add_url(url: str) -> bool:
                url = self._normalize_image_src(url)
                if not url or url in seen:
                    return False
                seen.add(url)
                all_urls.append(url)
                return True

            # Prefer ImageURL from JavaScript (Klaviyo/analytics)
            for script in soup.find_all('script'):
                if script.string and 'ImageURL' in script.string:
                    m = re.search(r'ImageURL:\s*["\']([^"\']+)["\']', script.string)
                    if m and add_url(m.group(1)):
                        break

            # Collect from product gallery / images (often multiple)
            img_selectors = [
                '.product-gallery img',
                '.product-images img',
                '.product-image img',
                'img[data-image]',
                'img[alt*="product"]',
                'img.product__image',
                '.image-element img',
            ]
            for selector in img_selectors:
                for img_elem in soup.select(selector):
                    src = (img_elem.get('data-image') or img_elem.get('data-src') or img_elem.get('src'))
                    if src:
                        add_url(src)

            # Shopify / JSON in script: product.images or variants with images
            for script in soup.find_all('script'):
                if not script.string or 'images' not in script.string.lower():
                    continue
                # Match JSON arrays of image URLs
                for m in re.finditer(r'"(https?://[^"]+\.(?:jpg|jpeg|png|webp)[^"]*)"', script.string):
                    add_url(m.group(1))
                for m in re.finditer(r'"//[^"]+\.(?:jpg|jpeg|png|webp)[^"]*"', script.string):
                    add_url('https:' + m.group(0).strip('"'))

            # Fallback: any img with reasonable size or product-like alt
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src')
                if not src:
                    continue
                width = img.get('width') or img.get('data-width')
                height = img.get('height') or img.get('data-height')
                alt_text = (img.get('alt') or '').lower()
                if width and height:
                    try:
                        if int(width) > 200 and int(height) > 200:
                            add_url(src)
                            continue
                    except (ValueError, TypeError):
                        pass
                if any(k in alt_text for k in ['product', 'jacket', 'shirt', 'pants', 'tee', 'hoodie', 'short', 'cap']):
                    add_url(src)

            if not all_urls:
                return (None, [])
            main_url = all_urls[0]
            additional = [u for u in all_urls[1:] if u != main_url]
            return (main_url, additional)
        except Exception as e:
            logger.error(f"Error extracting images: {e}")
            return (None, [])

    def _extract_category(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract product category/categories (e.g. 'Hoodies & Sweaters' -> 'Hoodies, Sweaters')."""
        try:
            # Breadcrumb: e.g. Home > Shop All > Hoodies & Sweaters
            breadcrumbs = soup.select('[class*="breadcrumb"] a, nav a, .breadcrumb a, [aria-label="breadcrumb"] a')
            for a in breadcrumbs:
                text = a.get_text(strip=True)
                if text and text.lower() not in ('home', 'shop', 'shop all', 'all') and len(text) > 1:
                    # Normalize "Hoodies & Sweaters" -> "Hoodies, Sweaters"
                    category = re.sub(r'\s*&\s*', ', ', text)
                    if len(category) > 2:
                        return category

            # Collection links in product section
            for a in soup.find_all('a', href=re.compile(r'/collections/')):
                href = a.get('href', '')
                if '/collections/shop-all' in href:
                    continue
                text = a.get_text(strip=True)
                if text and len(text) > 2:
                    category = re.sub(r'\s*&\s*', ', ', text)
                    return category

            # Meta or JSON-LD
            for script in soup.find_all('script', {'type': 'application/ld+json'}):
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict) and 'itemListElement' in data:
                        for item in data.get('itemListElement', []):
                            if isinstance(item, dict):
                                name = item.get('name', '')
                                if name and name.lower() not in ('home', 'shop all'):
                                    return re.sub(r'\s*&\s*', ', ', name)
                except (json.JSONDecodeError, TypeError):
                    continue
            return None
        except Exception as e:
            logger.error(f"Error extracting category: {e}")
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

    def _extract_price_info(self, soup: BeautifulSoup, page_url: Optional[str] = None) -> Dict[str, Optional[str]]:
        """Extract price and sale information in multiple currencies.
        Cultur Space (Shopify) often shows USD ($); we accept EUR (€) or USD and convert to multi-currency.
        When page_url is given, tries Shopify product JSON first (/products/{handle}.js) for accurate prices.
        """
        try:
            price_info = {'price': None, 'sale': None}
            eur_price = None
            eur_sale = None

            # 0) Shopify product JSON: reliable price/compare_at_price in cents (store currency)
            if page_url:
                handle = None
                if '/products/' in page_url:
                    handle = page_url.split('/products/')[-1].split('?')[0].strip('/')
                if handle:
                    try:
                        js_url = f"{self.BASE_URL}/products/{handle}.js"
                        r = self.session.get(js_url, timeout=10)
                        if r.ok:
                            data = r.json()
                            v = data.get('variants') or []
                            if v:
                                p = v[0]
                                # price in cents (Shopify)
                                raw_price = p.get('price')
                                raw_compare = p.get('compare_at_price')
                                if raw_price is not None:
                                    val = float(str(raw_price).replace(',', '.'))
                                    # Shopify .js can return cents (e.g. 13317) or decimal (133.17)
                                    if val >= 100 and val == int(val):
                                        eur_price = round(val / 100 / 1.08, 2)
                                    else:
                                        eur_price = round(val / 1.08, 2)
                                if raw_compare:
                                    cv = float(str(raw_compare).replace(',', '.'))
                                    if cv > 0:
                                        if cv >= 100 and cv == int(cv):
                                            eur_sale = round(cv / 100 / 1.08, 2)
                                        else:
                                            eur_sale = round(cv / 1.08, 2)
                    except Exception:
                        pass
            # Match prices: $123.45 / €123.45 / 123.45$ / 123.45 USD (no thousands separator)
            re_eur = re.compile(r'€\s*(\d{1,4}(?:[.,]\d{1,2})?)')
            re_usd = re.compile(r'\$\s*(\d{1,4}(?:[.,]\d{1,2})?)')
            re_usd_suffix = re.compile(r'(\d{1,4}(?:[.,]\d{1,2})?)\s*(?:\$|USD)')
            re_plain_num = re.compile(r'(\d{1,4}(?:[.,]\d{1,2})?)')

            def parse_amount(s: str) -> Optional[float]:
                if not s or '..' in s or s.count('.') > 1:
                    return None
                s_clean = s.replace(',', '.')
                m = re_plain_num.match(s_clean.strip())
                if not m:
                    return None
                try:
                    return float(m.group(1).replace(',', '.'))
                except ValueError:
                    return None

            def parse_amount_from_group(g: str) -> Optional[float]:
                """Parse a regex group; reject European thousands (e.g. 2.201.65)."""
                if not g or len(g) > 12:
                    return None
                if g.count('.') > 1 or g.count(',') > 1:
                    return None
                if g.count('.') == 1 and g.count(',') == 0:
                    parts = g.split('.')
                    if len(parts) == 2 and len(parts[1]) <= 2 and len(parts[0]) <= 4:
                        return float(g)
                if g.count(',') == 1 and g.count('.') == 0:
                    parts = g.split(',')
                    if len(parts) == 2 and len(parts[1]) <= 2 and len(parts[0]) <= 4:
                        return float(g.replace(',', '.'))
                return None

            # Plausible product price range (EUR): 3–2500. Reject noise (e.g. 1.45, 875 from "2.201.65").
            def plausible_price(v: Optional[float]) -> bool:
                return v is not None and 3 <= v <= 2500

            # Prefer main product block to avoid picking cart/related prices
            product_block = soup.find('main') or soup.find(class_=re.compile(r'product[^_]|product__info|product-single', re.I))
            block_text = product_block.get_text() if product_block else soup.get_text()
            full = block_text

            # 1) Main block text: Cultur Space (Shopify) shows e.g. "$133.17" or "133.17 $" in body
            all_usd = re_usd.findall(full)
            all_usd_suffix = re_usd_suffix.findall(full)
            all_eur = re_eur.findall(full)
            usd_vals = [v for s in (all_usd + all_usd_suffix) for v in [parse_amount_from_group(s)] if plausible_price(v)]
            if usd_vals and eur_price is None:
                usd_dedup = sorted(set(usd_vals), reverse=True)
                # Typical product price 10–500 USD; avoid cart totals or noise
                typical = [v for v in usd_dedup if 10 <= v <= 500]
                pick = typical[0] if typical else usd_dedup[0]
                eur_price = round(pick / 1.08, 2)
            if all_eur and eur_price is None:
                v = parse_amount_from_group(all_eur[0])
                if plausible_price(v):
                    eur_price = v
            if len(usd_vals) >= 2 and eur_sale is None:
                vals = sorted(set(usd_vals), reverse=True)
                typical_pair = [(a, b) for a, b in zip(vals, vals[1:]) if 10 <= b <= 500 and a > b]
                if typical_pair and eur_sale is None:
                    hi, lo = typical_pair[0]
                    if eur_price is None:
                        eur_price = round(hi / 1.08, 2)
                    eur_sale = round(lo / 1.08, 2)
                elif len(vals) >= 2 and vals[0] > vals[1] and vals[1] >= 3:
                    if eur_price is None:
                        eur_price = round(vals[0] / 1.08, 2)
                    eur_sale = round(vals[1] / 1.08, 2)

            # 2) Price-like elements (backup)
            price_selectors = [
                'span.price', '.product-price', '[data-price]', '.price-item',
                '[class*="price"]', '.product__price', '.price__regular', '.money',
            ]
            if eur_price is None or eur_sale is None:
                for selector in price_selectors:
                    for elem in soup.select(selector):
                        txt = elem.get_text(strip=True)
                        if not txt or len(txt) > 80:
                            continue
                        em = re_eur.search(txt)
                        if em and eur_price is None:
                            v = parse_amount_from_group(em.group(1))
                            if plausible_price(v):
                                eur_price = v
                        um = re_usd.search(txt)
                        if um and eur_price is None:
                            usd_val = parse_amount_from_group(um.group(1))
                            if plausible_price(usd_val):
                                eur_price = round(usd_val / 1.08, 2)
                        nums = [n for n in (parse_amount_from_group(m.group(1)) for m in re_plain_num.finditer(txt)) if plausible_price(n)]
                        if len(nums) >= 2 and eur_sale is None:
                            lo, hi = min(nums), max(nums)
                            is_usd = '$' in txt or 'USD' in txt
                            if eur_price is None and plausible_price(hi):
                                eur_price = round(hi / 1.08, 2) if is_usd else hi
                            if lo < hi and plausible_price(lo):
                                eur_sale = round(lo / 1.08, 2) if is_usd else lo
                        if eur_price is not None:
                            break
                    if eur_price is not None:
                        break

            # 3) Scripts: Price/CompareAtPrice (Klaviyo-style) or JSON "price" (Shopify)
            if eur_price is None or eur_sale is None:
                for script in soup.find_all('script'):
                    if not script.string:
                        continue
                    t = script.string
                    if 'Price' in t or 'price' in t:
                        pm = re.search(r'Price:\s*["\']([^"\']+)["\']', t)
                        if pm and eur_price is None:
                            v = parse_amount_from_group(re.sub(r'[^\d.,]', '', pm.group(1)))
                            if plausible_price(v):
                                eur_price = round(v / 1.08, 2) if ('$' in pm.group(1) or 'USD' in pm.group(1)) else v
                        sm = re.search(r'CompareAtPrice:\s*["\']([^"\']+)["\']', t)
                        if sm and sm.group(1).strip() and '0' not in sm.group(1) and eur_sale is None:
                            v = parse_amount_from_group(re.sub(r'[^\d.,]', '', sm.group(1)))
                            if plausible_price(v):
                                eur_sale = round(v / 1.08, 2) if ('$' in sm.group(1) or 'USD' in sm.group(1)) else v
                    # Shopify JSON: "price":12300 (cents)
                    if eur_price is None and '"price":' in t:
                        pos = t.find('"price":')
                        rest = t[pos + 7:].lstrip()
                        cents = re.match(r'(\d+)', rest)
                        if cents:
                            eur_price = round(int(cents.group(1)) / 100 / 1.08, 2)

            # Convert to multi-currency string (required format)
            if eur_price is not None and eur_price > 0:
                price_info['price'] = self._eur_to_multi_currency(eur_price)
            if eur_sale is not None and eur_sale > 0 and (eur_price is None or eur_sale < (eur_price or 0)):
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
            response = self.session.get(image_url, timeout=60)
            response.raise_for_status()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name

            try:
                # Load and process image
                image = Image.open(temp_path).convert('RGB')

                # Process image (SiglipImageProcessor returns pixel_values)
                inputs = self.processor(images=image, return_tensors="pt")

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Generate 768-dim embedding via SiglipModel.get_image_features
                with torch.no_grad():
                    out = self.model.get_image_features(**inputs)
                    # Handle tensor or BaseModelOutputWithPooling
                    if hasattr(out, "pooler_output") and out.pooler_output is not None:
                        embedding = out.pooler_output
                    elif hasattr(out, "last_hidden_state"):
                        embedding = out.last_hidden_state[:, 0]
                    else:
                        embedding = out
                    embedding = embedding.squeeze().cpu().float().numpy()

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

    def _generate_text_embedding(self, text: str) -> Optional[List[float]]:
        """Generate 768-dimensional text embedding using SigLIP text encoder (same model as image)."""
        if not text or not text.strip():
            return None
        try:
            # SigLIP tokenizer: use padding="max_length" and max_length=64 as per model training
            inputs = self.tokenizer(
                text.strip(),
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                out = self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                # Handle tensor or BaseModelOutputWithPooling
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    embedding = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    embedding = out.last_hidden_state[:, 0]
                else:
                    embedding = out
                embedding = embedding.squeeze().cpu().float().numpy()

            if len(embedding.shape) > 1:
                embedding = embedding.ravel()
            if embedding.shape[0] != 768:
                logger.warning(f"Text embedding dimension mismatch: {embedding.shape[0]}, expected 768")
                return None

            embedding = embedding.astype(np.float64) / np.linalg.norm(embedding)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
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
                'category': product_data.category,
                'gender': self.GENDER,
                'second_hand': self.SECOND_HAND,
                'image_embedding': product_data.image_embedding,
                'info_embedding': product_data.info_embedding,
                'price': product_data.price,
                'sale': product_data.sale,
                'metadata': product_data.metadata,
                'size': product_data.size,
                'additional_images': product_data.additional_images,
            }

            if self._supabase is not None:
                self._supabase.table('products').upsert(
                    data, on_conflict='source,product_url'
                ).execute()
            else:
                # REST fallback when supabase client fails (e.g. websockets.asyncio)
                rest_url = f"{self._supabase_url}/rest/v1/products"
                headers = {
                    "apikey": self._supabase_key,
                    "Authorization": f"Bearer {self._supabase_key}",
                    "Content-Type": "application/json",
                    "Prefer": "resolution=merge-duplicates,return=minimal",
                }
                resp = self.session.post(
                    f"{rest_url}?on_conflict=source,product_url",
                    headers=headers,
                    json=data,
                    timeout=30,
                )
                resp.raise_for_status()

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