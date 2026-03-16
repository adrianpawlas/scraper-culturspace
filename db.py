"""
Supabase PostgREST helper for products table.
Uses pure HTTP (no SDK) for reliable import in scheduled/cron environments.
Implements smart sync: insert new, update changed, skip unchanged, remove stale products.
"""
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
import requests
from urllib.parse import quote

logger = logging.getLogger(__name__)


class SupabaseREST:
    """Minimal Supabase PostgREST helper for products table.

    Implements:
    - Batch inserts (50 products per batch)
    - Smart upsert: only update if data changed
    - Track last_seen_run for stale detection
    - Stale product cleanup (2 consecutive runs not seen)
    """

    def __init__(self, url: str, key: str) -> None:
        self.base_url = url.rstrip("/")
        self.key = key
        self.session = requests.Session()
        self.session.headers.update({
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        })

    def get_products_for_source(self, source: str) -> List[Dict]:
        """Fetch all products for a given source with their current state."""
        url = f"{self.base_url}/rest/v1/products?source=eq.{source}&select=id,source,product_url,image_url,title,description,category,price,sale,metadata,size,additional_images,updated_at,last_seen_run"
        resp = self.session.get(url, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def get_current_run_number(self, source: str) -> int:
        """Get the current run number for this source. If no runs tracked, start at 1."""
        url = f"{self.base_url}/rest/v1/products?source=eq.{source}&select=last_seen_run&limit=1"
        resp = self.session.get(url, timeout=60)
        if resp.status_code == 200 and resp.json():
            max_run = max(p.get("last_seen_run", 0) or 0 for p in resp.json() if p.get("last_seen_run"))
            return max_run + 1
        return 1

    def _has_data_changed(self, existing: Dict, new: Dict) -> bool:
        """Check if any relevant data has changed between existing and new product."""
        fields_to_check = ["title", "description", "category", "price", "sale", "metadata", "size", "additional_images"]
        for field in fields_to_check:
            existing_val = existing.get(field)
            new_val = new.get(field)
            if existing_val != new_val:
                return True
        return False

    def _chunk_list(self, items: List, chunk_size: int) -> List[List]:
        """Split a list into chunks of specified size."""
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    def _insert_batch_with_retry(self, products: List[Dict], ignore_existing: bool, max_retries: int = 3) -> Tuple[int, int]:
        """Insert a batch of products with retry logic. Returns (successful_count, failed_count)."""
        if not products:
            return 0, 0

        all_keys = set()
        for p in products:
            all_keys.update(p.keys())
        normalized = [{k: p.get(k) for k in all_keys} for p in products]

        resolution = "resolution=ignore-duplicates" if ignore_existing else "resolution=merge-duplicates"
        headers = {"Prefer": f"{resolution},return=minimal"}

        endpoint = (
            f"{self.base_url}/rest/v1/products"
            "?on_conflict=source,product_url"
            "&columns=id,source,product_url,image_url,brand,title,description,category,gender,"
            "price,sale,metadata,size,second_hand,image_embedding,info_embedding,additional_images,updated_at,last_seen_run"
        )

        chunks = self._chunk_list(normalized, 50)
        successful = 0
        failed = 0
        failed_products = []

        for chunk_idx, chunk in enumerate(chunks):
            for attempt in range(max_retries):
                try:
                    resp = self.session.post(endpoint, headers=headers, data=json.dumps(chunk), timeout=120)
                    if resp.status_code in (200, 201, 204):
                        successful += len(chunk)
                        break
                    elif attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    else:
                        failed += len(chunk)
                        failed_products.extend(chunk)
                        logger.error(f"Batch insert failed after {max_retries} attempts: {resp.status_code} {resp.text}")
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    else:
                        failed += len(chunk)
                        failed_products.extend(chunk)
                        logger.error(f"Batch insert exception after {max_retries} attempts: {e}")

        if failed_products:
            self._log_failed_products(failed_products)

        return successful, failed

    def _log_failed_products(self, products: List[Dict]) -> None:
        """Log failed products to a local file."""
        try:
            with open("failed_products.log", "a") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n--- Failed products at {timestamp} ---\n")
                for p in products:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            logger.warning(f"Logged {len(products)} failed products to failed_products.log")
        except Exception as e:
            logger.error(f"Failed to log failed products: {e}")

    def smart_upsert_products(self, products: List[Dict], source: str, current_run: int, max_retries: int = 3) -> Dict:
        """Smart upsert with change detection and batch processing.

        Returns dict with:
        - new_count: products that were newly inserted
        - updated_count: products that had data changes
        - unchanged_count: products that already existed with same data
        - failed_count: products that failed to insert
        """
        if not products:
            return {"new_count": 0, "updated_count": 0, "unchanged_count": 0, "failed_count": 0}

        existing_products = self.get_products_for_source(source)
        existing_by_url = {p["product_url"]: p for p in existing_products}

        to_insert_new = []
        to_update = []
        unchanged_count = 0

        for product in products:
            product_url = product.get("product_url")
            existing = existing_by_url.get(product_url)

            if existing is None:
                product["last_seen_run"] = current_run
                to_insert_new.append(product)
            else:
                existing_id = existing.get("id")
                product["id"] = existing_id
                product["last_seen_run"] = current_run

                if self._has_data_changed(existing, product):
                    to_update.append(product)
                else:
                    unchanged_count += 1

        stats = {"new_count": 0, "updated_count": 0, "unchanged_count": unchanged_count, "failed_count": 0}

        if to_insert_new:
            successful, failed = self._insert_batch_with_retry(to_insert_new, ignore_existing=True, max_retries=max_retries)
            stats["new_count"] = successful
            stats["failed_count"] = failed

        if to_update:
            successful, failed = self._insert_batch_with_retry(to_update, ignore_existing=False, max_retries=max_retries)
            stats["updated_count"] = successful
            stats["failed_count"] += failed

        return stats

    def mark_products_seen(self, product_urls: List[str], source: str, current_run: int) -> None:
        """Mark products as seen in current run by updating last_seen_run."""
        if not product_urls:
            return

        for url in product_urls:
            update_data = {"last_seen_run": current_run}
            url_encoded = quote(url, safe='')
            update_url = f"{self.base_url}/rest/v1/products?source=eq.{source}&product_url=eq.{url_encoded}"
            self.session.patch(update_url, json=update_data, timeout=30)

    def delete_stale_products(self, source: str, current_run: int, max_missing_runs: int = 2) -> int:
        """Delete products not seen for consecutive runs.

        Args:
            source: The source to clean up
            current_run: The current run number
            max_missing_runs: Number of consecutive runs a product can be missing before deletion (default: 2)

        Returns number of products deleted.
        """
        products = self.get_products_for_source(source)
        to_delete = []

        for product in products:
            last_seen = product.get("last_seen_run") or 0
            runs_missed = current_run - last_seen
            if runs_missed >= max_missing_runs:
                to_delete.append(product["id"])

        deleted = 0
        for pid in to_delete:
            del_url = f"{self.base_url}/rest/v1/products?source=eq.{source}&id=eq.{pid}"
            del_resp = self.session.delete(del_url, timeout=60)
            if del_resp.status_code in (200, 204):
                deleted += 1
            else:
                logger.warning(f"Failed to delete stale product {pid}: {del_resp.status_code}")

        return deleted

    def upsert_products(self, products: List[Dict], ignore_existing: bool = True) -> None:
        """Legacy method - use smart_upsert_products instead."""
        if not products:
            return
        self._insert_batch_with_retry(products, ignore_existing)

    def delete_missing_for_source(self, source: str, current_ids: List[str]) -> int:
        """Legacy method - use delete_stale_products instead."""
        return self.delete_stale_products(source, self.get_current_run_number(source))
