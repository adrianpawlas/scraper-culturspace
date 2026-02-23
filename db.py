"""
Supabase PostgREST helper for products table.
Uses pure HTTP (no SDK) for reliable import in scheduled/cron environments.
Implements smart sync: insert new, keep existing unchanged, remove products no longer in catalog.
"""
import json
from typing import Dict, List
import requests


class SupabaseREST:
    """Minimal Supabase PostgREST helper for products table.

    Uses resolution=ignore-duplicates so we INSERT new products but never overwrite existing.
    delete_missing_for_source() removes products from this scraper's source that weren't
    found in the current run (discontinued, removed from catalog, etc.).
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

    def upsert_products(self, products: List[Dict], ignore_existing: bool = True) -> None:
        """Insert products into 'products' table. Uses ignore-duplicates to avoid overwriting existing rows.

        Args:
            products: List of product dicts with same keys across all items.
            ignore_existing: If True, use resolution=ignore-duplicates (insert only, skip existing).
                            If False, use resolution=merge-duplicates (update on conflict).
        """
        if not products:
            return

        # Deduplicate by (source, product_url) within this batch
        seen: Dict[str, Dict] = {}
        for p in products:
            source = p.get("source", "")
            product_url = p.get("product_url", "")
            key = f"{source}:{product_url}"
            if key and key != ":":
                seen[key] = p
        products = list(seen.values())

        # Normalize: PostgREST requires all objects in a request to have the same keys
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
            "price,sale,metadata,size,second_hand,image_embedding,info_embedding,additional_images"
        )

        chunk_size = 50  # Smaller chunks for large payloads (embeddings)
        for i in range(0, len(normalized), chunk_size):
            chunk = normalized[i : i + chunk_size]
            resp = self.session.post(endpoint, headers=headers, data=json.dumps(chunk), timeout=120)
            if resp.status_code not in (200, 201, 204):
                raise RuntimeError(f"Supabase upsert failed: {resp.status_code} {resp.text}")

    def delete_missing_for_source(self, source: str, current_ids: List[str]) -> int:
        """Delete products for this source that are not in current_ids (smart sync: remove discontinued).

        Only deletes rows where source=source and id not in current_ids.
        Returns number of rows deleted.
        """
        if current_ids is None:
            current_ids = []

        current_set = set(current_ids)

        url = f"{self.base_url}/rest/v1/products?source=eq.{source}&select=id"
        resp = self.session.get(url, timeout=60)
        resp.raise_for_status()

        all_ids = [r.get("id") for r in resp.json() if r.get("id") is not None]
        to_delete = [eid for eid in all_ids if eid not in current_set]

        deleted = 0
        for eid in to_delete:
            del_url = f"{self.base_url}/rest/v1/products?source=eq.{source}&id=eq.{eid}"
            del_resp = self.session.delete(del_url, timeout=60)
            if del_resp.status_code in (200, 204):
                deleted += 1
            else:
                raise RuntimeError(f"Supabase delete failed: {del_resp.status_code} {del_resp.text}")

        return deleted
