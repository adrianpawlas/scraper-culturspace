# Cultur Space Fashion Scraper

A comprehensive scraper for Cultur Space fashion store that extracts product information, generates image embeddings, and stores data in Supabase.

## Features

- **Complete Product Extraction**: Scrapes all products from the Cultur Space shop
- **Image Embeddings**: Generates 768-dimensional embeddings using Google SigLIP model
- **Multi-Currency Pricing**: Extracts prices in EUR and converts to multiple currencies
- **Comprehensive Metadata**: Extracts product descriptions, sizes, materials, and care instructions
- **Supabase Integration**: Automatically stores all data in your Supabase database
- **Error Handling**: Robust error handling and logging

## Requirements

- Python 3.8+
- Supabase account and database
- Internet connection for scraping and downloading images

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Database Schema

The scraper expects a Supabase table named `products` with the following structure:

```sql
create table public.products (
  id text not null,
  source text null,
  product_url text null,
  affiliate_url text null,
  image_url text not null,
  brand text null,
  title text not null,
  description text null,
  category text null,
  gender text null,
  search_tsv tsvector null,
  created_at timestamp with time zone null default now(),
  metadata text null,
  size text null,
  second_hand boolean null default false,
  embedding public.vector null,
  country text null,
  compressed_image_url text null,
  tags text[] null,
  search_vector tsvector null,
  title_tsv tsvector null,
  brand_tsvector tsvector null,
  description_tsvector tsvector null,
  other text null,
  price text null,
  sale text null,
  constraint products_pkey primary key (id),
  constraint products_source_product_url_key unique (source, product_url)
) TABLESPACE pg_default;
```

## Configuration

The scraper is pre-configured with your Supabase credentials. If you need to change them, edit the `SUPABASE_URL` and `SUPABASE_KEY` variables in the scripts, or set the environment variables `SUPABASE_URL` and `SUPABASE_KEY` (used by GitHub Actions).

**Writing to the `products` table** requires a key that can insert/update rows. The scraper prefers `SUPABASE_SERVICE_ROLE_KEY` (bypasses RLS). If that is not set, it uses `SUPABASE_KEY`. The anon key often cannot insert into `products` because of row-level security; use the **service_role** key from Supabase Dashboard → Project Settings → API → `service_role` (secret).

## GitHub Actions (daily + manual)

The workflow [`.github/workflows/scrape.yml`](.github/workflows/scrape.yml) runs the full scraper:

- **Schedule**: every day at **00:00 UTC**
- **Manual run**: go to [Actions → Cultur Space Scraper](https://github.com/adrianpawlas/scraper-culturspace/actions) and click **Run workflow**

**Required secrets** (repo **Settings → Secrets and variables → Actions**):

| Name                           | Value |
|--------------------------------|-------|
| `SUPABASE_URL`                 | Your Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY`    | **Recommended.** Service role key (Dashboard → Project Settings → API → `service_role`). Needed for inserting into `products` when RLS is enabled. |
| `SUPABASE_KEY`                | Fallback if `SUPABASE_SERVICE_ROLE_KEY` is not set (e.g. anon key; only works if RLS allows that role to insert). |

Set `SUPABASE_SERVICE_ROLE_KEY` so scheduled and manual runs can write to the database. If a run fails, the **scraper.log** is uploaded as an artifact for that run.

## Usage

### Test Run (Recommended)

Before running the full scraper, test it with a few products:

```bash
python culturspace_scraper_simple.py
```

This will test the scraper with 2 sample products and verify that basic scraping works correctly.

### Full Scrape with Embeddings

Run the complete scraper with image embeddings:

```bash
python run_full_scraper.py
```

Or run directly:

```bash
python culturspace_scraper.py
```

The scraper will:
1. Extract all product URLs from the shop pages (2 pages total)
2. Scrape each product page for details using Selenium
3. Download product images and generate 768-dimensional embeddings using Google SigLIP
4. Store everything in Supabase with proper error handling

## What Gets Extracted

### Required Fields (Always Present)
- **source**: "scraper"
- **brand**: "Cultur Space"
- **product_url**: Full URL of the product page
- **image_url**: Direct URL to product image
- **title**: Product name (cleaned and normalized)
- **gender**: "MAN" (all products are men's wear)
- **second_hand**: false (brand new products)
- **embedding**: 768-dimensional vector from Google SigLIP model
- **price**: Multi-currency price string (e.g., "109,00EUR,117,72USD,93,65GBP,...")
- **sale**: Sale price if available, null otherwise

### Optional Fields (When Available)
- **description**: Product description text
- **size**: Available sizes as comma-separated string
- **metadata**: JSON with additional info (materials, care instructions, etc.)
- **created_at**: Database insertion timestamp

### Data Quality
- **Products Found**: ~26 unique products across 2 pages
- **Image Embeddings**: High-quality 768-dim vectors using SigLIP
- **Price Conversion**: Automatic conversion to 11+ currencies
- **Error Handling**: Robust handling of missing data and network issues

## Output

- **Console**: Progress updates and success/failure messages
- **Log File**: `scraper.log` with detailed logging
- **Database**: All product data stored in Supabase

## Error Handling

The scraper includes comprehensive error handling:
- Network timeouts and retries
- Missing data graceful handling
- Selenium WebDriver management
- Image processing error recovery
- Database connection issues

## Performance & Technical Details

- **Total Products**: ~26 unique products to scrape
- **Image Processing**: SigLIP model generates 768-dimensional embeddings
- **GPU Support**: Automatic GPU detection for faster embedding generation
- **Rate Limiting**: 1-second delays between requests to be respectful
- **Error Recovery**: Continues processing even if individual products fail
- **Memory Management**: Automatic cleanup of temporary image files
- **Database**: Upsert operations prevent duplicates
- **Logging**: Comprehensive logging to `scraper.log`

## Troubleshooting

### Common Issues

1. **WebDriver Issues**: Make sure Chrome is installed and up to date
2. **Network Errors**: Check your internet connection
3. **Supabase Errors**: Verify your credentials and database schema
4. **Memory Issues**: For large scrapes, ensure sufficient RAM

### Logs

Check `scraper.log` for detailed error information and debugging.

## Legal and Ethical Considerations

- This scraper is designed to be respectful to the website
- Includes appropriate delays between requests
- Only scrapes publicly available product information
- Does not attempt to bypass any restrictions

## Support

If you encounter issues:
1. Run the test script first to isolate problems
2. Check the log files for error details
3. Verify your Supabase configuration
4. Ensure all dependencies are installed correctly