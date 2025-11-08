import csv
import os
import json
import asyncio
import aiohttp
import urllib.request
from urllib.parse import urlparse

# Configuration
url_staging_template = ""
url_localhost_template = "http://localhost:8000/v2/check/?part={category}&category=bag-lv&has_probs=true&heatmap=false&iqa_check=false"
url_localhost_raiki = "http://localhost:8001/api/v1/check/?part={category}&category=bag-lv&model_version=v1_3"
MAX_CONCURRENT_REQUESTS = 2
TEMP_IMAGES_DIR = "temp_images"

def map_category(category):
    """Map LV_P09 to LV_P05, return other categories unchanged"""
    if category == "LV_P04_case4":
        category = "LV_P04_m2"
    elif category == "LV_P09":
        category = "LV_P05"
    else:
        category = category
    return category

async def download_image(session, url, filename):
    """Async download image from URL to local file"""
    try:
        async with session.get(url) as response:
            if response.status == 200:
                with open(filename, 'wb') as f:
                    f.write(await response.read())
                return filename
            else:
                print(f"Failed to download {url}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

async def request_url(session, url, image_path):
    """Async function to send image to API"""
    try:
        with open(image_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('checkfile', f, filename=os.path.basename(image_path))
            async with session.post(url, data=data) as response:
                return await response.json()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {"result": "failed", "error": str(e)}

async def process_row(session, row, row_index, semaphore):
    """Process a single CSV row with semaphore for concurrency control"""
    async with semaphore:
        category = row.get('category', '').strip()
        link_img = row.get('link_img', '').strip()
        
        # Create a copy of the row to add new columns
        result_row = row.copy()
        
        if not category or not link_img:
            print(f"Row {row_index}: Missing category or link_img")
            result_row['new_pred_label'] = 'missing_data'
            result_row['new_real_prob'] = 'missing_data'
            return result_row
        
        # Map category (LV_P09 -> LV_P05)
        mapped_category = map_category(category)
        
        # Build API URL with mapped category
        api_url = url_localhost_template.format(category=mapped_category)
        
        # Download image
        image_filename = f"temp_images/image_{row_index}_{os.path.basename(urlparse(link_img).path)}"
        downloaded_path = await download_image(session, link_img, image_filename)
        
        if not downloaded_path:
            print(f"Row {row_index}: Failed to download image")
            result_row['new_pred_label'] = 'download_failed'
            result_row['new_real_prob'] = 'download_failed'
            return result_row
        
        try:
            # Call API
            response_json = await request_url(session, api_url, downloaded_path)
            
            if response_json.get("result") == "failed":
                result_row['new_pred_label'] = "failed"
                result_row['new_real_prob'] = "failed"
            else:
                result_row['new_pred_label'] = response_json.get("label", "unknown")
                result_row['new_real_prob'] = response_json.get("real_prob", 0)
            
            print(f"Row {row_index}: Processed {category} -> {mapped_category} - {result_row['new_pred_label']}")
            
        except Exception as e:
            print(f"Row {row_index}: Error processing - {e}")
            result_row['new_pred_label'] = "error"
            result_row['new_real_prob'] = "error"
        
        # Clean up downloaded image
        try:
            os.remove(downloaded_path)
        except:
            pass
            
        return result_row

async def process_csv(input_csv_path, output_csv_path):
    """Main function to process CSV file"""
    # Create temp images directory
    os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)
    
    results = []
    
    # Read input CSV
    with open(input_csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        
        if not rows:
            print("No rows found in CSV file")
            return
        
        # Check if required columns exist
        if 'category' not in reader.fieldnames or 'link_img' not in reader.fieldnames:
            print("CSV must contain 'category' and 'link_img' columns")
            return
        
        print(f"Total rows to process: {len(rows)}")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        # Create aiohttp session with SSL verification disabled for staging
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Create tasks for all rows
            tasks = [
                process_row(session, row, i + 2, semaphore)  # i+2 for CSV row number (header + 1-indexed)
                for i, row in enumerate(rows)
            ]
            
            # Process all rows concurrently
            results = await asyncio.gather(*tasks)
    
    # Write output CSV with new columns
    if results:
        # Get all fieldnames (original + new columns)
        fieldnames = list(results[0].keys())
        
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nCompleted! Results saved to {output_csv_path}")
        print(f"Total processed: {len(results)} rows")
    
    # Clean up temp directory
    try:
        os.rmdir(TEMP_IMAGES_DIR)
    except:
        pass

if __name__ == "__main__":
    # Usage: python csv_inference_processor.py input.csv output.csv
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python csv_inference_processor.py <input_csv> <output_csv>")
        print("Example: python csv_inference_processor.py data.csv results.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    
    if not os.path.exists(input_csv):
        print(f"Input file {input_csv} does not exist")
        sys.exit(1)
    
    asyncio.run(process_csv(input_csv, output_csv))
