import os
import requests
from urllib.parse import urlparse
import time
from pathlib import Path

def download_images(image_urls,
        download_folder="imgs",
        delay=1,
        large=True):
    """
    Download a list of images to a local folder.
    
    Args:
        image_urls (list): List of image URLs to download
        download_folder (str): Local folder to save images (default: "imgs")
        delay (int): Delay between downloads in seconds (default: 1)
        large (bool): Replace links with enlarged size (1200x900)? If False, use default thumbnail size (600x450)
    
    Returns:
        dict: Results with successful and failed downloads
    """
    
    # Create download folder if it doesn't exist
    Path(download_folder).mkdir(parents=True, exist_ok=True)
    
    results = {"successful": [], "failed": []}
    
    for i, url in enumerate(image_urls, 1):
        try:
            if large:
                url = url.replace("600x450","1200x900")

            print(f"Downloading image {i}/{len(image_urls)}: {url}")
            
            # Send GET request with headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Get filename from URL or create one
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            
            # If no filename in URL, create one based on index
            if not filename or '.' not in filename:
                filename = f"image_{i}.jpg"
            
            # Full path for saving
            filepath = os.path.join(download_folder, filename)
            
            # Handle duplicate filenames
            counter = 1
            original_filepath = filepath
            while os.path.exists(filepath):
                name, ext = os.path.splitext(original_filepath)
                filepath = f"{name}_{counter}{ext}"
                counter += 1
            
            # Save the image
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            results["successful"].append({"url": url, "filepath": filepath})
            print(f"✓ Saved: {filepath}")
            
        except requests.exceptions.RequestException as e:
            results["failed"].append({"url": url, "error": str(e)})
            print(f"✗ Failed to download {url}: {e}")
        
        except Exception as e:
            results["failed"].append({"url": url, "error": str(e)})
            print(f"✗ Error processing {url}: {e}")
        
        # Add delay between downloads to be respectful
        if i < len(image_urls):
            time.sleep(delay)
    
    return results