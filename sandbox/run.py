# A simple run script to execute craigslist search, parse through all listings and save the results.
import craigslistscraper as cs
import argparse
import logging
import os
from pathlib import Path
import os 
import joblib
from VehicleVision.listing import Listing

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Get your logger
logger = logging.getLogger("vv.cs.run")

# Initialize parser
parser = argparse.ArgumentParser(description = "Arguments for Craiglists Scraper")
# Adding optional argument
parser.add_argument("-q", "--query", help = "Search Query")
parser.add_argument("-c", "--city", help = "City")
parser.add_argument("-d", "--dir", help = "Download Directory")
parser.add_argument("-f", "--filename", help = "file name")
parser.add_argument("-i", "--imgdir", help = "Download Directory for Images")
parser.add_argument("-b", "--downloadimage", default=True, help = "Download Image Boolean")
args = parser.parse_args()

logging.info(f"Processing:\nquery={args.query}\ncity={args.city}\ndir={args.dir}")

search = cs.Search(
    query = args.query,
    city = args.city,
    category = "cto"
)
status = search.fetch()
if status != 200:
    raise Exception(f"Unable to fetch search with status <{status}>.")


parent_folder = Path(os.path.join(os.getcwd(), Path(args.imgdir)))
listings = []
# Create listings (save images to )
for i, ex in enumerate(search.ads):
    try:
        status=ex.fetch()
        if status != 200:
            logger.warning(f"Unable to fetch ad '{ex.title}' with status <{status}>.")
        listings.append(Listing.from_cs(ex, images_main_path = parent_folder, download_images_bool=args.downloadimage))
    except ValueError:
        logger.warning(f"listing {i} has no images and thus cannot be processed, skipping..")
logging.info(f"Processed {len(listings)} listings")
[li.process_images() for li in listings]
logging.info(f"Processed images for {len(listings)} listings")
[li.parse_description() for li in listings]
logging.info(f"Processed textual descriptions for {len(listings)} listings")

# Save to output directory
output_directory = Path(os.path.join(os.getcwd(), Path(args.dir, args.filename+".joblib")))
with open(output_directory, "wb") as f:
    joblib.dump(listings,f)

logging.info(f"All listings have been processed!")

