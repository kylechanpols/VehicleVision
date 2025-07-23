from VehicleVision.listing import Listing
from typing import List, Any

import numpy as np
import glob

import logging
import pandas as pd
import os 
from pathlib import Path
import joblib 
import traceback
import tqdm 
import argparse

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
logger = logging.getLogger(__file__)


def worker(ex: Listing):
    def column_wise_max(idx: int, matrix: List[List[Any]] | np.ndarray):
        return float(max(row[idx] for row in matrix if len(row) > 0))
    
    if not ex.price:
        logger.warning("uid missing")
        return None
    else:
        uid = ex.uid
    if not ex.price or ex.price == 0:
        logger.warning(str(uid)+": Price is 0, this listing will be ignored.")
        return None
    if not ex.text_features or ex.text_features == {}:
        logger.warning(str(uid)+": Price is 0, this listing will be ignored.")
        return None
    suv_idx = ex.clip_class_prompts.index('an SUV')
    convertible_idx = ex.clip_class_prompts.index('a convertible')
    sedan_idx = ex.clip_class_prompts.index('a sedan')
    coupe_idx = ex.clip_class_prompts.index('a coupe')
    hatchback_idx = ex.clip_class_prompts.index('a hatchback')
    window_damage_idx = ex.clip_condition_prompts.index('a car with damaged window')
    door_damage_idx = ex.clip_condition_prompts.index('a car with damaged door')
    bumper_damage_idx = ex.clip_condition_prompts.index('a car with damaged bumper')
    hood_damage_idx = ex.clip_condition_prompts.index('a car with damaged hood')
    excellent_cond_idx = ex.clip_condition_prompts.index('a car in excellent condition')
    average_cond_idx = ex.clip_condition_prompts.index('a car in average condition')
    
    return {
        'uid': uid,
        'price': ex.price,
        'suv_prob': column_wise_max(suv_idx, ex.clip_probs),
        'vert_prob': column_wise_max(convertible_idx, ex.clip_probs),
        'sedan_prob': column_wise_max(sedan_idx, ex.clip_probs),
        'coupe_prob': column_wise_max(coupe_idx, ex.clip_probs),
        'hatchback_prob': column_wise_max(hatchback_idx, ex.clip_probs),
        'window_damage_prob': column_wise_max(window_damage_idx, ex.clip_cond_probs),
        'door_damage_prob': column_wise_max(door_damage_idx, ex.clip_cond_probs),
        'bumper_damage_prob': column_wise_max(bumper_damage_idx, ex.clip_cond_probs),
        'hood_damage_prob': column_wise_max(hood_damage_idx, ex.clip_cond_probs),
        'excellent_cond_prob': column_wise_max(excellent_cond_idx, ex.clip_cond_probs),
        'average_cond_prob': column_wise_max(average_cond_idx, ex.clip_cond_probs),
        **ex.text_features
    }

def find_joblib_files(folder_path: str) -> List[str]:
    """
    Scan folder for all joblib files
    
    Args:
        folder_path: Path to the source folder
        
    Returns:
        List of joblib file paths
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    # Find all .joblib and .pkl files (common joblib extensions)
    joblib_files = []
    for extension in ['*.joblib', '*.pkl']:
        joblib_files.extend(folder.glob(extension))
    
    # Convert to strings and sort for consistent processing order
    joblib_files = sorted([str(f) for f in joblib_files])
    logger.info(f"Found {len(joblib_files)} joblib files in {folder_path}")
    
    return joblib_files

def load_joblib_file(filepath: str) -> List:
    """
    Load a single joblib file and return the list of CoolData instances
    
    Args:
        filepath: Path to the joblib file
        
    Returns:
        List of CoolData instances
    """
    with open(filepath, 'rb') as f:
        data_list = joblib.load(f)
        logger.info(f"Loaded {len(data_list)} items from {filepath}")
        return data_list
    
def process_all_joblibfiles(source_folder: str, n_jobs: int = -1, 
                           backend: str = 'threading', batch_size: int = 'auto',
                           verbose: int = 0) -> pd.DataFrame:
    """
    Main function to process all joblib files with joblib parallel processing
    
    Args:
        source_folder: Path to folder containing joblib files
        n_jobs: Number of jobs to run in parallel (-1 uses all processors)
        backend: Backend for parallel processing ('threading', 'multiprocessing', 'loky')
        batch_size: Size of batches ('auto' or integer)
        verbose: Verbosity level for joblib
        
    Returns:
        pd.DataFrame: Combined results from all processed CoolData instances
    """
    logger.info(f"Starting processing with n_jobs={n_jobs}, backend={backend}")
    
    # Step 1: Find all joblib files
    joblib_files = find_joblib_files(source_folder)
    
    if not joblib_files:
        logger.warning("No joblib files found")
        return pd.DataFrame()
    
    # Step 2: Load all CoolData instances from all files
    all_listings = []
    
    for filepath in tqdm.tqdm(joblib_files, desc="Loading Joblib files"):
        cooldata_list = load_joblib_file(filepath)
        all_listings.extend(cooldata_list)
    
    logger.info(f"Total Listings to process: {len(all_listings)}")
    
    if not all_listings:
        logger.warning("No Listings found")
        return pd.DataFrame()
    
    # Step 3: Process all instances with joblib Parallel
    logger.info("Starting parallel processing with joblib...")
    
    try:
        results = joblib.Parallel(
            n_jobs=n_jobs,
            backend=backend,
            batch_size=batch_size,
            verbose=verbose
        )(joblib.delayed(worker)(item) for item in tqdm.tqdm(all_listings, desc="Processing items:"))
        
        logger.info(f"Parallel processing completed, got {len(results)} results")
        
    except Exception as e:
        logger.error(f"Error during parallel processing: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    # TODO: Optimize, memory bottleneck
    clean_results = []
    for r in results:
        if r is not None:
            clean_results.append(r)
       
    # Step 5: Convert results to DataFrame
    if results:
        df = pd.DataFrame(clean_results)
        logger.info(f"Created DataFrame with shape {df.shape}")
        return df
    else:
        logger.warning("No successful results to convert to DataFrame")
        return pd.DataFrame()

if __name__ == "__main__":
    
    # Initialize parser
    parser = argparse.ArgumentParser(description = "Arguments for Craiglists Scraper")
    # Adding optional argument
    parser.add_argument("-s", "--src", help = "Path to Source Folder containing Listings")
    parser.add_argument("-f", "--file", help = "Path to output file")
    args = parser.parse_args()

    df = process_all_joblibfiles(args.src, verbose = 1)
    df.to_csv(args.file)