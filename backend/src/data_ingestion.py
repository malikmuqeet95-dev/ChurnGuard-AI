from pathlib import Path
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "telco_churn_raw.csv"

def download_data(url: str = RAW_DATA_URL, save_path: str = RAW_DATA_PATH):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_path = Path(save_path)

    if save_path.exists():
        logging.info(f"Raw dataset already exists at {save_path}. Skipping download.")
        return str(save_path)
    
    logging.info(f"Downloading raw dataset from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    save_path.write_bytes(response.content)
    logging.info(f"Raw dataset successfully saved to {save_path}")
    return str(save_path)

if __name__ == "__main__":
    download_data()
