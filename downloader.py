from ddgs import DDGS
from fastcore.all import L
from fastdownload import download_url
import requests
from fastai.vision.all import *
from config import CLASSES, DOWNLOAD_ROOT, MAX_IMAGES_PER_CLASS, RESIZE_TO
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageDatasetDownloader:
    def __init__(self, classes=CLASSES, root=DOWNLOAD_ROOT):
        self.classes = classes
        self.root = Path(root)

    def search_images(self, keywords, max_images=MAX_IMAGES_PER_CLASS):
        logger.info(f"Searching images for '{keywords}'...")
        results = L(DDGS().images(keywords, max_results=max_images)).itemgot('image')
        logger.info(f"Found {len(results)} images for '{keywords}'")
        return results

    def download_image_from_url(self, img_url, dest_path):
        logging.info(f"Downloading image from: {img_url}")
        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Downloaded image to: {dest_path}")
            return dest_path
        except Exception as e:
            logging.error(f"Failed to download image from url '{img_url}': {e}")
            raise

    def download_single_image(self, query, dest, retries=3):
        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, retries+1):
            try:
                url = self.search_images(query, 1)[0]
                download_url(url, dest_path, show_progress=True)
                logger.info(f"Downloaded '{query}' image to {dest_path}")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt} to download '{query}' failed: {e}")
                if attempt == retries:
                    logger.error(f"Failed to download '{query}' image after {retries} attempts.")
                    raise

    def prepare_dataset(self):
        for cls in self.classes:
            dest = self.root / cls
            dest.mkdir(parents=True, exist_ok=True)
            try:
                urls = self.search_images(f'{cls} photo')
                logger.info(f"Downloading images for '{cls}'...")
                download_images(dest, urls=urls)
                time.sleep(5)  # Be kind to the servers
                resize_images(dest, max_size=RESIZE_TO, dest=dest)
                logger.info(f"Images for '{cls}' downloaded and resized.")
            except Exception as e:
                logger.error(f"Error downloading images for '{cls}': {e}")
                # You can decide whether to raise here or continue with other classes

    def remove_bad_images(self):
        failed = verify_images(get_image_files(self.root))
        for f in failed:
            f.unlink()
        logger.info(f"Removed {len(failed)} corrupted images.")
