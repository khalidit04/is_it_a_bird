from pathlib import Path
from utils import check_internet
from downloader import ImageDatasetDownloader
from trainer import BirdClassifierTrainer
import logging

logging.basicConfig(level=logging.INFO)

SINGLE_IMAGE = "single_image"

def main():
    try:
        check_internet()
    except Exception as e:
        logging.error(f"Internet check failed: {e}")
        return

    downloader = ImageDatasetDownloader()

    # Ensure single_image folder exists
    Path(SINGLE_IMAGE).mkdir(parents=True, exist_ok=True)

    try:
        downloader.download_single_image('bird photos', Path(SINGLE_IMAGE) / 'bird.jpg')
        downloader.download_single_image('forest photos', Path(SINGLE_IMAGE) / 'forest.jpg')
    except Exception as e:
        logging.error(f"Failed downloading sample images: {e}")
        return

    try:
        downloader.prepare_dataset()
        downloader.remove_bad_images()
    except Exception as e:
        logging.error(f"Dataset preparation failed: {e}")
        return

    trainer = BirdClassifierTrainer()
    try:
        trainer.build_dataloaders()
        trainer.train(epochs=3)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return

    try:
        prediction = trainer.predict(str(Path(SINGLE_IMAGE) / 'bird.jpg'))
        logging.info(f"Prediction: {prediction}")
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return

    logging.info("All steps finished successfully.")


if __name__ == "__main__":
    main()
