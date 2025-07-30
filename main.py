import logging
from pathlib import Path
from fastai.vision.all import load_learner
from utils import check_internet
from downloader import ImageDatasetDownloader
from trainer import BirdClassifierTrainer

logging.basicConfig(level=logging.INFO)
SINGLE_IMAGE = Path("single_image")
MODEL_DIR = Path("bird_or_not")
MODEL_FILE = MODEL_DIR / "bird_model.pkl"

def main():
    logging.info("Starting the bird classifier pipeline...")

    if MODEL_FILE.exists():
        logging.info(f"Found existing model at {MODEL_FILE}. Loading for inference only.")
        learn = load_learner(MODEL_FILE)
        try:
            prediction = learn.predict(SINGLE_IMAGE / 'bird.jpg')
            logging.info(f"Prediction: {prediction}")
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
        return

    try:
        check_internet()
    except Exception as e:
        logging.error(f"Internet check failed: {e}")
        return

    SINGLE_IMAGE.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    downloader = ImageDatasetDownloader()
    try:
        downloader.download_single_image('bird photos', SINGLE_IMAGE / 'bird.jpg')
        downloader.download_single_image('forest photos', SINGLE_IMAGE / 'forest.jpg')
        downloader.prepare_dataset()
        downloader.remove_bad_images()
    except Exception as e:
        logging.error(f"Data preparation failed: {e}")
        return

    trainer = BirdClassifierTrainer()
    try:
        trainer.build_dataloaders()
        trainer.train(epochs=3)
        trainer.save_model(MODEL_FILE)
        logging.info(f"Model trained and saved at {MODEL_FILE}")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return

    try:
        prediction = trainer.predict(SINGLE_IMAGE / 'bird.jpg')
        logging.info(f"Prediction: {prediction}")
    except Exception as e:
        logging.error(f"Prediction failed: {e}")

    logging.info("All steps finished successfully.")

if __name__ == "__main__":
    main()
