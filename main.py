import logging
import argparse
from pathlib import Path

from fastai.vision.all import load_learner

from utils import check_internet, delete_folder
from downloader import ImageDatasetDownloader
from trainer import BirdClassifierTrainer

# --------- CONSTANTS & CONFIG ---------
SINGLE_IMAGE_DIR = Path("single_image")
MODEL_DIR = Path("bird_or_not")
MODEL_FILE = MODEL_DIR / "bird_model.pkl"
TEMP_IMAGE = Path("temp_input.jpg")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# --------- ARGUMENTS ---------
def parse_args():
    parser = argparse.ArgumentParser(description='Bird image classifier')
    parser.add_argument('--img-url', type=str, help='Image URL to classify')
    parser.add_argument('--train', action='store_true', help='Force model training, even if model exists')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs for training (default: 3)')
    return parser.parse_args()

# --------- INFERENCE ---------
def predict_image(model_file, img_path):
    learn = load_learner(model_file)
    # pred = learn.predict(img_path)
    pred_class, pred_idx, pred_probs = learn.predict(img_path)
    print(f"Predicted class: {pred_class}, Index: {pred_idx}, Probabilities: {pred_probs}")
    if pred_probs[pred_idx] < 0.5:
        result = None  # or 'unknown'
    else:
        result = pred_class
    logging.info(f"Prediction for '{img_path}': {result}")
    return result

# --------- TRAINING WORKFLOW ---------
def train_and_export_model(downloader, trainer, epochs):
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        # Dataset for training
        downloader.prepare_dataset()
        downloader.remove_bad_images()
        # Training and export
        trainer.build_dataloaders()
        trainer.train(epochs=epochs)
        trainer.save_model(MODEL_FILE)
        logging.info(f"Model trained and saved at {MODEL_FILE}")
    except Exception as e:
        logging.error(f"Full training pipeline failed: {e}")
        raise

# --------- IMAGE DOWNLOAD UTILITY ---------
def fetch_image_for_inference(downloader, img_url):
    try:
        return downloader.download_image_from_url(img_url, TEMP_IMAGE)
    except Exception as e:
        logging.error(f"Failed to download inference image: {e}")
        raise

# --------- MAIN ---------
def main():
    args = parse_args()

    # Connectivity (for all cases needing download/model build)
    try:
        check_internet()
    except Exception as e:
        logging.error(f"Internet check failed: {e}")
        return

    downloader = ImageDatasetDownloader()
    trainer = BirdClassifierTrainer()

    if args.img_url:
        try:
            img_to_check = fetch_image_for_inference(downloader, args.img_url)
        except Exception:
            return

    # Decide between train/load for handling the model
    if not MODEL_FILE.exists() or args.train:
        logging.info(
            f"{'Training requested' if args.train else 'No model found; training now.'}"
        )
        try:
            train_and_export_model(downloader, trainer, epochs=args.epochs)
            # Delete all training images except the exported model
            for sub in MODEL_DIR.iterdir():
                if sub.is_dir():
                    delete_folder(sub)
                elif sub.is_file() and sub != MODEL_FILE:
                    sub.unlink()
        except Exception:
            return

    # Always predict (model must exist now)
    try:
        predict_image(MODEL_FILE, img_to_check)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")

    # Clean temp file if needed
    if args.img_url and TEMP_IMAGE.exists():
        TEMP_IMAGE.unlink()

if __name__ == "__main__":
    main()
