from fastai.vision.all import *
from config import DOWNLOAD_ROOT, SEED, BATCH_SIZE

class BirdClassifierTrainer:
    def __init__(self, data_dir=DOWNLOAD_ROOT):
        self.data_dir = Path(data_dir)
        self.dls = None
        self.learn = None

    def build_dataloaders(self):
        print("Building DataLoaders...")
        self.dls = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=SEED),
            get_y=parent_label,
            item_tfms=[Resize(192, method='squish')]
        ).dataloaders(self.data_dir, bs=BATCH_SIZE)
        print("DataLoaders ready.")

    def train(self, epochs=3):
        print("Training model...")
        self.learn = vision_learner(self.dls, resnet18, metrics=error_rate)
        self.learn.fine_tune(epochs)
        print("Training complete.")

    def predict(self, img_path):
        print(f"Predicting on {img_path}...")
        is_bird,_,probs = self.learn.predict(PILImage.create(img_path))
        print(f"Prediction: {is_bird}, Probability: {probs[0]:.4f}")
        return is_bird, probs
