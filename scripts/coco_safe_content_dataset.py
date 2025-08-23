import os
import json
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

COCO_IMG_DIR = "D:/ProjetStage/ProjetStage/coco_data/train2017"
COCO_ANN_PATH = "D:/ProjetStage/ProjetStage/coco_data/annotations/instances_train2017.json"

SAFE_CLASS_IDX = 0  # Assuming class 0 is 'safe_content'
NUM_CLASSES = 7

class CocoSafeContentDataset(Dataset):
    def __init__(self, img_dir, ann_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(ann_path, 'r') as f:
            self.anns = json.load(f)
        self.imgs = {img['id']: img for img in self.anns['images']}
        self.safe_img_ids = self._get_safe_img_ids()

    def _get_safe_img_ids(self):
        # Images with no person, violence, or adult content annotations
        unsafe_cats = [1, 2, 3, 4, 5, 6]  # Map to your violation categories
        unsafe_img_ids = set()
        for ann in self.anns['annotations']:
            if ann['category_id'] in unsafe_cats:
                unsafe_img_ids.add(ann['image_id'])
        all_img_ids = set(self.imgs.keys())
        safe_img_ids = list(all_img_ids - unsafe_img_ids)
        return safe_img_ids

    def __len__(self):
        return len(self.safe_img_ids)

    def __getitem__(self, idx):
        img_id = self.safe_img_ids[idx]
        img_info = self.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = SAFE_CLASS_IDX
        return image, label

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CocoSafeContentDataset(COCO_IMG_DIR, COCO_ANN_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Safe content images: {len(dataset)}")
    # Example: Iterate and show batch shapes
    for images, labels in dataloader:
        print(images.shape, labels.shape)
        break
