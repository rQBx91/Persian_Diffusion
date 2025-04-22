import os
from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms
from torchvision.transforms import Compose

# Define image transformations
transform = Compose([
    transforms.Resize((32, 32)),  # Resize images to 32z32
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

# Define transforms
def apply_transforms(examples):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples

# Custom Dataset class for local files
class CustomImageDataset(TorchDataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return {"image": image}

def create_dataset_dict(folder_path):
    # List all image files in the folder
    image_files = sorted([os.path.join(folder_path, fname) for fname in os.listdir(folder_path)], key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    # Split into train and test sets
    train_files = image_files[:70000]
    test_files = image_files[70000:]

    # Create torch datasets
    train_dataset = CustomImageDataset(train_files)
    test_dataset = CustomImageDataset(test_files)

    # Convert to dataset
    train_dataset_hf = Dataset.from_generator(lambda: (data for data in train_dataset))
    test_dataset_hf = Dataset.from_generator(lambda: (data for data in test_dataset))

    # Create DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset_hf,
        "test": test_dataset_hf
    })

    return dataset_dict

# Save dataset to disk
def save_dataset_dict(dataset_dict, save_path):
    dataset_dict.save_to_disk(save_path)

# Load dataset from disk
def load_dataset_dict(save_path):
    return load_from_disk(save_path)


# Data loader function 
def data_loader(batch_size=64):
    folder_path = 'datasets/persian'

    #mode = 1 # Create
    mode = 0 # Load

    dataset_dict = None

    # Create dataset from local files and save 
    if mode == 1:
        dataset_dict = create_dataset_dict(folder_path)
        save_dataset_dict(dataset_dict, 'datasets/persian_dataset')

    # Load dataset from disk
    if mode == 0:
        dataset_dict = load_dataset_dict('datasets/persian_dataset')

    # Assign dataset 
    dataset = dataset_dict
    batch_size = batch_size
   
    # Transform dataset
    transformed_dataset = dataset.with_transform(apply_transforms)
   
    # Create dataloader
    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

    return dataloader

