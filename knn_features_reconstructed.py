import torch
import os
import pickle
from celebA_init import CelebADataset
from models_for_inversion import FaceNet64,IR152
import re
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms


def get_sorted_file_list(directory):
    return sorted(os.listdir(directory))

class Recovered_ImageDataset(Dataset):
    def __init__(self, main_dir, transform, file_list=None):
        self.main_dir = main_dir
        self.transform = transform
        if file_list is None:
            self.all_imgs = sorted(os.listdir(main_dir))
        else:
            self.all_imgs = file_list

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        label = self.extract_label(self.all_imgs[idx])
        return tensor_image, label

    @staticmethod
    def extract_label(file_name):
        # Extracts the label from the file name using a regular expression
        match = re.search(r'_(\d+).png$', file_name)
        return int(match.group(1)) if match else None

def main():
    # File path to save or load label_features
    model = 'IR152' #'IR152'#'facenet'#'IR152'
    diff = 1
    if(model == 'IR152'):
      
        reconstruct_folder = 'recovered_images_celebA_IR152' 
        
        
        if(diff):
            preloaded_weights = 'best_model_checkpoint_faceevolve_224_new.pth' 
            classifier = FaceNet64()
            label_features_path = 'label_features_IR_152_CelebA_reconstructed_diff.pkl'
        else:
            preloaded_weights = 'best_model_checkpoint_Prelu_152_IR_224_new.pth'
            label_features_path = 'label_features_IR_152_CelebA_reconstructed.pkl'
            classifier = IR152()
    else:
        
        reconstruct_folder = 'recovered_images_celebA_FaceNet64' 
        
        
        if(diff):
            preloaded_weights = 'best_model_checkpoint_Prelu_152_IR_224_new.pth'
            label_features_path = 'label_features_FaceNet64_CelebA_reconstructed_diff.pkl'
            classifier = IR152()
          
        else:
             preloaded_weights = 'best_model_checkpoint_faceevolve_224_new.pth' 
             classifier = FaceNet64()
             label_features_path = 'label_features_FaceNet64_CelebA_reconstructed.pkl'
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    # Check if the label_features file already exists
    if not os.path.exists(label_features_path):
        
        
        checkpoint = torch.load(preloaded_weights, map_location=torch.device('cpu'))
        reconstructed_file_name = reconstruct_folder # 'recovered_images_celebA_IR152'
        recovered_images_files = get_sorted_file_list(reconstructed_file_name)
        recovered_dataset = Recovered_ImageDataset(reconstructed_file_name, transform=transform, file_list=recovered_images_files)
        recovered_dataloader = DataLoader(recovered_dataset, batch_size=1, shuffle=False)

        

        
        classifier.load_state_dict(checkpoint['model_state_dict'])

        label_features = {}

        for i, (batch_images, batch_labels) in enumerate(recovered_dataloader):
            _, out = classifier(batch_images)
            fea = classifier.activations['output']

            for label, feature in zip(batch_labels, fea):
                if label.item() not in label_features:
                    label_features[label.item()] = []
                label_features[label.item()].append(feature.detach().cpu().numpy())

        # Save label_features using pickle
        with open(label_features_path, 'wb') as f:
            pickle.dump(label_features, f)

    else:
        print(f"'{label_features_path}' already exists. Loading data.")
        # Open the pickle file and load data
        with open('label_features_IR_152_CelebA.pkl', 'rb') as file:
            data = pickle.load(file)

        #import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
