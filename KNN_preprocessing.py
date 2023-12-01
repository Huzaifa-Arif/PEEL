import torch
import os
import pickle
from celebA_init import CelebADataset
from models_for_inversion import IR152,FaceNet64

def main():
    # File path to save or load label_features
    label_features_path = 'label_features_IR_152_CelebA.pkl'

    # Check if the label_features file already exists
    if not os.path.exists(label_features_path):
        checkpoint = torch.load('best_model_checkpoint_Prelu_152_IR_224_new.pth', map_location=torch.device('cpu'))
        train_loader = checkpoint['trainloader']

        # Create a new DataLoader
        celebAloader = torch.utils.data.DataLoader(dataset=train_loader.dataset, batch_size=64, 
                                                   num_workers=train_loader.num_workers, 
                                                   collate_fn=train_loader.collate_fn, 
                                                   pin_memory=train_loader.pin_memory, 
                                                   drop_last=train_loader.drop_last)

        classifier = IR152()
        classifier.load_state_dict(checkpoint['model_state_dict'])

        label_features = {}

        for i, (batch_images, batch_labels) in enumerate(celebAloader):
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

if __name__ == '__main__':
    main()
