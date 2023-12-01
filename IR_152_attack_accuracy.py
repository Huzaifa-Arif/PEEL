import torch
import time
#from IR_152_PReLU import IR_64,IR152_PReLU
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from models_for_inversion import FaceNet64,IR152
from celebA_init import CelebADataset
from utilities import extract_resnet_info,extract_number,plot_channels,extract_conv_details,plot_image, extract_prelu_details
from PIL import Image
import os
import re
from torch.utils.data import DataLoader, SubsetRandomSampler

torch.manual_seed(0)



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
    



# def test(model, criterion=None, dataloader=None, recovered_loader=None, samples_tested=100, device='cpu'):
#     tf = time.time()
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)
#     model.eval()
#     loss, cnt, ACC, correct_top5, recovered_ACC = 0.0, 0, 0, 0, 0

#     with torch.no_grad():
#         # Using zip to iterate over both dataloaders simultaneously
#         for i, (img, recovered_img) in enumerate(zip(dataloader, recovered_loader)):
#             img, recovered_img = img.to(device), recovered_img.to(device)
            
#            # plot_image(img)
#             plot_image(recovered_img)
            
            
            
#             #import pdb;pdb.set_trace()
            

#             bs = img.size(0)
#             #iden = iden.view(-1)

#             # For labeled data
#             _, out_prob = model(img)
#             out_iden = torch.argmax(out_prob, dim=1).view(-1)
#            # print("Orignal Identity:", out_iden)
#             #ACC += torch.sum(iden == out_iden).item()

       

#             # For recovered data
#             _, recovered_out_prob = model(recovered_img)
#             recovered_out_iden = torch.argmax(recovered_out_prob, dim=1).view(-1)
#             print("Recovered Identity:",recovered_out_iden)
            
#             recovered_ACC += torch.sum(out_iden == recovered_out_iden).item()

#             cnt += bs
            
#             _, top5 = torch.topk(recovered_out_prob, 5, dim=1)
#             for ind, top5pred in enumerate(top5):
#                 if out_iden[ind] in top5pred:
#                     correct_top5 += 1
            
#             if i == samples_tested:
#                 break

#         #labeled_acc = ACC * 100.0 / cnt
#         recovered_labeled_top5_acc = correct_top5 * 100.0 / cnt
#         recovered_acc = recovered_ACC * 100.0 / cnt

#     return  recovered_labeled_top5_acc , recovered_acc


def img_process(img = None):
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    
    image = Image.open(img).convert("RGB")
    tensor_image = transform(image).unsqueeze(0)
    return tensor_image
    
    

# def calculate_accuracy(true_images_dir, recovered_images_dir, model):
#     true_labels = {}
#     correct_predictions = 0
#     total_predictions = 0
    
    
#     # Map each true image to its label
#     for file in os.listdir(true_images_dir):
#         label = get_true_label(file)
#         if label is not None:
#             true_labels[file.split('_')[2]] = label
    
    
    
    
#     # Predict and calculate accuracy
#     for file in os.listdir(true_images_dir):
#         true_label = true_labels.get(file.split('_')[2].split('.')[0])
#         if true_label is not None:
#            # plot_image(os.path.join(recovered_images_dir, file))
#            # import pdb;pdb.set_trace()
#             proc_imag = img_process(os.path.join(true_images_dir, file))
#             _,output = model(proc_imag)
#             predicted_label = torch.argmax(output[0].item())
            
            
            
#             print(predicted_label)
#             print(true_label)
#             if predicted_label == true_label:
#                 correct_predictions += 1
#             total_predictions += 1
            
            
    
    
#     # # Predict and calculate accuracy
#     # for file in os.listdir(recovered_images_dir):
#     #     true_label = true_labels.get(file.split('_')[2].split('.')[0])
#     #     if true_label is not None:
#     #        # plot_image(os.path.join(recovered_images_dir, file))
#     #        # import pdb;pdb.set_trace()
#     #         proc_imag = img_process(os.path.join(recovered_images_dir, file))
#     #         _,output = model(proc_imag)
#     #         predicted_label = torch.argmax(output)
            
            
            
#     #         print(predicted_label)
#     #         print(true_label)
#     #         if predicted_label == true_label:
#     #             correct_predictions += 1
#     #         total_predictions += 1

#     return correct_predictions / total_predictions if total_predictions else 0


def test(model, criterion=None, dataloader=None,recovered_dataloader = None,samples_tested = 100, labels_list = None,device='cpu'):
    tf = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
  #  
    model.eval()
    loss, cnt, ACC, correct_top5 = 0.0, 0, 0, 0
    with torch.no_grad():
        for i, ((img, iden),(img2,iden2)) in enumerate(zip(dataloader,recovered_dataloader)):
            #img, iden = img.to(device), iden.to(device)

            bs = img.size(0)
            #iden = iden.view(-1)
            
            _, out_prob = model(img)
            _, out_prob2 = model(img2)
            #import pdb;pdb.set_trace()
            
            
            #print("Out_identity_orig:,", out_prob.view(-1)[labels_list])
            #print("Out_identity_pred:,", out_prob2.view(-1)[labels_list])
            
            #print(out_prob.view(-1)[labels_list].shape)
            out_iden = torch.argmax(out_prob.view(-1)[labels_list])
            out_p_iden = torch.argmax(out_prob2.view(-1)[labels_list])
            
            ACC += torch.sum(out_iden == out_p_iden).item()

            _, top5 = torch.topk(out_prob2.view(-1)[labels_list], 5)
            
           # plot_image(img)
           # plot_image(img2)
            
            print("Predicted orignal", out_iden)
            print("Predicted_recreate", out_p_iden)
            print("Predicted iden",iden)
            
            for ind, top5pred in enumerate(top5):
               # import pdb;pdb.set_trace()
                if out_iden in top5pred:
                    correct_top5 += 1

            cnt += bs
            
            if(i == 100):
                break

    return ACC * 100.0 / cnt, correct_top5 * 100.0 / cnt


def get_true_label(file_name):
    # Extracts the true label from the file name
    match = re.search(r'_\d+_(\d+).png', file_name)
    return int(match.group(1)) if match else None

def main():
  

    
    ### Pretrained_model and its details
    checkpoint = torch.load('best_model_checkpoint_Prelu_152_IR_224_new.pth',map_location=torch.device('cpu') )
    train_loader = checkpoint['trainloader']
    
    
    # Create a new DataLoader with the same dataset but with a batch size of 1
    celebAloader = torch.utils.data.DataLoader(dataset=train_loader.dataset , batch_size=1, 
                                                    num_workers=train_loader.num_workers, 
                                                    collate_fn=train_loader.collate_fn, 
                                                    pin_memory=train_loader.pin_memory, 
                                                    drop_last=train_loader.drop_last)
    

    diff  = 1
    if(diff):
        eval_model = FaceNet64() #IR152()
        eval_checkpoint = torch.load('best_model_checkpoint_faceevolve_224_new.pth',map_location=torch.device('cpu') )
        eval_model.load_state_dict(eval_checkpoint['model_state_dict'])
      
    else:
        eval_model = IR152()
        eval_model.load_state_dict(checkpoint['model_state_dict'])
    
    
    ######## Create a loader on the reconstructed images #################
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    
    #proc =[]
   # proc.append(transforms.ToTensor())
    #proc.append(transforms.Lambda(crop))
    #proc.append(transforms.ToPILImage())
    #proc.append(transforms.Resize((224, 224)))
   # proc.append(transforms.ToTensor())
    #proc.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    # Create a composed transform using your custom steps
    #custom_transform = transforms.Compose(proc)
    
    
    
    reconstructed_file_name = 'recovered_images_celebA_IR152_small'#'recovered_images_celebA_IR152'#'recovered_images_celebA_IR152'
    
    recovered_images_files = get_sorted_file_list(reconstructed_file_name)
    input_images_files = get_sorted_file_list('input_images_celebA_IR152_small') # 'input_images_celebA_IR152'

    # Check if both directories have the same number of files
    assert len(recovered_images_files) == len(input_images_files), "Directories have different number of files."
    
    
    

    recovered_dataset = Recovered_ImageDataset(reconstructed_file_name, transform=transform, file_list=recovered_images_files)
    dataset = Recovered_ImageDataset('input_images_celebA_IR152', transform=transform, file_list=input_images_files)
    
    # Create dataloaders without shuffling
    recovered_dataloader = DataLoader(recovered_dataset, batch_size=1, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    
    # recovered_dataset = Recovered_ImageDataset('recovered_images_brighter', transform= transform)
    # recovered_dataloader = DataLoader(recovered_dataset, batch_size=1, shuffle=False)

    
   
    # dataset = Recovered_ImageDataset('input_images_celebA_IR152', transform= transform)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    
    #import pdb;pdb.set_trace()
    
    

    
    labels_list = []

    for _, labels in dataloader:
        labels_list.extend(labels.tolist())
    
    acc = test(eval_model, criterion=None, dataloader= dataloader,recovered_dataloader= recovered_dataloader,samples_tested = 100,labels_list = labels_list, device='cpu')
    #accuracy = calculate_accuracy('input_images_celebA_IR152', 'recovered_images_celebA_IR152', eval_model) 
    #print(f"Accuracy: {accuracy * 100:.2f}%")

    #acc = test(eval_model, criterion=None, dataloader=dataloader,recovered_loader =  dataloader,samples_tested = len(recovered_dataloader), device='cpu')
        
    #print(acc)
    print(f"Top-1 Accuracy{acc[0]}")
    print(f'Top-5 Accuracy{acc[1]}')
    #print("Unique Values:",len(all_labels))      
    #print("Unique Values:",len(list(set(all_labels))))     
            

if __name__ == '__main__':
    main()
































# def get_identity(output):
#     # This function should convert the output of the model to a predicted identity
#     # For example, it might return the index of the highest score
#     #import pdb;pdb.set_trace()
#     #iden = iden.view(-1)
#     _, out_prob = output
#     out_iden = torch.argmax(out_prob, dim=1).view(-1)
#     #import pdb;pdb.set_trace()
  
#     return out_iden
# def calculate_accuracy(true_labels, predicted_labels):
#     # Ensure the true and predicted labels have the same length
#     assert len(true_labels) == len(predicted_labels), "Mismatch in the number of labels"
    
#     # Count the number of correct predictions
#     correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    
#     # Calculate accuracy
#     accuracy = correct_predictions / len(true_labels)
#     return accuracy