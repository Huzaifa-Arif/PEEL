### Brighten and view the image
            
         
            
            # rec_img_pil = tensor_to_pil(image)  # Convert to PIL image

            # # Enhance the brightness
            # brightened_img_pil = enhance_brightness(rec_img_pil, factor=0.5)  # Adjust factor as needed
            
            # # Optionally convert back to tensor
            # brightened_img_tensor = pil_to_tensor(brightened_img_pil)
            
            # # Visualization
            # plt.figure(figsize=(12, 6))
            # plt.subplot(1, 2, 1)
            # plt.imshow(rec_img_pil)
            # plt.title("Original Image")
            # plt.axis('off')
            
            # plt.subplot(1, 2, 2)
            # plt.imshow(brightened_img_pil)
            # plt.title("Brightened Image")
            # plt.axis('off')
            
            # plt.show()
            
            #import pdb;pdb.set_trace()
            
        
        # transform = transforms.Compose([  # Resize the image to 224x224
        #     transforms.ToTensor()           # Convert the image to a PyTorch tensor
        # ])
        
        # # Read the image
        # image = Image.open('recons_1').convert('RGB')
        # image_tensor = transform(image)
        # orignal_img = image_tensor.unsqueeze(0)
        
        
        
        
        #eval_model = IR152_PReLU()
        #eval_model.load_pretrained_weights(checkpoint_path='best_model_checkpoint_Prelu_152_IR_224.pth')
        
        #acc = test(eval_model, criterion= nn.CrossEntropyLoss(), dataloader=celebAloader, device='cpu')
        #print(f'Accuracy:{acc}')
        #acc_count = 0
       
        
        # true_labels = []
        # predicted_labels = []

        # for i, (img, iden) in enumerate(celebAloader): 
            
            
        #     print(f'Batch Number {i} out of Batches {len(celebAloader)}')
        #     input_image = img
        #     _, out_org = eval_model(img)
    
        # #    # Assuming out_org is a tensor or an array and you need the class index
        # #   # import pdb;pdb.set_trace()
        #     # Use argmax to get the predicted labels for the batch
        #     batch_predicted_labels = out_org.argmax(dim=1)
        
        #     # Extend the true labels and predicted labels lists
        #     true_labels.extend(iden.tolist())
        #     predicted_labels.extend(batch_predicted_labels.tolist())


        # # Now calculate the accuracy
        # accuracy = calculate_accuracy(true_labels, predicted_labels)
        # print(f"Model accuracy: {accuracy * 100:.2f}%")

    
        # ##### iterate through the images
        # for i, (img, iden) in enumerate(celebAloader):
            
        #     input_image = img
        #     _,out_org = eval_model(img)
            
            
            
            
            
            #if(iden == 334 or iden == 225):
             #   plot_image(img)
              #  print(iden)
            #import pdb;pdb.set_trace()
            #print(iden)
            
            # rec_img = Image.open('test_image_SwinIR_large.png').convert('RGB')


            # # Assuming you've resized the image using PIL
            # rec_img = rec_img.resize((112, 112)) 
            # # Convert to tensor
            # to_tensor = transforms.ToTensor()
            # rec_img = to_tensor(rec_img).unsqueeze(0)
   
            #import pdb;pdb.set_trace()
            
            
           # import pdb;pdb.set_trace()
            #import pdb;pdb.set_trace()
            # batch_size = 5
            
            # # Enhance the brightness
            
            # #enhancer = ImageEnhance.Brightness(img.squeeze(0))
            # # brightened_img = enhancer.enhance(2) 
            
            # #_,out_rec = model()
            # _,out_org  = eval_model(img)
            # _,out_rec  = eval_model(rec_img)
            
            
            # total_count = 0
            # acc_count = 0
            # for i in range(batch_size):
            #     #import pdb;pdb.set_trace()
            #     prob_org = out_org.detach().numpy()[i]
            #     prob_rec = out_rec.detach().numpy()[i]
            #     #print("The error:",torch.norm(rec_img[i] - img[i]))
            #     #print("The iden true is",iden[i]);
            #     #print("The iden from recovered is",torch.argmax(out_rec[i]));
            #     #print("The iden from model is",torch.argmax(out_org[i]));
            
            #     #plot_image(input_image[i])
            #     #plot_image(rec_img[i])
                
            #     top_5_org = np.argsort(prob_org)[-5:][::-1] 
            #     top_5_rec = np.argsort(prob_rec)[-5:][::-1] 
            #     #import pdb;pdb.set_trace()
            #     common_values = np.intersect1d(top_5_org,top_5_rec)
                
            #     if common_values.size > 0: 
            #         acc_count += 1
                
            #     total_count +=1
                
            #    # print("top 5 org,:", top_5_org)
            #    # print("top 5 rec:", top_5_rec)
                
            
            # Acc = acc_count/total_count * 100
            # print("Batch _accuracy:",Acc)    
            # #break
               
            
            #
            
            
            
            
# def increase_rate(p):
#     p*=1.1
#     return p

# def tensor_to_pil(tensor):
#     """Convert a PyTorch tensor to a PIL Image."""
#     return transforms.ToPILImage()(tensor.squeeze(0))

# def pil_to_tensor(pil_img):
#     """Convert a PIL Image to a PyTorch tensor."""
#     return transforms.ToTensor()(pil_img)

# def enhance_brightness(image, factor=1.5):
#     """Enhance the brightness of an image."""
#     enhancer = ImageEnhance.Brightness(image)
#     return enhancer.enhance(factor)



# def decay_lr(optimizer, factor):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] *= factor
            
            
            
            
            #  import cv2
            #  image_np = image.squeeze().permute(1, 2, 0).numpy()

            # # Scale the pixel values if they are in [0, 1]
            #  if image_np.max() <= 1:
            #      image_np = (image_np * 255).astype(np.uint8)

            #  # Convert BGR to HSV
            #  hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
            
            #  # Increase the brightness
            #  hsv[:, :, 2] = hsv[:, :, 2] + 0
            # # hsv[:, :, 2] = np.clip(hsv[:, :, 2] + 0, 0, 255)
            
            #  # Convert back to BGR
            #  img_brightened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            #  # Save or display the image
            #  cv2.imwrite("image_processed.jpg", img_brightened)
            
            
                  #eval_model.load_state_dict(checkpoint['model_state_dict'])
                  #eval_model = IR152_PReLU()

                  
            
            #plot_image(rec_img)
            #import pdb;pdb.set_trace()
            
            #break
            #import pdb;pdb.set_trace()
            

            #if(i==10):
            #    break
        
    ### Evaluation Model (currently I make it same as target classifier)
    
    # eval_model = IR152_PReLU()
    # eval_model.load_pretrained_weights(checkpoint_path='best_model_checkpoint_Prelu_152_IR_112.pth')
    # #true_identity = i  # Replace this with the actual way you obtain ground truth identity  
    # #import pdb;pdb.set_trace()
    
    # recovered_images = load_images_from_folder('recovered_images')
    # input_images = load_images_from_folder('input_images')




    # # # Placeholder for true identities and predictions

    # predictions_recovered = []
    # predictions_input = []
    
    # for i in range(len(recovered_images)):
    #     #true_identity = i  # Replace this with the actual way you obtain ground truth identity
    #    # import pdb;pdb.set_trace()
    #     # Get predictions for recovered images
    #     output_recovered = eval_model(recovered_images[i])
    #     identity_recovered = get_identity(output_recovered)
    #     predictions_recovered.append(identity_recovered)
    
    #     # Get predictions for input images
    #     output_input = eval_model(input_images[i])
    #     identity_input = get_identity(output_input)
    #     predictions_input.append(identity_input)
    
    #     #true_identities.append(true_identity)

    # # # Calculate accuracy
    # print(predictions_recovered)
    # print(predictions_input)
    # accuracy_recovered = accuracy_score(predictions_input, predictions_recovered)
    # # #accuracy_input = accuracy_score(true_identities, predictions_input)

    # # # Report accuracy
    # print(f"Accuracy for recovered images: {accuracy_recovered * 100}%")
    

    # Check if predictions are the same for both sets
    # same_prediction = [p_i == p_r for p_i, p_r in zip(predictions_input, predictions_recovered)]
    # accuracy_same = sum(same_prediction) / len(same_prediction)
    # print(f"Accuracy when predictions are the same: {accuracy_same * 100}%")
    
    ### Save the recovered images in a folder

    # Load and preprocess the image
    # current_directory = os.path.dirname(os.path.abspath(__file__))
    # image_filename = 'CelebA_ex1.jpg'
    # image_path = os.path.join(current_directory, image_filename)
    # image = Image.open(image_path).convert('RGB')
    
    # crop_size = 108
    # crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
    # offset_height = (218 - crop_size) // 2
    # offset_width = (178 - crop_size) // 2
    # proc = []
    # proc.append(transforms.ToTensor())
    # proc.append(transforms.Lambda(crop))
    # proc.append(transforms.ToPILImage())
    # proc.append(transforms.Resize((112, 112)))
    # proc.append(transforms.ToTensor())
    # proc.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    #   # Create a composed transform using your custom steps
    # custom_transform = transforms.Compose(proc)
    #   #to_tensor = transforms.ToTensor()
    #   #to_tensor = transforms.ToTensor()
    #   #image_tensor = to_tensor(resized_image)
    # resized_image = image
    
    # image_tensor = custom_transform (resized_image)
    # input_image = image_tensor.unsqueeze(0)
#    print(celebAloader)
    # Invert the model layers
    #input_recovered = invert_model_layers(model, input_image)