import pickle
import torch
import numpy as np


# def find_shortest_dist(fea_target, fea_fake):
#     shortest_dist = np.inf

#     # Ensure fea_target is a 2D array
#     if fea_target.ndim == 1:
#         fea_target = fea_target.reshape(1, -1)

#     for i in range(fea_fake.shape[0]):
#         # Ensure each fea_fake[i, :] is treated as a 2D array
#         fea_fake_i = fea_fake[i, :].reshape(1, -1)

#         # Compute the Euclidean distance
#         dist = np.linalg.norm(fea_fake_i - fea_target, axis=1)
#         min_d = np.min(dist)
#         if min_d < shortest_dist:
#             shortest_dist = min_d

#     return shortest_dist




def euclidean_distance(array1, array2):
    return np.linalg.norm(array1 - array2)




model = 'face'
diff = 1
if(model == 'IR152'):
    
   
    if(diff):
        reconstructed_features_file ='label_features_IR_152_CelebA_reconstructed_diff.pkl'
        orignal_features_file = 'label_features_FaceNet64_CelebA.pkl'
    else:
        reconstructed_features_file = 'label_features_IR_152_CelebA_reconstructed.pkl' 
        orignal_features_file = 'label_features_IR_152_CelebA.pkl'
        
    

else:
   if(diff):
       orignal_features_file = 'label_features_IR_152_CelebA.pkl'
       reconstructed_features_file ='label_features_FaceNet64_CelebA_reconstructed_diff.pkl'
   else:
       reconstructed_features_file = 'label_features_FaceNet64_CelebA_reconstructed.pkl' 
       orignal_features_file = 'label_features_FaceNet64_CelebA.pkl'
  
    



# Open the pickle file and load data
with open(orignal_features_file , 'rb') as file:
    orignal_features_dict = pickle.load(file)
    
with open(reconstructed_features_file  , 'rb') as file:
    reconstructed_features_dict= pickle.load(file)

# sum_distances = {}

# for key in reconstructed_features_dict.keys():
#     #import pdb;pdb.set_trace()
#     reconstructed_tensor = reconstructed_features_dict[key][0]
#     original_tensors = np.stack(orignal_features_dict[key])

#     # Compute the squared sum of minimum distances
#     sum_distance = find_shortest_dist(np.expand_dims(original_tensors, axis=1), np.expand_dims(reconstructed_tensor[0], axis=1)) 
#     sum_distances[key] = sum_distance

# # squared_sum_distances now contains the squared sum of minimum distances for each key
# for key, distance in sum_distances.items():
#     print(f"Squared sum of minimum Euclidean distance for key {key}: {distance}")
    

# total_distance = sum(sum_distances.values())
# average_distance = total_distance / len(sum_distances)
# print(f"Average distance: {average_distance}")
    
    
min_distances = {}

for key in reconstructed_features_dict.keys():
    
    if key in orignal_features_dict:
        original_tensors = orignal_features_dict[key]
        reconstructed_tensor = reconstructed_features_dict[key]
        
    
        # Initialize the minimum distance with a very large value
        min_distance = float('inf')
        for original_tensor in original_tensors:
            # import pdb;pdb.set_trace()
            distance = euclidean_distance(np.expand_dims(reconstructed_tensor[0], axis=1), np.expand_dims(original_tensor, axis=1))
            if distance < min_distance:
                min_distance = distance
    
        min_distances[key] = min_distance

# min_distances now contains the minimum distance for each key
for key, distance in min_distances.items():
    print(f"Minimum Euclidean distance for key {key}: {distance}")
    
    
total_distance = sum(min_distances.values())
average_distance = total_distance / len(min_distances)
print(f"Average distance: {average_distance}")

    


# Now you can use the `data` object
#print(data)  # Just an example, you might want to process the data differently
# total_sum = 0
# for k in reconstructed_features_dict.keys():
#     total_sum += len(reconstructed_features_dict[k])
    

#     #print(k)
# print(total_sum)

# for key in sorted(reconstructed_features_dict.keys()):
#     print(f'Key:{key} and the value is :{len(reconstructed_features_dict[key])}')

# print(len(set(reconstructed_features_dict.keys())))
