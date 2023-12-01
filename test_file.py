#from main import *

import torch
from utilities import plot_channels
saved_tensors = torch.load('/home/arifh/Desktop/IBM_2023/saved_tensors.pth')

# Accessing the tensors
# optimal_b1_loaded = loaded_tensors['optimal_b1']
# max_pool_out_loaded = loaded_tensors['max_pool_out']



# for key, value in loaded_tensors.items():
#     globals()[key] = value


optimal_b1 = saved_tensors['optimal_b1']
optimal_b2 = saved_tensors['optimal_b2']
optimal_b3 = saved_tensors['optimal_b3']
optimal_b4 = saved_tensors['optimal_b4']
max_pool_out = saved_tensors['max_pool_out']
res_4 = saved_tensors['res_4']
res_3 = saved_tensors['res_3']
res_2 = saved_tensors['res_2']
res_1 = saved_tensors['res_1']

plot_channels(max_pool_out,rec = False)
plot_channels(optimal_b1,rec = True)


print(torch.norm(optimal_b4 - res_3 ))
print(torch.norm(optimal_b3 - res_2 ))
print(torch.norm(optimal_b2 - res_1 ))
print(torch.norm(optimal_b1 - max_pool_out ))