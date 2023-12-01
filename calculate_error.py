
final  = []
layer4 = []
layer3 = []
layer2 = []
layer1 = []


############ RUn 1


final.append(0.01901240646839142)
layer4.append(0.00016434307326562703)
layer3.append(6.53878232697025e-05)
layer2.append(4.896027530776337e-05)
layer1.append(3.309729800093919e-05)



#### Run 2

final.append(0.020798368379473686)
layer4.append(3.156196544296108e-05)
layer3.append(7.235922385007143e-05)
layer2.append(5.475289799505845e-05)
layer1.append(2.9419752536341548e-05)



### Run3
final.append(0.019761038944125175)
layer4.append(7.274008385138586e-05)
layer3.append(7.318456482607871e-05)
layer2.append( 4.2605013732099906e-05)
layer1.append(4.25087237090338e-05)


import numpy as np

# Defining the lists
final = [0.01901240646839142, 0.020798368379473686, 0.019761038944125175]
layer4 = [0.00016434307326562703, 3.156196544296108e-05, 7.274008385138586e-05]
layer3 = [6.53878232697025e-05, 7.235922385007143e-05, 7.318456482607871e-05]
layer2 = [4.896027530776337e-05, 5.475289799505845e-05, 4.2605013732099906e-05]
layer1 = [3.309729800093919e-05, 2.9419752536341548e-05, 4.25087237090338e-05]

# Calculating the mean and standard deviation for each list
final_mean, final_std = np.mean(final), np.std(final)
layer4_mean, layer4_std = np.mean(layer4), np.std(layer4)
layer3_mean, layer3_std = np.mean(layer3), np.std(layer3)
layer2_mean, layer2_std = np.mean(layer2), np.std(layer2)
layer1_mean, layer1_std = np.mean(layer1), np.std(layer1)

# Printing the results
print("Final - Mean: {:.8f}, Std: {:.8f}".format(final_mean, final_std))
print("Layer 4 - Mean: {:.8f}, Std: {:.8f}".format(layer4_mean, layer4_std))
print("Layer 3 - Mean: {:.8f}, Std: {:.8f}".format(layer3_mean, layer3_std))
print("Layer 2 - Mean: {:.8f}, Std: {:.8f}".format(layer2_mean, layer2_std))
print("Layer 1 - Mean: {:.8f}, Std: {:.8f}".format(layer1_mean, layer1_std))
