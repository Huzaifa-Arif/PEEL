import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    print("CUDA is available.")
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    # Get the name of the current GPU
    current_gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"Current GPU: {current_gpu_name}")
else:
    print("CUDA is not available. Running on CPU.")