import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Print device ID and other information for each GPU
    for i in range(num_gpus):
        print(f"Device ID: {i}")
        print(f"Device Name: {torch.cuda.get_device_name(i)}")
        print(f"Device Capability: {torch.cuda.get_device_capability(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
        print(f"Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
        print()
else:
    print("CUDA is not available on this system.")
