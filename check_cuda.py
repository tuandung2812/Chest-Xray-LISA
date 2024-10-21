import torch

def check_torch_cuda():
    print("PyTorch version:", torch.__version__)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        
        # Get the number of GPUs available
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Try to create a tensor on the GPU
        try:
            x = torch.rand(3, 3).cuda()
            print("Tensor created on the GPU successfully.")
            print("Tensor:", x)
        except Exception as e:
            print("Error creating tensor on the GPU:", e)
    else:
        print("CUDA is not available. Please check your CUDA installation.")

if __name__ == "__main__":
    check_torch_cuda()
