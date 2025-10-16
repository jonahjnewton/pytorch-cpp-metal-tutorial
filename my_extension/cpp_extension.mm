#include <torch/extension.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <iostream>

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// Define a function to add tensors using Metal
torch::Tensor add_tensors_metal(torch::Tensor a, torch::Tensor b, const std::string& shaderFilePath) {

    // Check that device is MPS
    if (a.device().type() != torch::kMPS || b.device().type() != torch::kMPS) {
        throw std::runtime_error("Error: tensors must be on MPS device.");
    }

    // Check that tensors are contiguous
    // Contiguous means that the memory is contiguous
    a = a.contiguous();
    b = b.contiguous();

    // Get the total number of elements in the tensors
    int numElements = a.numel();

    // Create an empty tensor on the MPS device to hold the result
    torch::Tensor result = torch::empty({numElements}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kMPS));
    
    @autoreleasepool{
        // Get the default Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        // Load the Metal shader from the specified path
        NSError* error = nil;
        NSString* shaderSource = [
            NSString stringWithContentsOfFile:[NSString stringWithUTF8String:shaderFilePath.c_str()]
            encoding:NSUTF8StringEncoding 
            error:&error];
        if (error) {
            throw std::runtime_error("Failed to load Metal shader: " + std::string(error.localizedDescription.UTF8String));
        }

        // Compile the Metal shader source
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (!library) {
            throw std::runtime_error("Error compiling Metal shader: " + std::string(error.localizedDescription.UTF8String));
        }

        id<MTLFunction> function = [library newFunctionWithName:@"addTensors"];
        if (!function) {
            throw std::runtime_error("Error: Metal function addTensors not found.");
        }

        // Create a Metal compute pipeline state
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:nil];

        // Get PyTorch's command queue
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        // Get PyTorch's Metal command buffer
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();

        dispatch_sync(serialQueue, ^(){
            // Create a compute command encoder
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            // Set the compute pipeline state
            [encoder setComputePipelineState:pipelineState];

            // Set the buffers
            [encoder setBuffer:getMTLBufferStorage(a) offset:a.storage_offset() a.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(b) offset:b.storage_offset() b.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(result) offset:result.storage_offset() result.element_size() atIndex:2];

            // Dispatch the compute kernel
            MTLSize gridSize = MTLSizeMake(numElements, 1, 1);
            NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > numElements) {
                threadGroupSize = numElements;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            // Tell torch to commit the command buffer
            torch::mps::commit();
        });
    }
    
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensors_metal", &add_tensors_metal, "Add two tensors using Metal");
}4
