import torch
from torch import nn
from loftq import TrueQuantizedLinear
from loftq_cnn import TrueQuantizedConv2d

def test_quantization():
    # Create a test linear layer
    original_layer = nn.Linear(1024, 1024).to('mps')
    
    # Measure original model size
    original_size = sum(p.numel() * p.element_size() for p in original_layer.parameters()) / (1024**2)
    print(f"Original model size: {original_size:.2f} MB")
    
    # Create quantized version
    quantized_layer = TrueQuantizedLinear(
        original_layer,
        quantization_bits=2,
        reduced_rank=16
    )
    
    quantized_layer.quantize(original_layer.weight.data)
    
    # Measure quantized model size
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_layer.parameters()) / (1024**2)
    quantized_size += sum(b.numel() * b.element_size() for b in quantized_layer.buffers()) / (1024**2)
    print(f"Quantized model size: {quantized_size:.2f} MB")
    
    # Check compression ratio
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")
    
    # Test forward pass
    x = torch.randn(32, 1024).to('mps')
    out1 = original_layer(x)
    out2 = quantized_layer(x)
    
    # Compare output similarity (should be close but not identical)
    diff = (out1 - out2).abs().mean()
    print(f"Average absolute difference: {diff:.6f}")
    
    return original_size, quantized_size, diff

def test_conv2d_quantization():
    # Create a test conv layer
    original_conv = nn.Conv2d(
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=1
    ).to('mps')
    
    # Measure original model size
    original_size = sum(p.numel() * p.element_size() for p in original_conv.parameters()) / (1024**2)
    print(f"Original Conv2d size: {original_size:.4f} MB")
    
    # Create quantized version
    quantized_conv = TrueQuantizedConv2d(
        in_channels=64,
        out_channels=128,
        kernel_size=3,
        stride=1,
        padding=1,
        quantization_bits=2,
        reduced_rank=16,
        num_iters=5
    )
    
    # Quantize using weights from original layer
    quantized_conv.quantize(original_conv)
    
    # Measure quantized model size
    quantized_params = sum(p.numel() * p.element_size() for p in quantized_conv.parameters()) / (1024**2)
    quantized_buffers = sum(b.numel() * b.element_size() for b in quantized_conv.buffers()) / (1024**2)
    quantized_size = quantized_params + quantized_buffers
    print(f"Quantized Conv2d size: {quantized_size:.4f} MB")
    print(f"Parameters: {quantized_params:.4f} MB, Buffers: {quantized_buffers:.4f} MB")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")
    
    # Test forward pass with an image-like input
    x = torch.randn(1, 64, 32, 32).to('mps')  # batch_size, channels, height, width
    
    # Forward pass through both layers
    out1 = original_conv(x)
    out2 = quantized_conv(x)
    
    # Compare output similarity
    diff_1 = (out1 - out2).abs().mean()
    print(f"Average absolute difference in Barney method: {diff_1:.6f}")
    
    # Check if output shapes match
    print(f"Original output shape: {out1.shape}")
    print(f"Quantized output shape: {out2.shape}")
    
    return original_size, quantized_size, diff_1

if __name__ == '__main__':
    # test_quantization()
    test_conv2d_quantization()