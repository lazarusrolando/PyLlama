# PyLlama GGUF Runtime Notes

GGUF files store a header, metadata key/value pairs, tensor descriptors, and an aligned tensor-data region. Tensor descriptors contain the tensor name, GGML dimensions, type id, and byte offset.

PyLlama reverses GGML dimensions when presenting tensors to the Transformer runtime so matrices use normal NumPy `(out, in)` conventions.

Supported tensor types in this release:

- F32
- F16
- BF16
- Q4_0
- Q4_1
- Q5_0
- Q5_1
- Q8_0
- Q2_K
- Q4_K
- Q5_K
- Q6_K

Q4_K / Q5_K / Q6_K follow the K-quant block structures used by GGML: 256-value super-blocks with quantized sub-block scales/minimums or 6-bit/8-bit sub-block scales.
