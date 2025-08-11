#!/bin/bash

# Build script to compile GLSL compute shaders to SPIR-V bytecode
# Requires glslc (from Vulkan SDK) to be installed and in PATH

set -e  # Exit on any error

SHADER_DIR="src/shaders"
OUTPUT_DIR="src/shaders"

echo "Building Vulkan compute shaders..."

# Check if glslc is available
if ! command -v glslc &> /dev/null; then
    echo "Error: glslc not found. Please install the Vulkan SDK."
    echo "Download from: https://vulkan.lunarg.com/sdk/home"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# List of shaders to compile
SHADERS=(
    "fused_add_relu"
    "fused_matmul_bias_relu"
    "elementwise_add"
    "elementwise_sub"
    "elementwise_mul"
    "elementwise_div"
    "scalar_add"
    "scalar_mul"
    "matrix_mul"
    "relu"
    "sigmoid"
    "tanh"
    "softmax"
    "transpose"
    "copy"
    "sqrt"
    "gelu"
    "swish"
    "reduce_sum"
    "conv2d"
)

# Compile each shader
for shader in "${SHADERS[@]}"; do
    input_file="$SHADER_DIR/${shader}.comp"
    output_file="$OUTPUT_DIR/${shader}.spv"

    if [ -f "$input_file" ]; then
        echo "Compiling $shader..."
        glslc -fshader-stage=compute "$input_file" -o "$output_file"

        if [ $? -eq 0 ]; then
            echo "  ✓ $shader.comp -> $shader.spv"
        else
            echo "  ✗ Failed to compile $shader.comp"
            exit 1
        fi
    else
        echo "  ⚠ Warning: $input_file not found, skipping..."
    fi
done

echo ""
echo "Shader compilation complete!"
echo "Generated SPIR-V files in $OUTPUT_DIR/"

# Optional: Show file sizes
echo ""
echo "SPIR-V file sizes:"
for shader in "${SHADERS[@]}"; do
    spv_file="$OUTPUT_DIR/${shader}.spv"
    if [ -f "$spv_file" ]; then
        size=$(wc -c < "$spv_file")
        echo "  ${shader}.spv: ${size} bytes"
    fi
done

echo ""
echo "To use these shaders, make sure the SPIR-V files are included in your build."
echo "The Rust code will load them using include_bytes! macro."
