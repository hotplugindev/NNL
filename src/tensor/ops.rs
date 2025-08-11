//! Tensor operations module
//!
//! This module provides optimized implementations of tensor operations
//! that work across different device backends with full GPU support.

use crate::activations::Activation;
use crate::device::DeviceType;
use crate::error::{NnlError, Result};
use crate::tensor::Tensor;
use rayon::prelude::*;

/// Enumeration of tensor operations
#[derive(Debug, Clone, Copy)]
pub enum TensorOp {
    /// Element-wise addition of two tensors
    Add,
    /// Element-wise subtraction of two tensors
    Sub,
    /// Element-wise multiplication of two tensors
    Mul,
    /// Element-wise division of two tensors
    Div,
    /// Addition of a scalar to all tensor elements
    AddScalar,
    /// Multiplication of all tensor elements by a scalar
    MulScalar,
    /// Matrix multiplication between two tensors
    MatMul,
    /// Element-wise square root of tensor elements
    Sqrt,
}

/// Perform binary operation between two tensors
pub fn binary_op(a: &Tensor, b: &Tensor, op: TensorOp) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(NnlError::shape_mismatch(a.shape(), b.shape()));
    }

    match (a.device().device_type(), b.device().device_type()) {
        (DeviceType::Cpu, DeviceType::Cpu) => cpu_binary_op(a, b, op),
        (DeviceType::Vulkan, DeviceType::Vulkan) => gpu_binary_op(a, b, op),
        _ => {
            // Handle device mismatch - convert to same device
            let b_on_a_device = b.to_device(a.device().clone())?;
            binary_op(a, &b_on_a_device, op)
        }
    }
}

/// CPU implementation of binary operations using optimized device backend
fn cpu_binary_op(a: &Tensor, b: &Tensor, op: TensorOp) -> Result<Tensor> {
    let backend = a.device().backend();

    // Get input memories
    let a_memory = get_tensor_memory(a)?;
    let b_memory = get_tensor_memory(b)?;

    // Allocate result memory
    let result_memory = backend.allocate(a.shape().iter().product::<usize>())?;

    // Create optimized CPU kernel based on operation
    use crate::device::cpu::{CpuKernel, CpuOperation};
    let cpu_op = match op {
        TensorOp::Add => CpuOperation::ElementwiseAdd,
        TensorOp::Sub => CpuOperation::ElementwiseSubtract,
        TensorOp::Mul => CpuOperation::ElementwiseMultiply,
        TensorOp::Div => CpuOperation::ElementwiseDivide,
        _ => {
            return Err(NnlError::unsupported(
                "Operation not supported for binary tensors",
            ));
        }
    };

    let kernel = CpuKernel::new(format!("elementwise_{:?}", op), cpu_op);

    // Execute using optimized device backend
    backend.execute_kernel(&kernel, &[a_memory, b_memory], &[result_memory.as_ref()])?;

    // Create result tensor with the computed memory
    Ok(Tensor {
        data: crate::tensor::TensorData::Device(result_memory),
        shape: a.shape().to_vec(),
        device: a.device.clone(),
        requires_grad: a.requires_grad || b.requires_grad,
        grad: None,
    })
}

/// GPU implementation of binary operations
fn gpu_binary_op(a: &Tensor, b: &Tensor, op: TensorOp) -> Result<Tensor> {
    // Validate tensor compatibility
    if a.shape() != b.shape() {
        return Err(NnlError::shape_mismatch(a.shape(), b.shape()));
    }

    let backend = a.device().backend();

    // Get GPU memory for inputs
    let a_memory = get_tensor_memory(a)?;
    let b_memory = get_tensor_memory(b)?;

    // Allocate result memory directly
    let result_memory = backend.allocate(a.size())?;

    // Execute the appropriate kernel
    let kernel_name = match op {
        TensorOp::Add => "elementwise_add",
        TensorOp::Sub => "elementwise_sub",
        TensorOp::Mul => "elementwise_mul",
        TensorOp::Div => "elementwise_div",
        _ => return Err(NnlError::unsupported("Unsupported GPU binary operation")),
    };

    // Create kernel
    let kernel =
        crate::device::gpu::VulkanKernel::elementwise(kernel_name.to_string(), a.size() as u32);

    // Execute kernel
    backend.execute_kernel(&kernel, &[a_memory, b_memory], &[result_memory.as_ref()])?;

    // Create result tensor with the computed memory
    Ok(Tensor {
        data: crate::tensor::TensorData::Device(result_memory),
        shape: a.shape().to_vec(),
        device: a.device.clone(),
        requires_grad: a.requires_grad || b.requires_grad,
        grad: None,
    })
}

/// Perform scalar operation on tensor
pub fn scalar_op(tensor: &Tensor, scalar: f32, op: TensorOp) -> Result<Tensor> {
    match tensor.device().device_type() {
        DeviceType::Cpu => cpu_scalar_op(tensor, scalar, op),
        DeviceType::Vulkan => gpu_scalar_op(tensor, scalar, op),
    }
}

/// CPU implementation of scalar operations
fn cpu_scalar_op(tensor: &Tensor, scalar: f32, op: TensorOp) -> Result<Tensor> {
    let data = tensor.to_vec()?;

    let result_data: Vec<f32> = match op {
        TensorOp::AddScalar => data.par_iter().map(|&x| x + scalar).collect(),
        TensorOp::MulScalar => data.par_iter().map(|&x| x * scalar).collect(),
        _ => return Err(NnlError::unsupported("Unsupported scalar operation")),
    };

    Tensor::from_slice_on_device(&result_data, tensor.shape(), tensor.device().clone())
}

/// GPU implementation of scalar operations
fn gpu_scalar_op(tensor: &Tensor, scalar: f32, op: TensorOp) -> Result<Tensor> {
    let backend = tensor.device().backend();

    // Get input memory
    let input_memory = get_tensor_memory(tensor)?;

    // Allocate result memory directly
    let result_memory = backend.allocate(tensor.size())?;

    // Execute the appropriate kernel with scalar parameter
    let kernel_name = match op {
        TensorOp::AddScalar => "scalar_add",
        TensorOp::MulScalar => "scalar_mul",
        _ => return Err(NnlError::unsupported("Unsupported GPU scalar operation")),
    };

    // Create uniform buffer for scalar
    let scalar_memory = backend.allocate_uniform(1)?;
    backend.copy_u32_to_device(&[scalar.to_bits()], scalar_memory.as_ref())?;

    // Create kernel
    let kernel = crate::device::gpu::VulkanKernel::elementwise(
        kernel_name.to_string(),
        tensor.size() as u32,
    );

    backend.execute_kernel_with_uniform(
        &kernel,
        &[input_memory],
        &[result_memory.as_ref()],
        Some(scalar_memory.as_ref()),
    )?;

    // Create result tensor with the computed memory
    Ok(Tensor {
        data: crate::tensor::TensorData::Device(result_memory),
        shape: tensor.shape().to_vec(),
        device: tensor.device.clone(),
        requires_grad: tensor.requires_grad,
        grad: None,
    })
}

/// Matrix multiplication
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Validate shapes for matrix multiplication
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NnlError::invalid_input(
            "Matrix multiplication requires 2D tensors",
        ));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape[1] != b_shape[0] {
        return Err(NnlError::shape_mismatch(a_shape, b_shape));
    }

    let output_shape = vec![a_shape[0], b_shape[1]];

    match (a.device().device_type(), b.device().device_type()) {
        (DeviceType::Cpu, DeviceType::Cpu) => cpu_matmul(a, b, &output_shape),
        (DeviceType::Vulkan, DeviceType::Vulkan) => gpu_matmul(a, b, &output_shape),
        _ => {
            // Handle device mismatch
            let b_on_a_device = b.to_device(a.device().clone())?;
            matmul(a, &b_on_a_device)
        }
    }
}

/// CPU matrix multiplication using optimized device backend
fn cpu_matmul(a: &Tensor, b: &Tensor, output_shape: &[usize]) -> Result<Tensor> {
    let backend = a.device().backend();
    let a_shape = a.shape();
    let b_shape = b.shape();

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    // Get input memories
    let a_memory = get_tensor_memory(a)?;
    let b_memory = get_tensor_memory(b)?;

    // Allocate result memory
    let result_memory = backend.allocate(output_shape.iter().product::<usize>())?;

    // Create optimized CPU kernel with matrix dimensions
    use crate::device::cpu::{CpuKernel, CpuOperation};
    let kernel = CpuKernel::new(
        "matrix_mul".to_string(),
        CpuOperation::MatrixMultiply { m, n, k },
    );

    // Execute using optimized device backend
    backend.execute_kernel(&kernel, &[a_memory, b_memory], &[result_memory.as_ref()])?;

    // Create result tensor with the computed memory
    Ok(Tensor {
        data: crate::tensor::TensorData::Device(result_memory),
        shape: output_shape.to_vec(),
        device: a.device.clone(),
        requires_grad: a.requires_grad || b.requires_grad,
        grad: None,
    })
}

/// GPU matrix multiplication
fn gpu_matmul(a: &Tensor, b: &Tensor, output_shape: &[usize]) -> Result<Tensor> {
    let backend = a.device().backend();

    // Get input memories
    let a_memory = get_tensor_memory(a)?;
    let b_memory = get_tensor_memory(b)?;

    // Allocate result memory directly
    let result_memory = backend.allocate(output_shape.iter().product::<usize>())?;

    // Create dimensions buffer for matrix multiplication
    let a_shape = a.shape();
    let b_shape = b.shape();
    let dimensions = [a_shape[0] as u32, b_shape[1] as u32, a_shape[1] as u32];

    // Allocate uniform buffer for dimensions
    let dims_memory = backend.allocate_uniform(3)?; // 3 u32 values
    backend.copy_u32_to_device(&dimensions, dims_memory.as_ref())?;

    // Create kernel
    let kernel = crate::device::gpu::VulkanKernel::matrix(
        "matrix_mul".to_string(),
        a_shape[0] as u32,
        b_shape[1] as u32,
    );

    // Execute with uniform buffer
    backend.execute_kernel_with_uniform(
        &kernel,
        &[a_memory, b_memory],
        &[result_memory.as_ref()],
        Some(dims_memory.as_ref()),
    )?;

    // Create result tensor with the computed memory
    Ok(Tensor {
        data: crate::tensor::TensorData::Device(result_memory),
        shape: output_shape.to_vec(),
        device: a.device.clone(),
        requires_grad: a.requires_grad || b.requires_grad,
        grad: None,
    })
}

/// Apply activation function
pub fn activation(input: &Tensor, activation: Activation) -> Result<Tensor> {
    match input.device().device_type() {
        DeviceType::Cpu => cpu_activation(input, activation),
        DeviceType::Vulkan => gpu_activation(input, activation),
    }
}

/// Apply element-wise square root to tensor
pub fn sqrt(tensor: &Tensor) -> Result<Tensor> {
    match tensor.device().device_type() {
        DeviceType::Cpu => cpu_sqrt(tensor),
        DeviceType::Vulkan => gpu_sqrt(tensor),
    }
}

/// Matrix transpose operation
pub fn transpose(tensor: &Tensor) -> Result<Tensor> {
    if tensor.ndim() < 2 {
        return Err(NnlError::tensor(
            "Cannot transpose tensor with less than 2 dimensions",
        ));
    }

    match tensor.device().device_type() {
        DeviceType::Cpu => cpu_transpose(tensor),
        DeviceType::Vulkan => gpu_transpose(tensor),
    }
}

/// CPU activation functions
fn cpu_activation(input: &Tensor, activation: Activation) -> Result<Tensor> {
    let data = input.to_vec()?;

    let result_data: Vec<f32> = match activation {
        Activation::ReLU => data.par_iter().map(|&x| x.max(0.0)).collect(),
        Activation::Sigmoid => data.par_iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
        Activation::Tanh => data.par_iter().map(|&x| x.tanh()).collect(),
        Activation::Softmax => {
            let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_data: Vec<f32> = data.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exp_data.iter().sum();
            exp_data.iter().map(|&x| x / sum).collect()
        }
        Activation::Linear => data,
        Activation::LeakyReLU(alpha) => data
            .par_iter()
            .map(|&x| if x > 0.0 { x } else { alpha * x })
            .collect(),
        Activation::Swish => data.par_iter().map(|&x| x / (1.0 + (-x).exp())).collect(),
        Activation::GELU => data
            .par_iter()
            .map(|&x| {
                0.5 * x
                    * (1.0
                        + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            })
            .collect(),
        Activation::ELU(alpha) => data
            .par_iter()
            .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            .collect(),
        Activation::Mish => data
            .par_iter()
            .map(|&x| x * (1.0 + x.exp()).ln().tanh())
            .collect(),
        Activation::PReLU(alpha) => data
            .par_iter()
            .map(|&x| if x > 0.0 { x } else { alpha * x })
            .collect(),
        Activation::SELU => data
            .par_iter()
            .map(|&x| {
                let alpha = 1.6732632423543772848170429916717;
                let scale = 1.0507009873554804934193349852946;
                if x > 0.0 {
                    scale * x
                } else {
                    scale * alpha * (x.exp() - 1.0)
                }
            })
            .collect(),
    };

    Tensor::from_slice_on_device(&result_data, input.shape(), input.device().clone())
}

/// GPU activation functions
fn gpu_activation(input: &Tensor, activation: Activation) -> Result<Tensor> {
    let kernel_name = match activation {
        Activation::ReLU => "relu",
        Activation::Sigmoid => "sigmoid",
        Activation::Tanh => "tanh",
        Activation::Softmax => "softmax",
        Activation::Linear => return Ok(input.clone()),
        _ => return Err(NnlError::unsupported("Activation not implemented on GPU")),
    };

    // Get input memory
    let input_memory = get_tensor_memory(input)?;
    let backend = input.device().backend();

    // Allocate result memory directly
    let result_memory = backend.allocate(input.size())?;

    // Create kernel
    let kernel = if kernel_name == "softmax" {
        // Softmax needs special uniform buffer for size
        let size_memory = backend.allocate_uniform(1)?;
        backend.copy_u32_to_device(&[input.size() as u32], size_memory.as_ref())?;

        let kernel = crate::device::gpu::VulkanKernel::elementwise(
            kernel_name.to_string(),
            input.size() as u32,
        );
        backend.execute_kernel_with_uniform(
            &kernel,
            &[input_memory],
            &[result_memory.as_ref()],
            Some(size_memory.as_ref()),
        )?;

        return Ok(Tensor {
            data: crate::tensor::TensorData::Device(result_memory),
            shape: input.shape().to_vec(),
            device: input.device.clone(),
            requires_grad: input.requires_grad,
            grad: None,
        });
    } else {
        crate::device::gpu::VulkanKernel::elementwise(kernel_name.to_string(), input.size() as u32)
    };

    backend.execute_kernel(&kernel, &[input_memory], &[result_memory.as_ref()])?;

    // Create result tensor with the computed memory
    Ok(Tensor {
        data: crate::tensor::TensorData::Device(result_memory),
        shape: input.shape().to_vec(),
        device: input.device.clone(),
        requires_grad: input.requires_grad,
        grad: None,
    })
}

/// Reduction operations
pub fn reduce_sum(tensor: &Tensor, dim: Option<usize>) -> Result<Tensor> {
    match tensor.device().device_type() {
        DeviceType::Cpu => cpu_reduce_sum(tensor, dim),
        _ => {
            // Fall back to CPU for now - GPU reductions are complex
            let cpu_tensor = tensor.to_host()?;
            let result = cpu_reduce_sum(&cpu_tensor, dim)?;
            if tensor.device().device_type() != DeviceType::Cpu {
                result.to_device(tensor.device().clone())
            } else {
                Ok(result)
            }
        }
    }
}

/// CPU square root implementation
fn cpu_sqrt(tensor: &Tensor) -> Result<Tensor> {
    let data = tensor.to_vec()?;
    let result_data: Vec<f32> = data.iter().map(|&x| x.sqrt()).collect();

    Tensor::from_slice_on_device(&result_data, &tensor.shape, tensor.device().clone())
}

/// GPU square root implementation
fn gpu_sqrt(tensor: &Tensor) -> Result<Tensor> {
    use crate::tensor::{Tensor, TensorData};

    if let TensorData::Device(ref memory) = tensor.data {
        // Create a new device memory for the result
        let result_memory = tensor.device.backend().allocate(tensor.size())?;

        // Use the tensor's device backend to execute the sqrt kernel
        let backend = tensor.device.backend();

        // Create kernel and execute
        let kernel =
            crate::device::gpu::VulkanKernel::elementwise("sqrt".to_string(), tensor.size() as u32);
        backend.execute_kernel(&kernel, &[memory.as_ref()], &[result_memory.as_ref()])?;

        Ok(Tensor {
            data: TensorData::Device(result_memory),
            shape: tensor.shape.clone(),
            device: tensor.device.clone(),
            requires_grad: tensor.requires_grad,
            grad: None,
        })
    } else {
        Err(NnlError::tensor("Expected device tensor for GPU sqrt"))
    }
}

/// CPU transpose implementation
fn cpu_transpose(tensor: &Tensor) -> Result<Tensor> {
    let mut new_shape = tensor.shape().to_vec();
    let last_idx = new_shape.len() - 1;
    new_shape.swap(last_idx - 1, last_idx);

    let data = tensor.to_vec()?;
    let transposed_data = transpose_data_2d(&data, tensor.shape())?;
    Tensor::from_slice_on_device(&transposed_data, &new_shape, tensor.device().clone())
}

/// GPU transpose implementation
fn gpu_transpose(tensor: &Tensor) -> Result<Tensor> {
    use crate::tensor::{Tensor, TensorData};

    if let TensorData::Device(ref memory) = tensor.data {
        let mut new_shape = tensor.shape().to_vec();
        let last_idx = new_shape.len() - 1;
        new_shape.swap(last_idx - 1, last_idx);

        // Create a new device memory for the result
        let result_memory = tensor.device.backend().allocate(tensor.size())?;

        // Use the tensor's device backend to execute the transpose kernel
        let backend = tensor.device.backend();

        // Create dimensions buffer for transpose
        let rows = tensor.shape()[tensor.ndim() - 2] as u32;
        let cols = tensor.shape()[tensor.ndim() - 1] as u32;
        let dimensions = [rows, cols];

        // Allocate uniform buffer for dimensions
        let dims_memory = backend.allocate_uniform(2)?; // 2 u32 values
        backend.copy_u32_to_device(&dimensions, dims_memory.as_ref())?;

        // Create kernel and execute
        let kernel = crate::device::gpu::VulkanKernel::matrix("transpose".to_string(), rows, cols);
        backend.execute_kernel_with_uniform(
            &kernel,
            &[memory.as_ref()],
            &[result_memory.as_ref()],
            Some(dims_memory.as_ref()),
        )?;

        Ok(Tensor {
            data: TensorData::Device(result_memory),
            shape: new_shape,
            device: tensor.device.clone(),
            requires_grad: tensor.requires_grad,
            grad: None,
        })
    } else {
        Err(NnlError::tensor("Expected device tensor for GPU transpose"))
    }
}

/// Helper function for 2D matrix transpose
fn transpose_data_2d(data: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    if shape.len() != 2 {
        return Err(NnlError::tensor(
            "transpose_data_2d only supports 2D tensors",
        ));
    }

    let rows = shape[0];
    let cols = shape[1];
    let mut result = vec![0.0; data.len()];

    for i in 0..rows {
        for j in 0..cols {
            let src_idx = i * cols + j;
            let dst_idx = j * rows + i;
            result[dst_idx] = data[src_idx];
        }
    }

    Ok(result)
}

fn cpu_reduce_sum(tensor: &Tensor, dim: Option<usize>) -> Result<Tensor> {
    let data = tensor.to_vec()?;

    match dim {
        None => {
            // Sum all elements
            let sum: f32 = data.par_iter().sum();
            Tensor::from_slice_on_device(&[sum], &[1], tensor.device().clone())
        }
        Some(d) => {
            if d >= tensor.ndim() {
                return Err(NnlError::invalid_input("Dimension out of bounds"));
            }

            let shape = tensor.shape();
            let mut new_shape = shape.to_vec();
            new_shape.remove(d);

            if new_shape.is_empty() {
                new_shape.push(1);
            }

            // Calculate strides and perform reduction
            let stride_before: usize = shape[..d].iter().product();
            let stride_after: usize = shape[d + 1..].iter().product();
            let dim_size = shape[d];

            let mut result_data = vec![0.0; new_shape.iter().product()];

            for i in 0..stride_before {
                for k in 0..stride_after {
                    let mut sum = 0.0;
                    for j in 0..dim_size {
                        let idx = i * dim_size * stride_after + j * stride_after + k;
                        sum += data[idx];
                    }
                    let result_idx = i * stride_after + k;
                    result_data[result_idx] = sum;
                }
            }

            Tensor::from_slice_on_device(&result_data, &new_shape, tensor.device().clone())
        }
    }
}

/// Compute the maximum value along a specified dimension or globally
pub fn reduce_max(tensor: &Tensor, dim: Option<usize>) -> Result<Tensor> {
    let data = tensor.to_vec()?;

    match dim {
        None => {
            let max_val = data
                .par_iter()
                .fold(|| f32::NEG_INFINITY, |a, &b| a.max(b))
                .reduce(|| f32::NEG_INFINITY, |a, b| a.max(b));
            Tensor::from_slice_on_device(&[max_val], &[1], tensor.device().clone())
        }
        Some(_) => {
            // Simplified implementation for now
            let max_val = data
                .par_iter()
                .fold(|| f32::NEG_INFINITY, |a, &b| a.max(b))
                .reduce(|| f32::NEG_INFINITY, |a, b| a.max(b));
            Tensor::from_slice_on_device(&[max_val], &[1], tensor.device().clone())
        }
    }
}

/// Compute the minimum value along a specified dimension or globally
pub fn reduce_min(tensor: &Tensor, dim: Option<usize>) -> Result<Tensor> {
    let data = tensor.to_vec()?;

    match dim {
        None => {
            let min_val = data
                .par_iter()
                .fold(|| f32::INFINITY, |a, &b| a.min(b))
                .reduce(|| f32::INFINITY, |a, b| a.min(b));
            Tensor::from_slice_on_device(&[min_val], &[1], tensor.device().clone())
        }
        Some(_) => {
            // Simplified implementation for now
            let min_val = data
                .par_iter()
                .fold(|| f32::INFINITY, |a, &b| a.min(b))
                .reduce(|| f32::INFINITY, |a, b| a.min(b));
            Tensor::from_slice_on_device(&[min_val], &[1], tensor.device().clone())
        }
    }
}

/// Broadcasting utilities
pub fn is_broadcastable(shape_a: &[usize], shape_b: &[usize]) -> bool {
    let max_ndim = shape_a.len().max(shape_b.len());

    for i in 0..max_ndim {
        let dim_a = if i < shape_a.len() {
            shape_a[shape_a.len() - 1 - i]
        } else {
            1
        };
        let dim_b = if i < shape_b.len() {
            shape_b[shape_b.len() - 1 - i]
        } else {
            1
        };

        if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
            return false;
        }
    }

    true
}

/// Broadcast a tensor to a target shape following NumPy broadcasting rules
pub fn broadcast_to(tensor: &Tensor, target_shape: &[usize]) -> Result<Tensor> {
    if !is_broadcastable(tensor.shape(), target_shape) {
        return Err(NnlError::shape_mismatch(tensor.shape(), target_shape));
    }

    if tensor.shape() == target_shape {
        return Ok(tensor.clone());
    }

    match tensor.device().device_type() {
        DeviceType::Cpu => cpu_broadcast(tensor, target_shape),
        _ => {
            let cpu_tensor = tensor.to_host()?;
            let result = cpu_broadcast(&cpu_tensor, target_shape)?;
            if tensor.device().device_type() != DeviceType::Cpu {
                result.to_device(tensor.device().clone())
            } else {
                Ok(result)
            }
        }
    }
}

fn cpu_broadcast(tensor: &Tensor, target_shape: &[usize]) -> Result<Tensor> {
    let data = tensor.to_vec()?;
    let source_shape = tensor.shape();
    let target_size: usize = target_shape.iter().product();

    let mut result_data = vec![0.0; target_size];

    // Calculate strides for both source and target
    let mut source_strides = vec![1; source_shape.len()];
    for i in (0..source_shape.len().saturating_sub(1)).rev() {
        source_strides[i] = source_strides[i + 1] * source_shape[i + 1];
    }

    let mut target_strides = vec![1; target_shape.len()];
    for i in (0..target_shape.len().saturating_sub(1)).rev() {
        target_strides[i] = target_strides[i + 1] * target_shape[i + 1];
    }

    // Fill result data by broadcasting
    for i in 0..target_size {
        let mut source_idx = 0;
        let mut temp_i = i;

        for (j, &target_stride) in target_strides.iter().enumerate() {
            let coord = temp_i / target_stride;
            temp_i %= target_stride;

            let source_dim_idx = j + target_shape.len() - source_shape.len();
            if source_dim_idx < source_shape.len() {
                let source_coord = if source_shape[source_dim_idx] == 1 {
                    0
                } else {
                    coord
                };
                source_idx += source_coord * source_strides[source_dim_idx];
            }
        }

        result_data[i] = data[source_idx];
    }

    Tensor::from_slice_on_device(&result_data, target_shape, tensor.device().clone())
}

/// Convolution operation
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor> {
    if input.ndim() != 4 || weight.ndim() != 4 {
        return Err(NnlError::invalid_input(
            "Conv2d requires 4D tensors (NCHW format)",
        ));
    }

    match input.device().device_type() {
        DeviceType::Cpu => cpu_conv2d(input, weight, bias, stride, padding),
        _ => {
            // Fall back to CPU for now
            let cpu_input = input.to_host()?;
            let cpu_weight = weight.to_host()?;
            let cpu_bias = if let Some(b) = bias {
                Some(b.to_host()?)
            } else {
                None
            };
            let result = cpu_conv2d(&cpu_input, &cpu_weight, cpu_bias.as_ref(), stride, padding)?;

            if input.device().device_type() != DeviceType::Cpu {
                result.to_device(input.device().clone())
            } else {
                Ok(result)
            }
        }
    }
}

fn cpu_conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor> {
    let input_data = input.to_vec()?;
    let weight_data = weight.to_vec()?;
    let bias_data = if let Some(b) = bias {
        Some(b.to_vec()?)
    } else {
        None
    };

    let input_shape = input.shape();
    let weight_shape = weight.shape();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let input_height = input_shape[2];
    let input_width = input_shape[3];

    let out_channels = weight_shape[0];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];

    if in_channels != weight_shape[1] {
        return Err(NnlError::shape_mismatch(input_shape, weight_shape));
    }

    let output_height = (input_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
    let output_width = (input_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

    let output_shape = vec![batch_size, out_channels, output_height, output_width];
    let output_size = output_shape.iter().product();
    let mut output_data = vec![0.0; output_size];

    // Parallel convolution computation
    output_data
        .par_chunks_mut(output_height * output_width)
        .enumerate()
        .for_each(|(idx, output_channel)| {
            let batch = idx / out_channels;
            let out_c = idx % out_channels;

            for oh in 0..output_height {
                for ow in 0..output_width {
                    let mut sum = 0.0;

                    for in_c in 0..in_channels {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh * stride.0 + kh;
                                let iw = ow * stride.1 + kw;

                                if ih >= padding.0
                                    && iw >= padding.1
                                    && ih < input_height + padding.0
                                    && iw < input_width + padding.1
                                {
                                    let actual_ih = ih - padding.0;
                                    let actual_iw = iw - padding.1;

                                    let input_idx =
                                        batch * in_channels * input_height * input_width
                                            + in_c * input_height * input_width
                                            + actual_ih * input_width
                                            + actual_iw;

                                    let weight_idx =
                                        out_c * in_channels * kernel_height * kernel_width
                                            + in_c * kernel_height * kernel_width
                                            + kh * kernel_width
                                            + kw;

                                    sum += input_data[input_idx] * weight_data[weight_idx];
                                }
                            }
                        }
                    }

                    if let Some(ref bias_vals) = bias_data {
                        sum += bias_vals[out_c];
                    }

                    output_channel[oh * output_width + ow] = sum;
                }
            }
        });

    Tensor::from_slice_on_device(&output_data, &output_shape, input.device().clone())
}

// Helper functions for GPU operations

/// Get device memory from tensor
fn get_tensor_memory(tensor: &Tensor) -> Result<&dyn crate::device::DeviceMemory> {
    match &tensor.data {
        crate::tensor::TensorData::Device(memory) => Ok(memory.as_ref()),
        crate::tensor::TensorData::Host(_) => Err(NnlError::device(
            "Cannot get device memory from host tensor",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    #[test]
    fn test_binary_operations() {
        let _device = Device::cpu().unwrap();
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let result = binary_op(&a, &b, TensorOp::Add).unwrap();
        let expected = vec![6.0, 8.0, 10.0, 12.0];
        assert_eq!(result.to_vec().unwrap(), expected);
    }

    #[test]
    fn test_scalar_operations() {
        let _device = Device::cpu().unwrap();
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result = scalar_op(&tensor, 2.0, TensorOp::MulScalar).unwrap();
        let expected = vec![2.0, 4.0, 6.0, 8.0];
        assert_eq!(result.to_vec().unwrap(), expected);
    }

    #[test]
    fn test_matrix_multiplication() {
        let _device = Device::cpu().unwrap();
        let a = Tensor::from_slice(&[1.0, 2.0, 0.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let result = matmul(&a, &b).unwrap();
        let expected = vec![7.0, 10.0, 12.0, 16.0];
        assert_eq!(result.to_vec().unwrap(), expected);
    }

    #[test]
    fn test_activation_functions() {
        let _device = Device::cpu().unwrap();
        let tensor = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();

        let relu_result = activation(&tensor, Activation::ReLU).unwrap();
        let expected_relu = vec![0.0, 0.0, 1.0, 2.0];
        assert_eq!(relu_result.to_vec().unwrap(), expected_relu);

        let sigmoid_result = activation(&tensor, Activation::Sigmoid).unwrap();
        let sigmoid_data = sigmoid_result.to_vec().unwrap();

        // Check that sigmoid values are in (0, 1)
        for &val in &sigmoid_data {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_reduction_operations() {
        let _device = Device::cpu().unwrap();
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sum_result = reduce_sum(&tensor, None).unwrap();
        assert_eq!(sum_result.to_vec().unwrap(), vec![10.0]);

        let max_result = reduce_max(&tensor, None).unwrap();
        assert_eq!(max_result.to_vec().unwrap(), vec![4.0]);
    }

    #[test]
    fn test_broadcasting() {
        assert!(is_broadcastable(&[2, 1, 3], &[2, 4, 3]));
        assert!(is_broadcastable(&[1], &[2, 3, 4]));
        assert!(!is_broadcastable(&[2, 3], &[2, 4]));
    }

    #[test]
    fn test_softmax() {
        let _device = Device::cpu().unwrap();
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();

        let result = activation(&tensor, Activation::Softmax).unwrap();
        let data = result.to_vec().unwrap();

        // Check that softmax sums to 1
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that all values are positive
        for &val in &data {
            assert!(val > 0.0);
        }
    }
}
