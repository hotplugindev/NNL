//! Tensor module providing device-agnostic tensor operations
//!
//! This module implements a comprehensive tensor system that works seamlessly
//! across CPU,  and Vulkan backends with automatic memory management
//! and optimized operations.

use crate::device::{Device, DeviceMemory};
use crate::error::{NnlError, Result};

use ndarray::{Array, ArrayD, Dimension};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

pub mod ops;
pub mod view;

pub use view::TensorView;

/// Shape type alias for tensor dimensions
pub type Shape = Vec<usize>;

/// Multi-dimensional tensor with device support
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Internal tensor data representation
    pub data: TensorData,
    shape: Shape,
    /// Device where the tensor is stored
    pub device: Arc<Device>,
    requires_grad: bool,
    grad: Option<Box<Tensor>>,
}

/// Internal tensor data representation
#[derive(Debug, Clone)]
pub enum TensorData {
    /// Data stored on host (CPU)
    Host(ArrayD<f32>),
    /// Data stored on device (GPU)
    Device(Arc<dyn DeviceMemory>),
}

impl Tensor {
    /// Create a new tensor with zeros
    pub fn zeros(shape: &[usize]) -> Result<Self> {
        let device = Device::auto_select()?;
        Self::zeros_on_device(shape, device)
    }

    /// Create a new tensor with zeros on specific device
    pub fn zeros_on_device(shape: &[usize], device: Device) -> Result<Self> {
        let total_elements = shape.iter().product::<usize>();

        // Always use device memory for consistency and to leverage optimized backends
        let memory = device.backend().allocate(total_elements)?;

        // Zero-initialize the memory
        let zeros = vec![0.0f32; total_elements];
        device.backend().copy_to_device(&zeros, memory.as_ref())?;

        Ok(Self {
            data: TensorData::Device(memory),
            shape: shape.to_vec(),
            device: Arc::new(device),
            requires_grad: false,
            grad: None,
        })
    }

    /// Create a new tensor with ones
    pub fn ones(shape: &[usize]) -> Result<Self> {
        let mut tensor = Self::zeros(shape)?;
        tensor.fill(1.0)?;
        Ok(tensor)
    }

    /// Create a new tensor with ones on specific device
    pub fn ones_on_device(shape: &[usize], device: Device) -> Result<Self> {
        let mut tensor = Self::zeros_on_device(shape, device)?;
        tensor.fill(1.0)?;
        Ok(tensor)
    }

    /// Create a tensor from a slice of data
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Result<Self> {
        let device = Device::auto_select()?;
        Self::from_slice_on_device(data, shape, device)
    }

    /// Create a tensor from a slice of data on specific device
    pub fn from_slice_on_device(data: &[f32], shape: &[usize], device: Device) -> Result<Self> {
        let expected_elements = shape.iter().product::<usize>();
        if data.len() != expected_elements {
            return Err(NnlError::shape_mismatch(
                &[expected_elements],
                &[data.len()],
            ));
        }

        // Always use device memory for consistency and to leverage optimized backends
        let memory = device.backend().allocate(data.len())?;
        device.backend().copy_to_device(data, memory.as_ref())?;
        Ok(Self {
            data: TensorData::Device(memory),
            shape: shape.to_vec(),
            device: Arc::new(device),
            requires_grad: false,
            grad: None,
        })
    }

    /// Create a tensor from an ndarray
    pub fn from_array<D>(array: Array<f32, D>) -> Result<Self>
    where
        D: Dimension,
    {
        let shape = array.shape().to_vec();
        let data = array.into_raw_vec();
        Self::from_slice(&data, &shape)
    }

    /// Create a tensor from a 2D array literal
    pub fn from_array_2d(data: &[&[f32]]) -> Result<Self> {
        if data.is_empty() {
            return Err(NnlError::tensor("Cannot create tensor from empty array"));
        }

        let rows = data.len();
        let cols = data[0].len();

        // Verify all rows have the same length
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(NnlError::tensor(format!(
                    "Inconsistent row length at row {}: expected {}, got {}",
                    i,
                    cols,
                    row.len()
                )));
            }
        }

        let flat_data: Vec<f32> = data.iter().flat_map(|row| row.iter().copied()).collect();
        Self::from_slice(&flat_data, &[rows, cols])
    }

    /// Create a random tensor with normal distribution
    pub fn randn(shape: &[usize]) -> Result<Self> {
        use rand::prelude::*;
        use rand_distr::StandardNormal;

        let total_elements = shape.iter().product::<usize>();
        let mut rng = thread_rng();
        let data: Vec<f32> = (0..total_elements)
            .map(|_| rng.sample(StandardNormal))
            .collect();

        Self::from_slice(&data, shape)
    }

    /// Create a random tensor with uniform distribution
    pub fn rand(shape: &[usize]) -> Result<Self> {
        use rand::prelude::*;

        let total_elements = shape.iter().product::<usize>();
        let mut rng = thread_rng();
        let data: Vec<f32> = (0..total_elements).map(|_| rng.r#gen()).collect();

        Self::from_slice(&data, shape)
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Check if tensor requires gradient computation
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set gradient requirement
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Get gradient tensor
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref().map(|g| g.as_ref())
    }

    /// Set gradient tensor
    pub fn set_grad(&mut self, grad: Option<Tensor>) {
        self.grad = grad.map(Box::new);
    }

    /// Fill tensor with a scalar value
    pub fn fill(&mut self, value: f32) -> Result<()> {
        match &mut self.data {
            TensorData::Host(array) => {
                array.fill(value);
                Ok(())
            }
            TensorData::Device(_) => {
                // For device tensors, we need to use a kernel
                // This is a simplified implementation
                let host_data = vec![value; self.size()];
                self.copy_from_slice(&host_data)
            }
        }
    }

    /// Copy data from a slice
    pub fn copy_from_slice(&mut self, data: &[f32]) -> Result<()> {
        if data.len() != self.size() {
            return Err(NnlError::shape_mismatch(&[self.size()], &[data.len()]));
        }

        match &mut self.data {
            TensorData::Host(array) => {
                for (dst, &src) in array.iter_mut().zip(data.iter()) {
                    *dst = src;
                }
                Ok(())
            }
            TensorData::Device(memory) => {
                self.device.backend().copy_to_device(data, memory.as_ref())
            }
        }
    }

    /// Copy data to a slice
    pub fn copy_to_slice(&self, data: &mut [f32]) -> Result<()> {
        if data.len() != self.size() {
            return Err(NnlError::shape_mismatch(&[self.size()], &[data.len()]));
        }

        match &self.data {
            TensorData::Host(array) => {
                for (dst, &src) in data.iter_mut().zip(array.iter()) {
                    *dst = src;
                }
                Ok(())
            }
            TensorData::Device(memory) => self.device.backend().copy_to_host(memory.as_ref(), data),
        }
    }

    /// Convert tensor to a Vec<f32>
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let mut data = vec![0.0; self.size()];
        self.copy_to_slice(&mut data)?;
        Ok(data)
    }

    /// Convert tensor to host (CPU) storage
    pub fn to_host(&self) -> Result<Tensor> {
        match &self.data {
            TensorData::Host(_) => Ok(self.clone()),
            TensorData::Device(_) => {
                let data = self.to_vec()?;
                let cpu_device = Device::cpu()?;
                Self::from_slice_on_device(&data, &self.shape, cpu_device)
            }
        }
    }

    /// Convert tensor to device storage
    pub fn to_device(&self, device: Device) -> Result<Tensor> {
        if self.device.device_type() == device.device_type() {
            return Ok(self.clone());
        }

        let data = self.to_vec()?;
        Self::from_slice_on_device(&data, &self.shape, device)
    }

    /// Reshape tensor to new shape
    /// Reshape tensor to new dimensions
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor> {
        let new_size = new_shape.iter().product::<usize>();
        if new_size != self.size() {
            return Err(NnlError::shape_mismatch(&[self.size()], &[new_size]));
        }

        // GPU-native reshape - just change shape metadata, no data copying needed
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            device: self.device.clone(),
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
        })
    }

    /// Transpose tensor (swap last two dimensions)
    pub fn transpose(&self) -> Result<Tensor> {
        ops::transpose(self)
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, ops::TensorOp::Add)
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, ops::TensorOp::Sub)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, ops::TensorOp::Mul)
    }

    /// Element-wise division
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        self.binary_op(other, ops::TensorOp::Div)
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        ops::matmul(self, other)
    }

    /// Scalar addition
    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        self.scalar_op(scalar, ops::TensorOp::AddScalar)
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        self.scalar_op(scalar, ops::TensorOp::MulScalar)
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Result<Tensor> {
        ops::sqrt(self)
    }

    /// Sum all elements
    pub fn sum(&self) -> Result<f32> {
        let result_tensor = ops::reduce_sum(self, None)?;
        let data = result_tensor.to_vec()?;
        Ok(data[0])
    }

    /// Mean of all elements
    pub fn mean(&self) -> Result<f32> {
        let sum = self.sum()?;
        Ok(sum / self.size() as f32)
    }

    /// Maximum element
    pub fn max(&self) -> Result<f32> {
        let result_tensor = ops::reduce_max(self, None)?;
        let data = result_tensor.to_vec()?;
        Ok(data[0])
    }

    /// Minimum element
    pub fn min(&self) -> Result<f32> {
        let result_tensor = ops::reduce_min(self, None)?;
        let data = result_tensor.to_vec()?;
        Ok(data[0])
    }

    /// Apply activation function
    pub fn activation(&self, activation: crate::activations::Activation) -> Result<Tensor> {
        ops::activation(self, activation)
    }

    /// Internal binary operation
    fn binary_op(&self, other: &Tensor, op: ops::TensorOp) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(NnlError::shape_mismatch(&self.shape, &other.shape));
        }

        ops::binary_op(self, other, op)
    }

    /// Internal scalar operation
    fn scalar_op(&self, scalar: f32, op: ops::TensorOp) -> Result<Tensor> {
        ops::scalar_op(self, scalar, op)
    }

    /// Get a view of the tensor data
    pub fn view(&self) -> TensorView {
        TensorView::new(self)
    }

    /// Check if tensor shapes are broadcastable
    pub fn is_broadcastable_with(&self, other: &Tensor) -> bool {
        ops::is_broadcastable(&self.shape, &other.shape)
    }

    /// Broadcast tensor to target shape
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Result<Tensor> {
        ops::broadcast_to(self, target_shape)
    }

    /// Clone tensor data to new tensor
    /// Clone tensor data (creates a new tensor with copied data)
    pub fn clone_data(&self) -> Result<Tensor> {
        match &self.data {
            TensorData::Device(memory) => {
                // GPU-native clone using copy kernel
                let new_memory = self.device.backend().allocate(self.size())?;

                let backend = self.device.backend();
                if backend
                    .as_any()
                    .downcast_ref::<crate::device::gpu::VulkanBackend>()
                    .is_some()
                {
                    let kernel = crate::device::gpu::VulkanKernel::elementwise(
                        "copy".to_string(),
                        (self.size() * std::mem::size_of::<f32>() / std::mem::size_of::<f32>())
                            as u32,
                    );
                    backend.execute_kernel(&kernel, &[memory.as_ref()], &[new_memory.as_ref()])?;
                } else {
                    // Fallback for other backends
                    let data = self.to_vec()?;
                    return Self::from_slice_on_device(&data, &self.shape, (*self.device).clone());
                }

                Ok(Tensor {
                    data: TensorData::Device(new_memory),
                    shape: self.shape.clone(),
                    device: self.device.clone(),
                    requires_grad: self.requires_grad,
                    grad: self.grad.clone(),
                })
            }
            TensorData::Host(_) => {
                // CPU clone - use existing implementation
                let data = self.to_vec()?;
                Self::from_slice_on_device(&data, &self.shape, (*self.device).clone())
            }
        }
    }
}

// Implement standard operators
impl Add for &Tensor {
    type Output = Result<Tensor>;

    fn add(self, other: &Tensor) -> Self::Output {
        self.add(other)
    }
}

impl Sub for &Tensor {
    type Output = Result<Tensor>;

    fn sub(self, other: &Tensor) -> Self::Output {
        self.sub(other)
    }
}

impl Mul for &Tensor {
    type Output = Result<Tensor>;

    fn mul(self, other: &Tensor) -> Self::Output {
        self.mul(other)
    }
}

impl Div for &Tensor {
    type Output = Result<Tensor>;

    fn div(self, other: &Tensor) -> Self::Output {
        self.div(other)
    }
}

// Display implementation
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_vec() {
            Ok(data) => {
                write!(
                    f,
                    "Tensor(shape={:?}, device={:?}, data=[",
                    self.shape,
                    self.device.device_type()
                )?;

                let max_display = 10;
                for (i, &val) in data.iter().take(max_display).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:.4}", val)?;
                }

                if data.len() > max_display {
                    write!(f, ", ...")?;
                }

                write!(f, "])")
            }
            Err(_) => write!(
                f,
                "Tensor(shape={:?}, device={:?}, data=<unavailable>)",
                self.shape,
                self.device.device_type()
            ),
        }
    }
}

// Serialization support
/// Serializable representation of a tensor for persistence
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializableTensor {
    /// Tensor data as a flat vector
    pub data: Vec<f32>,
    /// Shape of the tensor
    pub shape: Shape,
    /// Whether gradients are required for this tensor
    pub requires_grad: bool,
}

impl From<&Tensor> for SerializableTensor {
    fn from(tensor: &Tensor) -> Self {
        Self {
            data: tensor.to_vec().unwrap_or_default(),
            shape: tensor.shape.clone(),
            requires_grad: tensor.requires_grad,
        }
    }
}

impl TryFrom<SerializableTensor> for Tensor {
    type Error = NnlError;

    fn try_from(serializable: SerializableTensor) -> Result<Self> {
        let mut tensor = Self::from_slice(&serializable.data, &serializable.shape)?;
        tensor.set_requires_grad(serializable.requires_grad);
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::zeros(&[2, 3]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.size(), 6);
        assert_eq!(tensor.ndim(), 2);
    }

    #[test]
    fn test_tensor_from_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_slice(&data, &[2, 2]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);

        let retrieved_data = tensor.to_vec().unwrap();
        assert_eq!(retrieved_data, data);
    }

    #[test]
    fn test_tensor_from_array_2d() {
        let row1 = [1.0, 2.0];
        let row2 = [3.0, 4.0];
        let data = [&row1[..], &row2[..]];
        let tensor = Tensor::from_array_2d(&data).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);

        let retrieved_data = tensor.to_vec().unwrap();
        assert_eq!(retrieved_data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_fill() {
        let mut tensor = Tensor::zeros(&[3, 3]).unwrap();
        tensor.fill(5.0).unwrap();

        let data = tensor.to_vec().unwrap();
        assert!(data.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_tensor_reshape() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let reshaped = tensor.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.to_vec().unwrap(), tensor.to_vec().unwrap());
    }

    #[test]
    fn test_tensor_transpose() -> Result<()> {
        let row1 = [1.0, 2.0];
        let row2 = [3.0, 4.0];
        let data = [&row1[..], &row2[..]];
        let tensor = Tensor::from_array_2d(&data)?;
        let transposed = tensor.transpose()?;
        assert_eq!(transposed.shape(), &[2, 2]);

        let expected = vec![1.0, 3.0, 2.0, 4.0];
        assert_eq!(transposed.to_vec()?, expected);
        Ok(())
    }

    #[test]
    fn test_tensor_arithmetic() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[2.0, 2.0, 2.0, 2.0], &[2, 2]).unwrap();

        let sum = a.add(&b).unwrap();
        assert_eq!(sum.to_vec().unwrap(), vec![3.0, 4.0, 5.0, 6.0]);

        let product = a.mul(&b).unwrap();
        assert_eq!(product.to_vec().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_tensor_reductions() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let sum = tensor.sum().unwrap();
        assert_eq!(sum, 10.0);

        let mean = tensor.mean().unwrap();
        assert_eq!(mean, 2.5);
    }

    #[test]
    fn test_tensor_device_conversion() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(
            tensor.device().device_type(),
            crate::device::DeviceType::Cpu
        );

        let host_tensor = tensor.to_host().unwrap();
        assert_eq!(host_tensor.to_vec().unwrap(), tensor.to_vec().unwrap());
    }

    #[test]
    fn test_tensor_serialization() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let serializable = SerializableTensor::from(&tensor);
        let deserialized = Tensor::try_from(serializable).unwrap();

        assert_eq!(tensor.to_vec().unwrap(), deserialized.to_vec().unwrap());
        assert_eq!(tensor.shape(), deserialized.shape());
    }
}
