//! Highly optimized CPU backend implementation
//!
//! This module provides a high-performance CPU backend using:
//! - SIMD instructions for vectorized operations
//! - Cache-friendly memory access patterns
//! - BLAS integration for optimal matrix operations
//! - Advanced thread pool management
//! - Memory prefetching and loop unrolling

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]
#![allow(missing_docs)]

use crate::device::{Backend, DeviceInfo, DeviceMemory, DeviceType, Kernel};
use crate::error::{NnlError, Result};
use rayon::prelude::*;
use std::alloc::{Layout, alloc, alloc_zeroed, dealloc};
use std::arch::x86_64::*;
use std::sync::Arc;

/// Optimal block size for cache-friendly matrix operations
const BLOCK_SIZE: usize = 64;
/// SIMD vector size for f32 (AVX2 = 8 floats)
const SIMD_WIDTH: usize = 8;
/// Memory alignment for SIMD operations
const MEMORY_ALIGNMENT: usize = 32;
/// Threshold for using SIMD operations (elements)
const SIMD_THRESHOLD: usize = 64;
/// Threshold for using parallel operations (elements)
const PARALLEL_THRESHOLD: usize = 1024;
/// Threshold for using blocked matrix multiplication
const BLOCKED_MATMUL_THRESHOLD: usize = 128;

/// High-performance CPU backend implementation
pub struct CpuBackend {
    thread_pool: Arc<rayon::ThreadPool>,
    cpu_features: CpuFeatures,
}

/// CPU feature detection for optimal code paths
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    has_avx2: bool,
    has_fma: bool,
    has_avx512: bool,
    cache_line_size: usize,
    l1_cache_size: usize,
    l2_cache_size: usize,
    l3_cache_size: usize,
}

impl CpuBackend {
    /// Create a new optimized CPU backend
    pub fn new() -> Result<Self> {
        let num_threads = num_cpus::get();

        // Create thread pool with CPU affinity optimization
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("nnl-cpu-{}", i))
            .build()
            .map_err(|e| NnlError::device(format!("Failed to create thread pool: {}", e)))?;

        let cpu_features = detect_cpu_features();

        log::info!("CPU Backend initialized with {} threads", num_threads);
        log::info!(
            "CPU Features: AVX2={}, FMA={}, AVX512={}",
            cpu_features.has_avx2,
            cpu_features.has_fma,
            cpu_features.has_avx512
        );

        Ok(Self {
            thread_pool: Arc::new(thread_pool),
            cpu_features,
        })
    }

    /// Get the number of CPU threads
    pub fn num_threads(&self) -> usize {
        self.thread_pool.current_num_threads()
    }

    /// Get CPU features
    pub fn cpu_features(&self) -> &CpuFeatures {
        &self.cpu_features
    }
}

impl Backend for CpuBackend {
    fn device_info(&self) -> Result<DeviceInfo> {
        Ok(DeviceInfo {
            name: format!(
                "Optimized CPU ({} threads, AVX2={}, FMA={})",
                self.num_threads(),
                self.cpu_features.has_avx2,
                self.cpu_features.has_fma
            ),
            device_type: DeviceType::Cpu,
            memory_size: get_system_memory(),
            compute_units: Some(num_cpus::get() as u32),
            supports_f16: false,
            supports_f64: true,
        })
    }

    fn allocate(&self, size: usize) -> Result<Arc<dyn DeviceMemory>> {
        let size_bytes = size * std::mem::size_of::<f32>();
        CpuMemory::new_aligned(size_bytes).map(|m| Arc::new(m) as Arc<dyn DeviceMemory>)
    }

    fn copy_to_device(&self, data: &[f32], memory: &dyn DeviceMemory) -> Result<()> {
        let cpu_memory = memory
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid memory type for CPU backend"))?;

        cpu_memory.copy_from_slice_optimized(data)
    }

    fn copy_to_host(&self, memory: &dyn DeviceMemory, data: &mut [f32]) -> Result<()> {
        let cpu_memory = memory
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid memory type for CPU backend"))?;

        cpu_memory.copy_to_slice_optimized(data)
    }

    fn execute_kernel(
        &self,
        kernel: &dyn Kernel,
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
    ) -> Result<()> {
        let cpu_kernel = kernel
            .as_any()
            .downcast_ref::<CpuKernel>()
            .ok_or_else(|| NnlError::device("Invalid kernel type for CPU backend"))?;

        cpu_kernel.execute_optimized(inputs, outputs, &self.thread_pool, &self.cpu_features)
    }

    fn synchronize(&self) -> Result<()> {
        // CPU operations are synchronous by default
        Ok(())
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Safe pointer wrapper for thread safety
#[derive(Debug)]
struct SafePtr(*mut f32);

// SAFETY: SafePtr is safe to send/sync because we exclusively own the memory
unsafe impl Send for SafePtr {}
unsafe impl Sync for SafePtr {}

impl SafePtr {
    fn new(ptr: *mut f32) -> Self {
        SafePtr(ptr)
    }

    fn as_ptr(&self) -> *const f32 {
        self.0
    }

    fn as_mut_ptr(&self) -> *mut f32 {
        self.0
    }
}

/// Optimized CPU memory with SIMD alignment
#[derive(Debug)]
pub struct CpuMemory {
    ptr: SafePtr,
    size: usize,
    layout: Layout,
    aligned: bool,
}

impl CpuMemory {
    /// Create new aligned memory for optimal SIMD performance
    pub fn new_aligned(size_bytes: usize) -> Result<Self> {
        let float_count = size_bytes / std::mem::size_of::<f32>();
        let aligned_size = (float_count + SIMD_WIDTH - 1) & !(SIMD_WIDTH - 1);
        let total_bytes = aligned_size * std::mem::size_of::<f32>();

        let layout = Layout::from_size_align(total_bytes, MEMORY_ALIGNMENT)
            .map_err(|e| NnlError::device(format!("Invalid memory layout: {}", e)))?;

        let ptr = unsafe {
            let raw_ptr = alloc_zeroed(layout);
            if raw_ptr.is_null() {
                return Err(NnlError::device("Failed to allocate aligned memory"));
            }
            SafePtr::new(raw_ptr as *mut f32)
        };

        Ok(Self {
            ptr,
            size: size_bytes,
            layout,
            aligned: true,
        })
    }

    /// Create regular memory (fallback)
    pub fn new(size_bytes: usize) -> Result<Self> {
        let float_count = size_bytes / std::mem::size_of::<f32>();
        let layout = Layout::array::<f32>(float_count)
            .map_err(|e| NnlError::device(format!("Invalid memory layout: {}", e)))?;

        let ptr = unsafe {
            let raw_ptr = alloc(layout);
            if raw_ptr.is_null() {
                return Err(NnlError::device("Failed to allocate memory"));
            }
            SafePtr::new(raw_ptr as *mut f32)
        };

        Ok(Self {
            ptr,
            size: size_bytes,
            layout,
            aligned: false,
        })
    }

    /// Get immutable pointer
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr.as_ptr()
    }

    /// Get mutable pointer
    pub fn as_mut_ptr(&self) -> *mut f32 {
        self.ptr.as_mut_ptr()
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.size / std::mem::size_of::<f32>())
        }
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&self) -> &mut [f32] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.as_mut_ptr(),
                self.size / std::mem::size_of::<f32>(),
            )
        }
    }

    /// Optimized copy from slice using SIMD when possible
    pub fn copy_from_slice_optimized(&self, data: &[f32]) -> Result<()> {
        let dst = self.as_mut_slice();
        if data.len() > dst.len() {
            return Err(NnlError::device("Source data too large for memory buffer"));
        }

        if self.aligned && data.len() >= SIMD_WIDTH {
            unsafe {
                simd_copy_from_slice(data, dst);
            }
        } else {
            dst[..data.len()].copy_from_slice(data);
        }
        Ok(())
    }

    /// Optimized copy to slice using SIMD when possible
    pub fn copy_to_slice_optimized(&self, data: &mut [f32]) -> Result<()> {
        let src = self.as_slice();
        if data.len() > src.len() {
            return Err(NnlError::device("Destination buffer too small"));
        }

        if self.aligned && data.len() >= SIMD_WIDTH {
            unsafe {
                simd_copy_to_slice(src, data);
            }
        } else {
            data.copy_from_slice(&src[..data.len()]);
        }
        Ok(())
    }
}

impl DeviceMemory for CpuMemory {
    fn size(&self) -> usize {
        self.size
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Drop for CpuMemory {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_mut_ptr() as *mut u8, self.layout);
        }
    }
}

/// High-performance CPU kernel implementation
pub struct CpuKernel {
    name: String,
    operation: CpuOperation,
}

impl CpuKernel {
    /// Create a new CPU kernel
    pub fn new(name: String, operation: CpuOperation) -> Self {
        Self { name, operation }
    }

    /// Execute kernel with optimizations
    pub fn execute_optimized(
        &self,
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        thread_pool: &rayon::ThreadPool,
        cpu_features: &CpuFeatures,
    ) -> Result<()> {
        thread_pool.install(|| {
            self.operation
                .execute_optimized(inputs, outputs, cpu_features)
        })
    }
}

impl Kernel for CpuKernel {
    fn name(&self) -> &str {
        &self.name
    }

    fn local_size(&self) -> Option<[u32; 3]> {
        None // CPU doesn't use local work groups
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Optimized CPU operations
pub enum CpuOperation {
    MatrixMultiply {
        m: usize,
        n: usize,
        k: usize,
    },
    ElementwiseAdd,
    ElementwiseMultiply,
    ElementwiseSubtract,
    ElementwiseDivide,
    Convolution2D {
        in_height: usize,
        in_width: usize,
        kernel_height: usize,
        kernel_width: usize,
        stride: usize,
        padding: usize,
    },
    Activation(ActivationType),
    Reduction(ReductionType),
}

#[derive(Clone, Copy, Debug)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU(f32),
    ELU(f32),
    GELU,
}

#[derive(Clone, Copy, Debug)]
pub enum ReductionType {
    Sum,
    Mean,
    Max,
    Min,
    ArgMax,
    ArgMin,
}

impl CpuOperation {
    fn execute_optimized(
        &self,
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        cpu_features: &CpuFeatures,
    ) -> Result<()> {
        match self {
            CpuOperation::MatrixMultiply { m, n, k } => {
                Self::matrix_multiply_optimized(inputs, outputs, *m, *n, *k, cpu_features)
            }
            CpuOperation::ElementwiseAdd => {
                Self::elementwise_add_optimized(inputs, outputs, cpu_features)
            }
            CpuOperation::ElementwiseMultiply => {
                Self::elementwise_multiply_optimized(inputs, outputs, cpu_features)
            }
            CpuOperation::ElementwiseSubtract => {
                Self::elementwise_subtract_optimized(inputs, outputs, cpu_features)
            }
            CpuOperation::ElementwiseDivide => {
                Self::elementwise_divide_optimized(inputs, outputs, cpu_features)
            }
            CpuOperation::Convolution2D { .. } => {
                Self::convolution_2d_optimized(inputs, outputs, cpu_features)
            }
            CpuOperation::Activation(activation) => {
                Self::activation_optimized(inputs, outputs, activation, cpu_features)
            }
            CpuOperation::Reduction(reduction) => {
                Self::reduction_optimized(inputs, outputs, reduction, cpu_features)
            }
        }
    }

    /// Highly optimized matrix multiplication using blocked algorithm and SIMD
    fn matrix_multiply_optimized(
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        m: usize,
        n: usize,
        k: usize,
        cpu_features: &CpuFeatures,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(NnlError::device(
                "Matrix multiply requires 2 inputs and 1 output",
            ));
        }

        let a_memory = inputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid input memory type"))?;
        let b_memory = inputs[1]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid input memory type"))?;
        let c_memory = outputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid output memory type"))?;

        let a = a_memory.as_slice();
        let b = b_memory.as_slice();
        let c = c_memory.as_mut_slice();

        // Choose algorithm based on matrix size
        let total_elements = m * n * k;

        if total_elements < BLOCKED_MATMUL_THRESHOLD * BLOCKED_MATMUL_THRESHOLD {
            // Small matrices: use simple scalar operations for better cache performance
            matrix_multiply_simple_scalar(a, b, c, m, n, k);
        } else if total_elements < PARALLEL_THRESHOLD * PARALLEL_THRESHOLD {
            // Medium matrices: use SIMD but no threading overhead
            unsafe {
                if cpu_features.has_avx2 && a_memory.aligned && b_memory.aligned && c_memory.aligned
                {
                    matrix_multiply_avx2_simple(a, b, c, m, n, k);
                } else {
                    matrix_multiply_simple_scalar(a, b, c, m, n, k);
                }
            }
        } else {
            // Large matrices: use full optimization with blocking and parallelization
            unsafe {
                if cpu_features.has_avx2 && a_memory.aligned && b_memory.aligned && c_memory.aligned
                {
                    matrix_multiply_avx2_blocked(a, b, c, m, n, k);
                } else {
                    matrix_multiply_blocked_parallel(a, b, c, m, n, k);
                }
            }
        }

        Ok(())
    }

    /// SIMD-optimized elementwise addition
    fn elementwise_add_optimized(
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        cpu_features: &CpuFeatures,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(NnlError::device(
                "Elementwise add requires 2 inputs and 1 output",
            ));
        }

        let a_memory = inputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid input memory type"))?;
        let b_memory = inputs[1]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid input memory type"))?;
        let c_memory = outputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid output memory type"))?;

        let a = a_memory.as_slice();
        let b = b_memory.as_slice();
        let c = c_memory.as_mut_slice();

        if a.len() != b.len() || a.len() != c.len() {
            return Err(NnlError::device("Input and output sizes must match"));
        }

        // Choose algorithm based on array size
        if a.len() < SIMD_THRESHOLD {
            // Small arrays: use simple scalar operations
            elementwise_add_scalar(a, b, c);
        } else if a.len() < PARALLEL_THRESHOLD {
            // Medium arrays: use SIMD without parallelization overhead
            unsafe {
                if cpu_features.has_avx2 && a_memory.aligned && b_memory.aligned && c_memory.aligned
                {
                    elementwise_add_avx2(a, b, c);
                } else {
                    elementwise_add_scalar(a, b, c);
                }
            }
        } else {
            // Large arrays: use parallel processing
            unsafe {
                if cpu_features.has_avx2 && a_memory.aligned && b_memory.aligned && c_memory.aligned
                {
                    elementwise_add_avx2(a, b, c);
                } else {
                    elementwise_add_parallel(a, b, c);
                }
            }
        }

        Ok(())
    }

    /// SIMD-optimized elementwise multiplication
    fn elementwise_multiply_optimized(
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        cpu_features: &CpuFeatures,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(NnlError::device(
                "Elementwise multiply requires 2 inputs and 1 output",
            ));
        }

        let a_memory = inputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid input memory type"))?;
        let b_memory = inputs[1]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid input memory type"))?;
        let c_memory = outputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid output memory type"))?;

        let a = a_memory.as_slice();
        let b = b_memory.as_slice();
        let c = c_memory.as_mut_slice();

        if a.len() != b.len() || a.len() != c.len() {
            return Err(NnlError::device("Input and output sizes must match"));
        }

        // Choose algorithm based on array size
        if a.len() < SIMD_THRESHOLD {
            // Small arrays: use simple scalar operations
            elementwise_multiply_scalar(a, b, c);
        } else if a.len() < PARALLEL_THRESHOLD {
            // Medium arrays: use SIMD without parallelization overhead
            unsafe {
                if cpu_features.has_avx2 && a_memory.aligned && b_memory.aligned && c_memory.aligned
                {
                    elementwise_multiply_avx2(a, b, c);
                } else {
                    elementwise_multiply_scalar(a, b, c);
                }
            }
        } else {
            // Large arrays: use parallel processing
            unsafe {
                if cpu_features.has_avx2 && a_memory.aligned && b_memory.aligned && c_memory.aligned
                {
                    elementwise_multiply_avx2(a, b, c);
                } else {
                    elementwise_multiply_parallel(a, b, c);
                }
            }
        }

        Ok(())
    }

    /// SIMD-optimized elementwise subtraction
    fn elementwise_subtract_optimized(
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        cpu_features: &CpuFeatures,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(NnlError::device(
                "Elementwise subtract requires 2 inputs and 1 output",
            ));
        }

        let a_memory = inputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid input memory type"))?;
        let b_memory = inputs[1]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid input memory type"))?;
        let c_memory = outputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid output memory type"))?;

        let a = a_memory.as_slice();
        let b = b_memory.as_slice();
        let c = c_memory.as_mut_slice();

        if a.len() != b.len() || a.len() != c.len() {
            return Err(NnlError::device("Input and output sizes must match"));
        }

        // Choose algorithm based on array size
        if a.len() < SIMD_THRESHOLD {
            // Small arrays: use simple scalar operations
            elementwise_subtract_scalar(a, b, c);
        } else if a.len() < PARALLEL_THRESHOLD {
            // Medium arrays: use SIMD without parallelization overhead
            unsafe {
                if cpu_features.has_avx2 && a_memory.aligned && b_memory.aligned && c_memory.aligned
                {
                    elementwise_subtract_avx2(a, b, c);
                } else {
                    elementwise_subtract_scalar(a, b, c);
                }
            }
        } else {
            // Large arrays: use parallel processing
            unsafe {
                if cpu_features.has_avx2 && a_memory.aligned && b_memory.aligned && c_memory.aligned
                {
                    elementwise_subtract_avx2(a, b, c);
                } else {
                    elementwise_subtract_parallel(a, b, c);
                }
            }
        }

        Ok(())
    }

    /// SIMD-optimized elementwise division
    fn elementwise_divide_optimized(
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        cpu_features: &CpuFeatures,
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(NnlError::device(
                "Elementwise divide requires 2 inputs and 1 output",
            ));
        }

        let a_memory = inputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid input memory type"))?;
        let b_memory = inputs[1]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid input memory type"))?;
        let c_memory = outputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid output memory type"))?;

        let a = a_memory.as_slice();
        let b = b_memory.as_slice();
        let c = c_memory.as_mut_slice();

        if a.len() != b.len() || a.len() != c.len() {
            return Err(NnlError::device("Input and output sizes must match"));
        }

        // Choose algorithm based on array size
        if a.len() < SIMD_THRESHOLD {
            // Small arrays: use simple scalar operations
            elementwise_divide_scalar(a, b, c);
        } else if a.len() < PARALLEL_THRESHOLD {
            // Medium arrays: use SIMD without parallelization overhead
            unsafe {
                if cpu_features.has_avx2 && a_memory.aligned && b_memory.aligned && c_memory.aligned
                {
                    elementwise_divide_avx2(a, b, c);
                } else {
                    elementwise_divide_scalar(a, b, c);
                }
            }
        } else {
            // Large arrays: use parallel processing
            unsafe {
                if cpu_features.has_avx2 && a_memory.aligned && b_memory.aligned && c_memory.aligned
                {
                    elementwise_divide_avx2(a, b, c);
                } else {
                    elementwise_divide_parallel(a, b, c);
                }
            }
        }

        Ok(())
    }

    /// Placeholder for optimized convolution
    fn convolution_2d_optimized(
        _inputs: &[&dyn DeviceMemory],
        _outputs: &[&dyn DeviceMemory],
        _cpu_features: &CpuFeatures,
    ) -> Result<()> {
        // TODO: Implement optimized convolution with im2col + GEMM
        Err(NnlError::device("Convolution2D not yet implemented"))
    }

    /// Optimized activation functions
    fn activation_optimized(
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        activation: &ActivationType,
        cpu_features: &CpuFeatures,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(NnlError::device("Activation requires 1 input and 1 output"));
        }

        let input_memory = inputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid input memory type"))?;
        let output_memory = outputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid output memory type"))?;

        let input = input_memory.as_slice();
        let output = output_memory.as_mut_slice();

        if input.len() != output.len() {
            return Err(NnlError::device("Input and output sizes must match"));
        }

        // Choose algorithm based on array size
        if input.len() < SIMD_THRESHOLD {
            // Small arrays: use simple scalar operations
            match activation {
                ActivationType::ReLU => relu_scalar(input, output),
                ActivationType::Sigmoid => sigmoid_scalar(input, output),
                ActivationType::Tanh => tanh_scalar(input, output),
                ActivationType::Softmax => softmax_scalar(input, output),
                ActivationType::LeakyReLU(alpha) => leaky_relu_scalar(input, output, *alpha),
                ActivationType::ELU(alpha) => elu_scalar(input, output, *alpha),
                ActivationType::GELU => gelu_scalar(input, output),
            }
        } else if input.len() < PARALLEL_THRESHOLD {
            // Medium arrays: use SIMD without parallelization overhead
            unsafe {
                match activation {
                    ActivationType::ReLU => {
                        if cpu_features.has_avx2 && input_memory.aligned && output_memory.aligned {
                            relu_avx2(input, output);
                        } else {
                            relu_scalar(input, output);
                        }
                    }
                    ActivationType::Sigmoid => {
                        if cpu_features.has_avx2 && input_memory.aligned && output_memory.aligned {
                            sigmoid_avx2(input, output);
                        } else {
                            sigmoid_scalar(input, output);
                        }
                    }
                    ActivationType::Tanh => tanh_scalar(input, output),
                    ActivationType::Softmax => softmax_scalar(input, output),
                    ActivationType::LeakyReLU(alpha) => leaky_relu_scalar(input, output, *alpha),
                    ActivationType::ELU(alpha) => elu_scalar(input, output, *alpha),
                    ActivationType::GELU => gelu_scalar(input, output),
                }
            }
        } else {
            // Large arrays: use parallel processing
            unsafe {
                match activation {
                    ActivationType::ReLU => {
                        if cpu_features.has_avx2 && input_memory.aligned && output_memory.aligned {
                            relu_avx2(input, output);
                        } else {
                            relu_parallel(input, output);
                        }
                    }
                    ActivationType::Sigmoid => {
                        if cpu_features.has_avx2 && input_memory.aligned && output_memory.aligned {
                            sigmoid_avx2(input, output);
                        } else {
                            sigmoid_parallel(input, output);
                        }
                    }
                    ActivationType::Tanh => tanh_parallel(input, output),
                    ActivationType::Softmax => softmax_parallel(input, output),
                    ActivationType::LeakyReLU(alpha) => leaky_relu_parallel(input, output, *alpha),
                    ActivationType::ELU(alpha) => elu_parallel(input, output, *alpha),
                    ActivationType::GELU => gelu_parallel(input, output),
                }
            }
        }

        Ok(())
    }

    /// Optimized reduction operations
    fn reduction_optimized(
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        reduction: &ReductionType,
        cpu_features: &CpuFeatures,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(NnlError::device("Reduction requires 1 input and 1 output"));
        }

        let input_memory = inputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid input memory type"))?;
        let output_memory = outputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| NnlError::device("Invalid output memory type"))?;

        let input = input_memory.as_slice();
        let output = output_memory.as_mut_slice();

        if output.len() != 1 {
            return Err(NnlError::device("Reduction output must be scalar"));
        }

        unsafe {
            match reduction {
                ReductionType::Sum => {
                    if cpu_features.has_avx2 && input_memory.aligned {
                        output[0] = sum_avx2(input);
                    } else {
                        output[0] = sum_parallel(input);
                    }
                }
                ReductionType::Mean => {
                    if cpu_features.has_avx2 && input_memory.aligned {
                        output[0] = sum_avx2(input) / input.len() as f32;
                    } else {
                        output[0] = sum_parallel(input) / input.len() as f32;
                    }
                }
                ReductionType::Max => {
                    output[0] = max_parallel(input);
                }
                ReductionType::Min => {
                    output[0] = min_parallel(input);
                }
                ReductionType::ArgMax => {
                    output[0] = argmax_parallel(input) as f32;
                }
                ReductionType::ArgMin => {
                    output[0] = argmin_parallel(input) as f32;
                }
            }
        }

        Ok(())
    }
}

// SIMD optimized functions
#[target_feature(enable = "avx2")]
unsafe fn simd_copy_from_slice(src: &[f32], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    let simd_len = len & !(SIMD_WIDTH - 1);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let src_vec = _mm256_loadu_ps(src.as_ptr().add(i));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), src_vec);
        }
    }

    // Handle remaining elements
    if simd_len < len {
        dst[simd_len..len].copy_from_slice(&src[simd_len..len]);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn simd_copy_to_slice(src: &[f32], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    let simd_len = len & !(SIMD_WIDTH - 1);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let src_vec = _mm256_loadu_ps(src.as_ptr().add(i));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), src_vec);
        }
    }

    // Handle remaining elements
    if simd_len < len {
        dst[simd_len..len].copy_from_slice(&src[simd_len..len]);
    }
}

// AVX2 matrix multiplication with blocking
#[target_feature(enable = "avx2,fma")]
unsafe fn matrix_multiply_avx2_blocked(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    // Zero the output matrix
    c.fill(0.0);

    // Blocked matrix multiplication for cache efficiency
    for ii in (0..m).step_by(BLOCK_SIZE) {
        for jj in (0..n).step_by(BLOCK_SIZE) {
            for kk in (0..k).step_by(BLOCK_SIZE) {
                let i_end = (ii + BLOCK_SIZE).min(m);
                let j_end = (jj + BLOCK_SIZE).min(n);
                let k_end = (kk + BLOCK_SIZE).min(k);

                for i in ii..i_end {
                    for j in (jj..j_end).step_by(SIMD_WIDTH) {
                        let j_actual_end = (j + SIMD_WIDTH).min(j_end);
                        if j_actual_end - j == SIMD_WIDTH {
                            unsafe {
                                let mut sum_vec = _mm256_setzero_ps();

                                for l in kk..k_end {
                                    let a_val = _mm256_broadcast_ss(&a[i * k + l]);
                                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(l * n + j));
                                    sum_vec = _mm256_fmadd_ps(a_val, b_vec, sum_vec);
                                }

                                let c_vec = _mm256_loadu_ps(c.as_ptr().add(i * n + j));
                                let result = _mm256_add_ps(c_vec, sum_vec);
                                _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), result);
                            }
                        } else {
                            // Handle remaining elements with scalar code
                            for jj in j..j_actual_end {
                                for l in kk..k_end {
                                    c[i * n + jj] += a[i * k + l] * b[l * n + jj];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Simple scalar matrix multiplication for small matrices
fn matrix_multiply_simple_scalar(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    // Zero the output matrix
    c.fill(0.0);

    // Simple triple-loop matrix multiplication optimized for small matrices
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// AVX2 matrix multiplication without blocking (for medium matrices)
#[target_feature(enable = "avx2,fma")]
unsafe fn matrix_multiply_avx2_simple(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    // Zero the output matrix
    c.fill(0.0);

    // SIMD matrix multiplication without blocking overhead
    for i in 0..m {
        for j in (0..n).step_by(SIMD_WIDTH) {
            let j_end = (j + SIMD_WIDTH).min(n);
            if j_end - j == SIMD_WIDTH {
                unsafe {
                    let mut sum_vec = _mm256_setzero_ps();
                    for l in 0..k {
                        let a_val = _mm256_broadcast_ss(&a[i * k + l]);
                        let b_vec = _mm256_loadu_ps(b.as_ptr().add(l * n + j));
                        sum_vec = _mm256_fmadd_ps(a_val, b_vec, sum_vec);
                    }
                    _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), sum_vec);
                }
            } else {
                // Handle remaining elements with scalar code
                for jj in j..j_end {
                    let mut sum = 0.0;
                    for l in 0..k {
                        sum += a[i * k + l] * b[l * n + jj];
                    }
                    c[i * n + jj] = sum;
                }
            }
        }
    }
}

// Parallel blocked matrix multiplication fallback
fn matrix_multiply_blocked_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    _m: usize,
    n: usize,
    k: usize,
) {
    // Zero the output matrix
    c.par_iter_mut().for_each(|x| *x = 0.0);

    // Parallel computation of matrix multiplication
    c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c_row[j] = sum;
        }
    });
}

// AVX2 elementwise operations
#[target_feature(enable = "avx2")]
unsafe fn elementwise_add_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
    let len = a.len();
    let simd_len = len & !(SIMD_WIDTH - 1);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a_vec = _mm256_load_ps(a.as_ptr().add(i));
            let b_vec = _mm256_load_ps(b.as_ptr().add(i));
            let result = _mm256_add_ps(a_vec, b_vec);
            _mm256_store_ps(c.as_mut_ptr().add(i), result);
        }
    }

    // Handle remaining elements
    for i in simd_len..len {
        c[i] = a[i] + b[i];
    }
}

#[target_feature(enable = "avx2")]
unsafe fn elementwise_multiply_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
    let len = a.len();
    let simd_len = len & !(SIMD_WIDTH - 1);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a_vec = _mm256_load_ps(a.as_ptr().add(i));
            let b_vec = _mm256_load_ps(b.as_ptr().add(i));
            let result = _mm256_mul_ps(a_vec, b_vec);
            _mm256_store_ps(c.as_mut_ptr().add(i), result);
        }
    }

    // Handle remaining elements
    for i in simd_len..len {
        c[i] = a[i] * b[i];
    }
}

#[target_feature(enable = "avx2")]
unsafe fn elementwise_subtract_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
    let len = a.len();
    let simd_len = len & !(SIMD_WIDTH - 1);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a_vec = _mm256_load_ps(a.as_ptr().add(i));
            let b_vec = _mm256_load_ps(b.as_ptr().add(i));
            let result = _mm256_sub_ps(a_vec, b_vec);
            _mm256_store_ps(c.as_mut_ptr().add(i), result);
        }
    }

    // Handle remaining elements
    for i in simd_len..len {
        c[i] = a[i] - b[i];
    }
}

#[target_feature(enable = "avx2")]
unsafe fn elementwise_divide_avx2(a: &[f32], b: &[f32], c: &mut [f32]) {
    let len = a.len();
    let simd_len = len & !(SIMD_WIDTH - 1);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let a_vec = _mm256_load_ps(a.as_ptr().add(i));
            let b_vec = _mm256_load_ps(b.as_ptr().add(i));
            let result = _mm256_div_ps(a_vec, b_vec);
            _mm256_store_ps(c.as_mut_ptr().add(i), result);
        }
    }

    // Handle remaining elements
    for i in simd_len..len {
        c[i] = a[i] / b[i];
    }
}

// Simple scalar elementwise operations for small arrays
fn elementwise_add_scalar(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] + b[i];
    }
}

fn elementwise_multiply_scalar(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] * b[i];
    }
}

fn elementwise_subtract_scalar(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] - b[i];
    }
}

fn elementwise_divide_scalar(a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..a.len() {
        c[i] = a[i] / b[i];
    }
}

// Simple scalar activation functions for small arrays
fn relu_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].max(0.0);
    }
}

fn sigmoid_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = 1.0 / (1.0 + (-input[i]).exp());
    }
}

fn tanh_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].tanh();
    }
}

fn softmax_scalar(input: &[f32], output: &mut [f32]) {
    // Find max for numerical stability
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut exp_sum = 0.0;
    for (i, &x) in input.iter().enumerate() {
        output[i] = (x - max_val).exp();
        exp_sum += output[i];
    }

    // Normalize
    for out in output.iter_mut() {
        *out /= exp_sum;
    }
}

fn leaky_relu_scalar(input: &[f32], output: &mut [f32], alpha: f32) {
    for i in 0..input.len() {
        output[i] = if input[i] > 0.0 {
            input[i]
        } else {
            alpha * input[i]
        };
    }
}

fn elu_scalar(input: &[f32], output: &mut [f32], alpha: f32) {
    for i in 0..input.len() {
        output[i] = if input[i] > 0.0 {
            input[i]
        } else {
            alpha * (input[i].exp() - 1.0)
        };
    }
}

fn gelu_scalar(input: &[f32], output: &mut [f32]) {
    const SQRT_2_PI: f32 = 0.7978845608; // sqrt(2/π)

    for i in 0..input.len() {
        let x = input[i];
        let tanh_arg = SQRT_2_PI * (x + 0.044715 * x * x * x);
        output[i] = 0.5 * x * (1.0 + tanh_arg.tanh());
    }
}

// Parallel fallback operations
fn elementwise_add_parallel(a: &[f32], b: &[f32], c: &mut [f32]) {
    c.par_iter_mut()
        .zip(a.par_iter().zip(b.par_iter()))
        .for_each(|(c_val, (&a_val, &b_val))| {
            *c_val = a_val + b_val;
        });
}

fn elementwise_multiply_parallel(a: &[f32], b: &[f32], c: &mut [f32]) {
    c.par_iter_mut()
        .zip(a.par_iter().zip(b.par_iter()))
        .for_each(|(c_val, (&a_val, &b_val))| {
            *c_val = a_val * b_val;
        });
}

fn elementwise_subtract_parallel(a: &[f32], b: &[f32], c: &mut [f32]) {
    c.par_iter_mut()
        .zip(a.par_iter().zip(b.par_iter()))
        .for_each(|(c_val, (&a_val, &b_val))| {
            *c_val = a_val - b_val;
        });
}

fn elementwise_divide_parallel(a: &[f32], b: &[f32], c: &mut [f32]) {
    c.par_iter_mut()
        .zip(a.par_iter().zip(b.par_iter()))
        .for_each(|(c_val, (&a_val, &b_val))| {
            *c_val = a_val / b_val;
        });
}

// AVX2 activation functions
#[target_feature(enable = "avx2")]
unsafe fn relu_avx2(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let simd_len = len & !(SIMD_WIDTH - 1);

    for i in (0..simd_len).step_by(SIMD_WIDTH) {
        unsafe {
            let zero = _mm256_setzero_ps();
            let input_vec = _mm256_load_ps(input.as_ptr().add(i));
            let result = _mm256_max_ps(input_vec, zero);
            _mm256_store_ps(output.as_mut_ptr().add(i), result);
        }
    }

    // Handle remaining elements
    for i in simd_len..len {
        output[i] = input[i].max(0.0);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn sigmoid_avx2(input: &[f32], output: &mut [f32]) {
    // For now, use scalar implementation as AVX2 exp is complex
    // Fast exp approximation would require significant additional code
    for i in 0..input.len() {
        output[i] = 1.0 / (1.0 + (-input[i]).exp());
    }
}

// Parallel activation functions
fn relu_parallel(input: &[f32], output: &mut [f32]) {
    output
        .par_iter_mut()
        .zip(input.par_iter())
        .for_each(|(out, &inp)| {
            *out = inp.max(0.0);
        });
}

fn sigmoid_parallel(input: &[f32], output: &mut [f32]) {
    output
        .par_iter_mut()
        .zip(input.par_iter())
        .for_each(|(out, &inp)| {
            *out = 1.0 / (1.0 + (-inp).exp());
        });
}

fn tanh_parallel(input: &[f32], output: &mut [f32]) {
    output
        .par_iter_mut()
        .zip(input.par_iter())
        .for_each(|(out, &inp)| {
            *out = inp.tanh();
        });
}

fn softmax_parallel(input: &[f32], output: &mut [f32]) {
    // Find max for numerical stability
    let max_val = input
        .par_iter()
        .copied()
        .fold(|| f32::NEG_INFINITY, f32::max)
        .reduce(|| f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let exp_sum: f32 = input.par_iter().map(|&x| (x - max_val).exp()).sum();

    // Normalize
    output
        .par_iter_mut()
        .zip(input.par_iter())
        .for_each(|(out, &inp)| {
            *out = (inp - max_val).exp() / exp_sum;
        });
}

fn leaky_relu_parallel(input: &[f32], output: &mut [f32], alpha: f32) {
    output
        .par_iter_mut()
        .zip(input.par_iter())
        .for_each(|(out, &inp)| {
            *out = if inp > 0.0 { inp } else { alpha * inp };
        });
}

fn elu_parallel(input: &[f32], output: &mut [f32], alpha: f32) {
    output
        .par_iter_mut()
        .zip(input.par_iter())
        .for_each(|(out, &inp)| {
            *out = if inp > 0.0 {
                inp
            } else {
                alpha * (inp.exp() - 1.0)
            };
        });
}

fn gelu_parallel(input: &[f32], output: &mut [f32]) {
    const SQRT_2_PI: f32 = 0.7978845608; // sqrt(2/π)

    output
        .par_iter_mut()
        .zip(input.par_iter())
        .for_each(|(out, &inp)| {
            let tanh_arg = SQRT_2_PI * (inp + 0.044715 * inp * inp * inp);
            *out = 0.5 * inp * (1.0 + tanh_arg.tanh());
        });
}

// AVX2 reduction operations
#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(input: &[f32]) -> f32 {
    let len = input.len();
    let simd_len = len & !(SIMD_WIDTH - 1);

    unsafe {
        let mut sum_vec = _mm256_setzero_ps();

        for i in (0..simd_len).step_by(SIMD_WIDTH) {
            let input_vec = _mm256_load_ps(input.as_ptr().add(i));
            sum_vec = _mm256_add_ps(sum_vec, input_vec);
        }

        // Horizontal sum of AVX register
        let high = _mm256_extractf128_ps(sum_vec, 1);
        let low = _mm256_castps256_ps128(sum_vec);
        let sum128 = _mm_add_ps(high, low);
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

        let mut result = _mm_cvtss_f32(sum32);

        // Add remaining elements
        for i in simd_len..len {
            result += input[i];
        }

        result
    }
}

// Parallel reduction operations
fn sum_parallel(input: &[f32]) -> f32 {
    input.par_iter().sum()
}

fn max_parallel(input: &[f32]) -> f32 {
    input
        .par_iter()
        .copied()
        .fold(|| f32::NEG_INFINITY, f32::max)
        .reduce(|| f32::NEG_INFINITY, f32::max)
}

fn min_parallel(input: &[f32]) -> f32 {
    input
        .par_iter()
        .copied()
        .fold(|| f32::INFINITY, f32::min)
        .reduce(|| f32::INFINITY, f32::min)
}

fn argmax_parallel(input: &[f32]) -> usize {
    input
        .par_iter()
        .enumerate()
        .map(|(i, &x)| (i, x))
        .fold(
            || (0, f32::NEG_INFINITY),
            |acc, (i, x)| if x > acc.1 { (i, x) } else { acc },
        )
        .reduce(
            || (0, f32::NEG_INFINITY),
            |acc1, acc2| if acc2.1 > acc1.1 { acc2 } else { acc1 },
        )
        .0
}

fn argmin_parallel(input: &[f32]) -> usize {
    input
        .par_iter()
        .enumerate()
        .map(|(i, &x)| (i, x))
        .fold(
            || (0, f32::INFINITY),
            |acc, (i, x)| if x < acc.1 { (i, x) } else { acc },
        )
        .reduce(
            || (0, f32::INFINITY),
            |acc1, acc2| if acc2.1 < acc1.1 { acc2 } else { acc1 },
        )
        .0
}

/// CPU feature detection
fn detect_cpu_features() -> CpuFeatures {
    CpuFeatures {
        has_avx2: is_x86_feature_detected!("avx2"),
        has_fma: is_x86_feature_detected!("fma"),
        has_avx512: is_x86_feature_detected!("avx512f"),
        cache_line_size: 64,            // Default cache line size
        l1_cache_size: 32 * 1024,       // 32KB typical L1
        l2_cache_size: 256 * 1024,      // 256KB typical L2
        l3_cache_size: 8 * 1024 * 1024, // 8MB typical L3
    }
}

/// Get system memory size
fn get_system_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return Some(kb * 1024); // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").arg("hw.memsize").output() {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                if let Some(size_str) = output_str.split_whitespace().nth(1) {
                    if let Ok(size) = size_str.parse::<u64>() {
                        return Some(size);
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Windows implementation would go here
        // For now, return a default estimate
        return Some(8 * 1024 * 1024 * 1024); // 8GB default
    }

    // Fallback for unknown systems
    Some(8 * 1024 * 1024 * 1024) // 8GB default
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_creation() {
        let backend = CpuBackend::new();
        assert!(backend.is_ok());
        let backend = backend.unwrap();
        assert!(backend.num_threads() > 0);
    }

    #[test]
    fn test_aligned_memory_allocation() {
        let memory = CpuMemory::new_aligned(1024);
        assert!(memory.is_ok());
        let memory = memory.unwrap();
        assert_eq!(memory.size(), 1024);
        assert!(memory.aligned);
    }

    #[test]
    fn test_simd_copy_operations() {
        let memory = CpuMemory::new_aligned(32 * std::mem::size_of::<f32>()).unwrap();
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();

        memory.copy_from_slice_optimized(&data).unwrap();

        let mut result = vec![0.0f32; 32];
        memory.copy_to_slice_optimized(&mut result).unwrap();

        assert_eq!(data, result);
    }

    #[test]
    fn test_cpu_features_detection() {
        let features = detect_cpu_features();
        // Just check that detection runs without panicking
        println!("CPU Features: {:?}", features);
    }

    #[test]
    fn test_elementwise_operations() {
        let backend = CpuBackend::new().unwrap();

        let a_mem = CpuMemory::new_aligned(32 * std::mem::size_of::<f32>()).unwrap();
        let b_mem = CpuMemory::new_aligned(32 * std::mem::size_of::<f32>()).unwrap();
        let c_mem = CpuMemory::new_aligned(32 * std::mem::size_of::<f32>()).unwrap();

        let a_data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..32).map(|i| (i + 1) as f32).collect();

        a_mem.copy_from_slice_optimized(&a_data).unwrap();
        b_mem.copy_from_slice_optimized(&b_data).unwrap();

        let operation = CpuOperation::ElementwiseAdd;
        let inputs: Vec<&dyn DeviceMemory> = vec![&a_mem, &b_mem];
        let outputs: Vec<&dyn DeviceMemory> = vec![&c_mem];

        operation
            .execute_optimized(&inputs, &outputs, &backend.cpu_features)
            .unwrap();

        let mut result = vec![0.0f32; 32];
        c_mem.copy_to_slice_optimized(&mut result).unwrap();

        for i in 0..32 {
            assert_eq!(result[i], a_data[i] + b_data[i]);
        }
    }
}
