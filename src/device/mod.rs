//! Device and backend abstraction layer
//!
//! This module provides a unified interface for different compute backends
//! including CPU, and Vulkan compute shaders.

use crate::error::Result;
use std::cell::RefCell;
use std::fmt;
use std::sync::Arc;

pub mod cpu;
pub mod gpu;

/// Available device types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU execution with SIMD optimizations
    Cpu,
    /// Vulkan compute shaders (AMD/Intel/NVIDIA)
    Vulkan,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::Vulkan => write!(f, "Vulkan"),
        }
    }
}

// Thread-local default device to prevent unwanted auto-selection
thread_local! {
    static DEFAULT_DEVICE: RefCell<Option<Device>> = RefCell::new(None);
}

/// Set the thread-local default device
pub fn set_default_device(device: Device) {
    DEFAULT_DEVICE.with(|d| {
        *d.borrow_mut() = Some(device);
    });
}

/// Clear the thread-local default device
pub fn clear_default_device() {
    DEFAULT_DEVICE.with(|d| {
        *d.borrow_mut() = None;
    });
}

/// Device information and capabilities
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name identifier
    pub name: String,
    /// Type of the device (CPU, Vulkan, etc.)
    pub device_type: DeviceType,
    /// Available memory in bytes
    pub memory_size: Option<u64>,
    /// Number of compute units (cores, SMs, etc.)
    pub compute_units: Option<u32>,
    /// Whether the device supports half-precision floating point
    pub supports_f16: bool,
    /// Whether the device supports double-precision floating point
    pub supports_f64: bool,
}

/// Unified device abstraction
#[derive(Clone)]
pub struct Device {
    backend: Arc<dyn Backend + Send + Sync>,
    info: DeviceInfo,
}

impl Device {
    /// Create a new device with the specified backend
    pub fn new(backend: Arc<dyn Backend + Send + Sync>, info: DeviceInfo) -> Self {
        Self { backend, info }
    }

    /// Auto-select the best available device
    pub fn auto_select() -> Result<Self> {
        // Check for thread-local default device first
        let default_device = DEFAULT_DEVICE.with(|d| d.borrow().clone());
        if let Some(device) = default_device {
            return Ok(device);
        }

        // Check if CPU-only mode is requested via environment variable
        if std::env::var("NNL_CPU_ONLY").is_ok() {
            log::info!("NNL_CPU_ONLY set, using CPU device");
            return Self::cpu();
        }

        // Try GPU backends first, then fall back to CPU
        if let Ok(device) = Self::vulkan() {
            log::info!("Selected Vulkan device: {}", device.info.name);
            return Ok(device);
        }

        // Fall back to CPU
        let device = Self::cpu()?;
        log::info!("Selected CPU device: {}", device.info.name);
        Ok(device)
    }

    /// Create a CPU device
    pub fn cpu() -> Result<Self> {
        let backend = Arc::new(cpu::CpuBackend::new()?);
        let info = DeviceInfo {
            name: "CPU".to_string(),
            device_type: DeviceType::Cpu,
            memory_size: None, // System RAM
            compute_units: Some(num_cpus::get() as u32),
            supports_f16: false,
            supports_f64: true,
        };
        let device = Self::new(backend, info);

        // Set this CPU device as the thread-local default
        set_default_device(device.clone());
        Ok(device)
    }

    /// Create a Vulkan device
    pub fn vulkan() -> Result<Self> {
        // Check if CPU-only mode is requested
        if std::env::var("NNL_CPU_ONLY").is_ok() {
            return Err(crate::error::NnlError::device(
                "Vulkan device creation blocked by NNL_CPU_ONLY environment variable",
            ));
        }

        let backend = Arc::new(gpu::VulkanBackend::new()?);
        let info = backend.device_info()?;
        Ok(Self::new(backend, info))
    }

    /// Get device information
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Get device type
    pub fn device_type(&self) -> DeviceType {
        self.info.device_type
    }

    /// Get the backend reference
    pub fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }

    /// Check if device supports half precision
    pub fn supports_f16(&self) -> bool {
        self.info.supports_f16
    }

    /// Check if device supports double precision
    pub fn supports_f64(&self) -> bool {
        self.info.supports_f64
    }

    /// Get available memory in bytes
    pub fn memory_size(&self) -> Option<u64> {
        self.info.memory_size
    }

    /// Synchronize device operations
    pub fn synchronize(&self) -> Result<()> {
        self.backend.synchronize()
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device").field("info", &self.info).finish()
    }
}

/// Backend trait for different compute devices
pub trait Backend {
    /// Get device information
    fn device_info(&self) -> Result<DeviceInfo>;

    /// Allocate memory on device
    fn allocate(&self, size: usize) -> Result<Arc<dyn DeviceMemory>>;

    /// Allocate uniform buffer on device
    fn allocate_uniform(&self, size: usize) -> Result<Arc<dyn DeviceMemory>> {
        // Default implementation delegates to regular allocate for backwards compatibility
        self.allocate(size)
    }

    /// Copy data from host to device
    fn copy_to_device(&self, data: &[f32], memory: &dyn DeviceMemory) -> Result<()>;

    /// Copy u32 data from host to device (for uniform buffers)
    fn copy_u32_to_device(&self, data: &[u32], memory: &dyn DeviceMemory) -> Result<()> {
        // Default implementation converts u32 to f32 for backwards compatibility
        let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        self.copy_to_device(&f32_data, memory)
    }

    /// Copy data from device to host
    fn copy_to_host(&self, memory: &dyn DeviceMemory, data: &mut [f32]) -> Result<()>;

    /// Execute a kernel/compute shader
    fn execute_kernel(
        &self,
        kernel: &dyn Kernel,
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
    ) -> Result<()>;

    /// Execute a kernel with an optional uniform buffer
    fn execute_kernel_with_uniform(
        &self,
        kernel: &dyn Kernel,
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        uniform: Option<&dyn DeviceMemory>,
    ) -> Result<()> {
        // Default implementation delegates to execute_kernel for backwards compatibility
        if uniform.is_some() {
            return Err(crate::error::NnlError::device(
                "Uniform buffers not supported by this backend",
            ));
        }
        self.execute_kernel(kernel, inputs, outputs)
    }

    /// Synchronize all operations
    fn synchronize(&self) -> Result<()>;

    /// Check if backend is available
    fn is_available(&self) -> bool;

    /// Downcast to any for type checking
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Device memory abstraction
pub trait DeviceMemory: std::fmt::Debug + Send + Sync {
    /// Get memory size in bytes
    fn size(&self) -> usize;

    /// Get device type
    fn device_type(&self) -> DeviceType;

    /// Downcast to any for type checking
    fn as_any(&self) -> &dyn std::any::Any;

    /// Downcast to mutable any for type checking
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Kernel/compute shader abstraction
pub trait Kernel {
    /// Get kernel name/identifier
    fn name(&self) -> &str;

    /// Get required local work group size
    fn local_size(&self) -> Option<[u32; 3]>;

    /// Downcast to any for type checking
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Utility functions for device detection and benchmarking
pub mod utils {
    use super::*;

    /// List all available devices
    pub fn list_devices() -> Vec<DeviceInfo> {
        let mut devices = Vec::new();

        // Always have CPU
        if let Ok(cpu) = Device::cpu() {
            devices.push(cpu.info().clone());
        }

        if let Ok(vulkan) = Device::vulkan() {
            devices.push(vulkan.info().clone());
        }

        devices
    }

    /// Benchmark devices for selection
    pub fn benchmark_devices() -> Result<Vec<(DeviceInfo, f64)>> {
        let devices = list_devices();
        let mut results = Vec::new();

        for device_info in devices {
            let _device = match device_info.device_type {
                DeviceType::Cpu => Device::cpu()?,
                DeviceType::Vulkan => Device::vulkan()?,
            };

            let start = std::time::Instant::now();
            // Run a simple benchmark kernel
            benchmark_matrix_multiply(&_device)?;
            let duration = start.elapsed().as_secs_f64();

            results.push((device_info, duration));
        }

        // Sort by performance (lower time is better)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(results)
    }

    fn benchmark_matrix_multiply(_device: &Device) -> Result<()> {
        // Simple matrix multiplication benchmark
        // This is a placeholder - real implementation would depend on kernel system
        std::thread::sleep(std::time::Duration::from_millis(1));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_auto_select() {
        let device = Device::auto_select();
        assert!(device.is_ok());
        let device = device.unwrap();
        println!("Auto-selected device: {:?}", device.device_type());
    }

    #[test]
    fn test_cpu_device() {
        let device = Device::cpu();
        assert!(device.is_ok());
        let device = device.unwrap();
        assert_eq!(device.device_type(), DeviceType::Cpu);
        assert!(device.supports_f64());
    }

    #[test]
    fn test_list_devices() {
        let devices = utils::list_devices();
        assert!(!devices.is_empty());
        // Should at least have CPU
        assert!(devices.iter().any(|d| d.device_type == DeviceType::Cpu));
    }

    #[test]
    fn test_device_info_display() {
        let info = DeviceInfo {
            name: "Test Device".to_string(),
            device_type: DeviceType::Cpu,
            memory_size: Some(8_000_000_000),
            compute_units: Some(8),
            supports_f16: false,
            supports_f64: true,
        };

        assert_eq!(format!("{}", info.device_type), "CPU");
    }
}
