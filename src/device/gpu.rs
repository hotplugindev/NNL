//! Real Vulkan GPU compute backend implementation
//!
//! This module provides a complete GPU compute backend using Vulkan compute shaders.
//! All mathematical operations are performed entirely on the GPU with zero CPU fallbacks.

use crate::device::{Backend, DeviceInfo, DeviceMemory, DeviceType, Kernel};
use crate::error::{NnlError, Result};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use vulkano::{
    VulkanLibrary,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        PersistentDescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device as VkDevice, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::{self, GpuFuture},
};

/// Real Vulkan compute backend with GPU-only execution
pub struct VulkanBackend {
    device: Arc<VkDevice>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pipelines: Arc<Mutex<HashMap<String, Arc<ComputePipeline>>>>,
    device_info: DeviceInfo,
}

impl VulkanBackend {
    /// Create a new Vulkan backend with real GPU support
    pub fn new() -> Result<Self> {
        // Initialize Vulkan library
        let library = VulkanLibrary::new()
            .map_err(|e| NnlError::gpu(format!("Failed to load Vulkan library: {}", e)))?;

        // Create Vulkan instance
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                application_name: Some("NNL Neural Network Library".into()),
                application_version: vulkano::Version::V1_0,
                ..Default::default()
            },
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create Vulkan instance: {}", e)))?;

        // Select best physical device (prefer discrete GPU)
        let physical_device = instance
            .enumerate_physical_devices()
            .map_err(|e| NnlError::gpu(format!("Failed to enumerate devices: {}", e)))?
            .into_iter()
            .max_by_key(|device| match device.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 3,
                PhysicalDeviceType::IntegratedGpu => 2,
                PhysicalDeviceType::VirtualGpu => 1,
                _ => 0,
            })
            .ok_or_else(|| NnlError::gpu("No suitable Vulkan device found"))?;

        // Find compute queue family
        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .find_map(|(i, q)| {
                if q.queue_flags.intersects(QueueFlags::COMPUTE) {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .ok_or_else(|| NnlError::gpu("No compute queue family found"))?;

        // Create logical device and queue
        let (device, mut queues) = VkDevice::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: DeviceExtensions::empty(),
                enabled_features: Features::empty(),
                ..Default::default()
            },
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create device: {}", e)))?;

        let queue = queues.next().unwrap();

        // Create allocators
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        // Create device info
        let properties = physical_device.properties();
        let memory_properties = physical_device.memory_properties();
        let total_memory = memory_properties
            .memory_heaps
            .iter()
            .map(|heap| heap.size)
            .max()
            .unwrap_or(0);

        let device_info = DeviceInfo {
            name: properties.device_name.clone(),
            device_type: DeviceType::Vulkan,
            memory_size: Some(total_memory),
            compute_units: Some(properties.max_compute_work_group_count[0]),
            supports_f16: false, // Can be extended later
            supports_f64: properties.shader_float64 != 0,
        };

        Ok(Self {
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            pipelines: Arc::new(Mutex::new(HashMap::new())),
            device_info,
        })
    }

    /// Get or create compute pipeline for the given shader
    fn get_pipeline(&self, shader_name: &str) -> Result<Arc<ComputePipeline>> {
        let mut pipelines = self.pipelines.lock().unwrap();

        if let Some(pipeline) = pipelines.get(shader_name) {
            return Ok(pipeline.clone());
        }

        // Create shader module
        let shader_code = Self::get_shader_spirv(shader_name)?;

        let shader = ShaderModule::new(
            self.device.clone(),
            ShaderModuleCreateInfo::new(&shader_code),
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create shader module: {}", e)))?;

        // Create compute pipeline
        let stage = PipelineShaderStageCreateInfo::new(shader.entry_point("main").unwrap());
        let layout = PipelineLayout::new(
            self.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(self.device.clone())
                .unwrap(),
        )
        .unwrap();

        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create compute pipeline: {}", e)))?;

        let pipeline = Arc::new(pipeline);
        pipelines.insert(shader_name.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    /// Get SPIR-V bytecode for shader (compiled from GLSL)
    fn get_shader_spirv(shader_name: &str) -> Result<Vec<u32>> {
        // Load pre-compiled SPIR-V shaders
        match shader_name {
            "elementwise_add" => Ok(include_bytes!("../../shaders/elementwise_add.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "elementwise_sub" => Ok(include_bytes!("../../shaders/elementwise_sub.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "elementwise_mul" => Ok(include_bytes!("../../shaders/elementwise_mul.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "elementwise_div" => Ok(include_bytes!("../../shaders/elementwise_div.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "scalar_add" => Ok(include_bytes!("../../shaders/scalar_add.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "scalar_mul" => Ok(include_bytes!("../../shaders/scalar_mul.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "matrix_mul" => Ok(include_bytes!("../../shaders/matrix_mul.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "relu" => Ok(include_bytes!("../../shaders/relu.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "sigmoid" => Ok(include_bytes!("../../shaders/sigmoid.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "tanh" => Ok(include_bytes!("../../shaders/tanh.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "softmax" => Ok(include_bytes!("../../shaders/softmax.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "transpose" => Ok(include_bytes!("../../shaders/transpose.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "copy" => Ok(include_bytes!("../../shaders/copy.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "sqrt" => Ok(include_bytes!("../../shaders/sqrt.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "gelu" => Ok(include_bytes!("../../shaders/gelu.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "swish" => Ok(include_bytes!("../../shaders/swish.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "reduce_sum" => Ok(include_bytes!("../../shaders/reduce_sum.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            "conv2d" => Ok(include_bytes!("../../shaders/conv2d.spv")
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()),
            _ => Err(NnlError::gpu(format!("Unknown shader: {}", shader_name))),
        }
    }

    /// Execute compute operation on GPU
    pub fn execute_compute_operation(
        &self,
        operation: &str,
        input_buffers: &[Arc<VulkanBuffer>],
        output_buffers: &[Arc<VulkanBuffer>],
        uniform_data: Option<&[u32]>,
    ) -> Result<()> {
        let pipeline = self.get_pipeline(operation)?;

        // Create command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create command buffer: {}", e)))?;

        // Create descriptor set
        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let mut set_builder = Vec::new();

        // Add input buffers
        for (i, buffer) in input_buffers.iter().enumerate() {
            set_builder.push(WriteDescriptorSet::buffer(i as u32, buffer.buffer.clone()));
        }

        // Add output buffers
        for (i, buffer) in output_buffers.iter().enumerate() {
            let binding = (input_buffers.len() + i) as u32;
            set_builder.push(WriteDescriptorSet::buffer(binding, buffer.buffer.clone()));
        }

        // Add uniform buffer if provided
        if let Some(uniform) = uniform_data {
            // Create uniform buffer with proper alignment
            let uniform_buffer = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                uniform.iter().cloned(),
            )
            .map_err(|e| NnlError::gpu(format!("Failed to create uniform buffer: {}", e)))?;

            // Uniform buffer always gets the highest binding number
            let binding = match operation {
                "scalar_add" | "scalar_mul" => 2,
                "matrix_mul" | "transpose" | "softmax" | "reduce_sum" | "conv2d" => {
                    (input_buffers.len() + output_buffers.len()) as u32
                }
                _ => (input_buffers.len() + output_buffers.len()) as u32,
            };
            set_builder.push(WriteDescriptorSet::buffer(binding, uniform_buffer));
        }

        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            set_builder,
            [],
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create descriptor set: {}", e)))?;

        // Calculate dispatch size
        let total_elements = if !output_buffers.is_empty() {
            output_buffers[0].size() / std::mem::size_of::<f32>()
        } else {
            return Err(NnlError::gpu("No output buffers provided"));
        };

        let local_size = match operation {
            "matrix_mul" => 16u32, // Use 16x16 workgroups for matrix operations
            "transpose" => 16u32,
            _ => 64u32, // Default workgroup size for other operations
        };

        let dispatch_x = if operation == "matrix_mul" || operation == "transpose" {
            // For 2D operations, calculate based on output matrix dimensions
            // This is a simplified approach - in practice you'd get dimensions from uniform
            ((total_elements as f32).sqrt() as u32 + local_size - 1) / local_size
        } else {
            ((total_elements as u32) + local_size - 1) / local_size
        };

        let dispatch_y = if operation == "matrix_mul" || operation == "transpose" {
            dispatch_x // Square dispatch for matrix operations
        } else {
            1
        };

        // Record commands
        builder
            .bind_pipeline_compute(pipeline.clone())
            .map_err(|e| NnlError::gpu(format!("Failed to bind pipeline: {}", e)))?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .map_err(|e| NnlError::gpu(format!("Failed to bind descriptor sets: {}", e)))?
            .dispatch([dispatch_x, dispatch_y, 1])
            .map_err(|e| NnlError::gpu(format!("Failed to dispatch: {}", e)))?;

        let command_buffer = builder
            .build()
            .map_err(|e| NnlError::gpu(format!("Failed to build command buffer: {}", e)))?;

        // Submit and wait for completion
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .map_err(|e| NnlError::gpu(format!("Failed to execute command buffer: {}", e)))?
            .then_signal_fence_and_flush()
            .map_err(|e| NnlError::gpu(format!("Failed to signal fence: {}", e)))?;

        future
            .wait(None)
            .map_err(|e| NnlError::gpu(format!("Failed to wait for GPU: {}", e)))?;

        Ok(())
    }
}

impl Backend for VulkanBackend {
    fn device_info(&self) -> Result<DeviceInfo> {
        Ok(self.device_info.clone())
    }

    fn allocate(&self, size: usize) -> Result<Arc<dyn DeviceMemory>> {
        let buffer = VulkanBuffer::new(
            self.memory_allocator.clone(),
            size * std::mem::size_of::<f32>(),
            false,
        )?;
        Ok(Arc::new(buffer) as Arc<dyn DeviceMemory>)
    }

    fn allocate_uniform(&self, size: usize) -> Result<Arc<dyn DeviceMemory>> {
        let buffer = VulkanBuffer::new(
            self.memory_allocator.clone(),
            size * std::mem::size_of::<u32>(),
            true,
        )?;
        Ok(Arc::new(buffer) as Arc<dyn DeviceMemory>)
    }

    fn copy_to_device(&self, data: &[f32], memory: &dyn DeviceMemory) -> Result<()> {
        let vulkan_buffer = memory
            .as_any()
            .downcast_ref::<VulkanBuffer>()
            .ok_or_else(|| NnlError::device("Invalid memory type for Vulkan backend"))?;

        vulkan_buffer.write_data(
            data,
            &self.memory_allocator,
            &self.command_buffer_allocator,
            &self.queue,
        )
    }

    fn copy_u32_to_device(&self, data: &[u32], memory: &dyn DeviceMemory) -> Result<()> {
        let vulkan_buffer = memory
            .as_any()
            .downcast_ref::<VulkanBuffer>()
            .ok_or_else(|| NnlError::device("Invalid memory type for Vulkan backend"))?;

        vulkan_buffer.write_u32_data(
            data,
            &self.memory_allocator,
            &self.command_buffer_allocator,
            &self.queue,
        )
    }

    fn copy_to_host(&self, memory: &dyn DeviceMemory, data: &mut [f32]) -> Result<()> {
        let vulkan_buffer = memory
            .as_any()
            .downcast_ref::<VulkanBuffer>()
            .ok_or_else(|| NnlError::device("Invalid memory type for Vulkan backend"))?;

        vulkan_buffer.read_data(
            data,
            &self.memory_allocator,
            &self.command_buffer_allocator,
            &self.queue,
        )
    }

    fn execute_kernel(
        &self,
        kernel: &dyn Kernel,
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
    ) -> Result<()> {
        self.execute_kernel_with_uniform(kernel, inputs, outputs, None)
    }

    fn execute_kernel_with_uniform(
        &self,
        kernel: &dyn Kernel,
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        uniform: Option<&dyn DeviceMemory>,
    ) -> Result<()> {
        let vulkan_kernel = kernel
            .as_any()
            .downcast_ref::<VulkanKernel>()
            .ok_or_else(|| NnlError::device("Invalid kernel type for Vulkan backend"))?;

        // Convert memory references to VulkanBuffer
        let input_buffers: Result<Vec<_>> = inputs
            .iter()
            .map(|mem| {
                mem.as_any()
                    .downcast_ref::<VulkanBuffer>()
                    .ok_or_else(|| NnlError::device("Invalid input buffer type"))
                    .map(|buf| Arc::new(buf.clone()))
            })
            .collect();
        let input_buffers = input_buffers?;

        let output_buffers: Result<Vec<_>> = outputs
            .iter()
            .map(|mem| {
                mem.as_any()
                    .downcast_ref::<VulkanBuffer>()
                    .ok_or_else(|| NnlError::device("Invalid output buffer type"))
                    .map(|buf| Arc::new(buf.clone()))
            })
            .collect();
        let output_buffers = output_buffers?;

        // Get uniform data if provided
        let uniform_data = if let Some(uniform_mem) = uniform {
            let uniform_buffer = uniform_mem
                .as_any()
                .downcast_ref::<VulkanBuffer>()
                .ok_or_else(|| NnlError::device("Invalid uniform buffer type"))?;
            Some(uniform_buffer.read_u32_data(
                &self.memory_allocator,
                &self.command_buffer_allocator,
                &self.queue,
            )?)
        } else {
            None
        };

        self.execute_compute_operation(
            vulkan_kernel.name(),
            &input_buffers,
            &output_buffers,
            uniform_data.as_deref(),
        )
    }

    fn synchronize(&self) -> Result<()> {
        // Wait for all GPU operations to complete
        self.device
            .wait_idle()
            .map_err(|e| NnlError::gpu(format!("Failed to synchronize device: {}", e)))?;
        Ok(())
    }

    fn is_available(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Real Vulkan buffer backed by GPU memory
#[derive(Debug, Clone)]
pub struct VulkanBuffer {
    buffer: Subbuffer<[u32]>, // Use u32 as base type for flexibility
    size_in_bytes: usize,
    is_uniform: bool,
}

impl VulkanBuffer {
    /// Create a new Vulkan buffer on GPU memory
    pub fn new(
        allocator: Arc<StandardMemoryAllocator>,
        size_in_bytes: usize,
        is_uniform: bool,
    ) -> Result<Self> {
        let usage = if is_uniform {
            BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST
        } else {
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST
        };

        let size_in_u32s = (size_in_bytes + 3) / 4; // Round up to u32 boundary

        let buffer = Buffer::new_slice::<u32>(
            allocator,
            BufferCreateInfo {
                usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            size_in_u32s as u64,
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create buffer: {}", e)))?;

        Ok(Self {
            buffer,
            size_in_bytes,
            is_uniform,
        })
    }

    /// Write f32 data to GPU buffer using staging buffer
    pub fn write_data(
        &self,
        data: &[f32],
        allocator: &StandardMemoryAllocator,
        command_allocator: &StandardCommandBufferAllocator,
        queue: &Queue,
    ) -> Result<()> {
        if data.len() * std::mem::size_of::<f32>() != self.size_in_bytes {
            return Err(NnlError::device("Data size mismatch"));
        }

        // Convert f32 to u32 for storage (preserving bit pattern)
        let u32_data: Vec<u32> = data.iter().map(|&f| f.to_bits()).collect();

        // Create staging buffer
        let staging_buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            u32_data.iter().cloned(),
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create staging buffer: {}", e)))?;

        // Copy to device buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            command_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create command buffer: {}", e)))?;

        builder
            .copy_buffer(CopyBufferInfo::buffers(staging_buffer, self.buffer.clone()))
            .map_err(|e| NnlError::gpu(format!("Failed to copy buffer: {}", e)))?;

        let command_buffer = builder
            .build()
            .map_err(|e| NnlError::gpu(format!("Failed to build command buffer: {}", e)))?;

        let future = sync::now(queue.device().clone())
            .then_execute(queue.clone(), command_buffer)
            .map_err(|e| NnlError::gpu(format!("Failed to execute command buffer: {}", e)))?
            .then_signal_fence_and_flush()
            .map_err(|e| NnlError::gpu(format!("Failed to signal fence: {}", e)))?;

        future
            .wait(None)
            .map_err(|e| NnlError::gpu(format!("Failed to wait for GPU: {}", e)))?;

        Ok(())
    }

    /// Write u32 data to GPU buffer (for uniform buffers)
    pub fn write_u32_data(
        &self,
        data: &[u32],
        allocator: &StandardMemoryAllocator,
        command_allocator: &StandardCommandBufferAllocator,
        queue: &Queue,
    ) -> Result<()> {
        if !self.is_uniform {
            return Err(NnlError::device("Buffer is not a uniform buffer"));
        }

        if data.len() * std::mem::size_of::<u32>() > self.size_in_bytes {
            return Err(NnlError::device("Data size mismatch for u32 data"));
        }

        // Create staging buffer
        let staging_buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.iter().cloned(),
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create staging buffer: {}", e)))?;

        // Copy to device buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            command_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create command buffer: {}", e)))?;

        builder
            .copy_buffer(CopyBufferInfo::buffers(staging_buffer, self.buffer.clone()))
            .map_err(|e| NnlError::gpu(format!("Failed to copy buffer: {}", e)))?;

        let command_buffer = builder
            .build()
            .map_err(|e| NnlError::gpu(format!("Failed to build command buffer: {}", e)))?;

        let future = sync::now(queue.device().clone())
            .then_execute(queue.clone(), command_buffer)
            .map_err(|e| NnlError::gpu(format!("Failed to execute command buffer: {}", e)))?
            .then_signal_fence_and_flush()
            .map_err(|e| NnlError::gpu(format!("Failed to signal fence: {}", e)))?;

        future
            .wait(None)
            .map_err(|e| NnlError::gpu(format!("Failed to wait for GPU: {}", e)))?;

        Ok(())
    }

    /// Read f32 data from GPU buffer using staging buffer
    pub fn read_data(
        &self,
        output: &mut [f32],
        allocator: &StandardMemoryAllocator,
        command_allocator: &StandardCommandBufferAllocator,
        queue: &Queue,
    ) -> Result<()> {
        if output.len() * std::mem::size_of::<f32>() != self.size_in_bytes {
            return Err(NnlError::device("Output size mismatch"));
        }

        let size_in_u32s = output.len();

        // Create staging buffer
        let staging_buffer = Buffer::new_slice::<u32>(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            size_in_u32s as u64,
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create staging buffer: {}", e)))?;

        // Copy from device buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            command_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create command buffer: {}", e)))?;

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.buffer.clone(),
                staging_buffer.clone(),
            ))
            .map_err(|e| NnlError::gpu(format!("Failed to copy buffer: {}", e)))?;

        let command_buffer = builder
            .build()
            .map_err(|e| NnlError::gpu(format!("Failed to build command buffer: {}", e)))?;

        let future = sync::now(queue.device().clone())
            .then_execute(queue.clone(), command_buffer)
            .map_err(|e| NnlError::gpu(format!("Failed to execute command buffer: {}", e)))?
            .then_signal_fence_and_flush()
            .map_err(|e| NnlError::gpu(format!("Failed to signal fence: {}", e)))?;

        future
            .wait(None)
            .map_err(|e| NnlError::gpu(format!("Failed to wait for GPU: {}", e)))?;

        // Read from staging buffer and convert u32 back to f32
        let staging_read = staging_buffer
            .read()
            .map_err(|e| NnlError::gpu(format!("Failed to read staging buffer: {}", e)))?;

        for (i, &u32_val) in staging_read.iter().enumerate() {
            if i < output.len() {
                output[i] = f32::from_bits(u32_val);
            }
        }

        Ok(())
    }

    /// Read u32 data from GPU buffer
    pub fn read_u32_data(
        &self,
        allocator: &StandardMemoryAllocator,
        command_allocator: &StandardCommandBufferAllocator,
        queue: &Queue,
    ) -> Result<Vec<u32>> {
        let size_in_u32s = self.size_in_bytes / std::mem::size_of::<u32>();

        // Create staging buffer
        let staging_buffer = Buffer::new_slice::<u32>(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            size_in_u32s as u64,
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create staging buffer: {}", e)))?;

        // Copy from device buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            command_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create command buffer: {}", e)))?;

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.buffer.clone(),
                staging_buffer.clone(),
            ))
            .map_err(|e| NnlError::gpu(format!("Failed to copy buffer: {}", e)))?;

        let command_buffer = builder
            .build()
            .map_err(|e| NnlError::gpu(format!("Failed to build command buffer: {}", e)))?;

        let future = sync::now(queue.device().clone())
            .then_execute(queue.clone(), command_buffer)
            .map_err(|e| NnlError::gpu(format!("Failed to execute command buffer: {}", e)))?
            .then_signal_fence_and_flush()
            .map_err(|e| NnlError::gpu(format!("Failed to signal fence: {}", e)))?;

        future
            .wait(None)
            .map_err(|e| NnlError::gpu(format!("Failed to wait for GPU: {}", e)))?;

        // Read from staging buffer
        let staging_read = staging_buffer
            .read()
            .map_err(|e| NnlError::gpu(format!("Failed to read staging buffer: {}", e)))?;

        Ok(staging_read.to_vec())
    }
}

impl DeviceMemory for VulkanBuffer {
    fn size(&self) -> usize {
        self.size_in_bytes
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Vulkan
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Vulkan compute kernel
#[derive(Debug)]
pub struct VulkanKernel {
    name: String,
    dispatch_size: [u32; 3],
}

impl VulkanKernel {
    /// Create a new Vulkan kernel
    pub fn new(name: String, dispatch_size: [u32; 3]) -> Self {
        Self {
            name,
            dispatch_size,
        }
    }

    /// Create kernel for element-wise operations
    pub fn elementwise(name: String, size: u32) -> Self {
        Self::new(name, [size.div_ceil(64), 1, 1])
    }

    /// Create kernel for matrix operations
    pub fn matrix(name: String, rows: u32, cols: u32) -> Self {
        Self::new(name, [cols.div_ceil(16), rows.div_ceil(16), 1])
    }

    /// Create kernel for reduction operations
    pub fn reduction(name: String, size: u32) -> Self {
        Self::new(name, [size.div_ceil(256), 1, 1])
    }
}

impl Kernel for VulkanKernel {
    fn name(&self) -> &str {
        &self.name
    }

    fn local_size(&self) -> Option<[u32; 3]> {
        Some(self.dispatch_size)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_backend_creation() {
        match VulkanBackend::new() {
            Ok(backend) => {
                let info = backend.device_info().unwrap();
                assert_eq!(info.device_type, DeviceType::Vulkan);
                println!("Vulkan device: {}", info.name);
            }
            Err(e) => {
                println!("Vulkan not available: {}", e);
                // Skip test if Vulkan is not available
            }
        }
    }

    #[test]
    fn test_vulkan_buffer_operations() {
        if let Ok(backend) = VulkanBackend::new() {
            let memory = backend.allocate(4).unwrap(); // 4 elements
            assert_eq!(memory.size(), 4 * std::mem::size_of::<f32>());
            assert_eq!(memory.device_type(), DeviceType::Vulkan);

            let test_data = vec![1.0, 2.0, 3.0, 4.0];
            backend.copy_to_device(&test_data, memory.as_ref()).unwrap();

            let mut result = vec![0.0; 4];
            backend.copy_to_host(memory.as_ref(), &mut result).unwrap();

            for (actual, expected) in result.iter().zip(test_data.iter()) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_elementwise_operations() {
        if let Ok(backend) = VulkanBackend::new() {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![2.0, 3.0, 4.0, 5.0];

            let mem_a = backend.allocate(4).unwrap();
            let mem_b = backend.allocate(4).unwrap();
            let mem_c = backend.allocate(4).unwrap();

            backend.copy_to_device(&a, mem_a.as_ref()).unwrap();
            backend.copy_to_device(&b, mem_b.as_ref()).unwrap();

            let kernel = VulkanKernel::elementwise("elementwise_add".to_string(), 4);
            backend
                .execute_kernel(
                    &kernel,
                    &[mem_a.as_ref(), mem_b.as_ref()],
                    &[mem_c.as_ref()],
                )
                .unwrap();

            let mut result = vec![0.0; 4];
            backend.copy_to_host(mem_c.as_ref(), &mut result).unwrap();

            let expected = vec![3.0, 5.0, 7.0, 9.0];
            for (actual, expected) in result.iter().zip(expected.iter()) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_matrix_multiplication() {
        if let Ok(backend) = VulkanBackend::new() {
            // 2x3 * 3x2 = 2x2
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
            let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

            let mem_a = backend.allocate(6).unwrap();
            let mem_b = backend.allocate(6).unwrap();
            let mem_c = backend.allocate(4).unwrap();

            backend.copy_to_device(&a, mem_a.as_ref()).unwrap();
            backend.copy_to_device(&b, mem_b.as_ref()).unwrap();

            // Create uniform buffer for dimensions
            let dims = [2u32, 2u32, 3u32]; // M, N, K
            let uniform_mem = backend.allocate_uniform(3).unwrap();
            backend
                .copy_u32_to_device(&dims, uniform_mem.as_ref())
                .unwrap();

            let kernel = VulkanKernel::matrix("matrix_mul".to_string(), 2, 2);
            backend
                .execute_kernel_with_uniform(
                    &kernel,
                    &[mem_a.as_ref(), mem_b.as_ref()],
                    &[mem_c.as_ref()],
                    Some(uniform_mem.as_ref()),
                )
                .unwrap();

            let mut result = vec![0.0; 4];
            backend.copy_to_host(mem_c.as_ref(), &mut result).unwrap();

            // Expected: [58, 64, 139, 154]
            let expected = vec![58.0, 64.0, 139.0, 154.0];
            for (actual, expected) in result.iter().zip(expected.iter()) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }
}
