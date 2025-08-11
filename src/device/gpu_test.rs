//! Comprehensive tests for the Vulkan GPU backend
//!
//! This module provides extensive testing for the real Vulkan compute backend,
//! ensuring all operations work correctly on GPU hardware.

use crate::device::{Backend, Device, DeviceType, VulkanBackend, VulkanBuffer, VulkanKernel};
use crate::error::Result;
use std::sync::Arc;

/// Test basic Vulkan backend initialization
#[test]
fn test_vulkan_backend_initialization() {
    match VulkanBackend::new() {
        Ok(backend) => {
            let info = backend.device_info().unwrap();
            assert_eq!(info.device_type, DeviceType::Vulkan);
            assert!(info.memory_size.is_some());
            assert!(info.compute_units.is_some());
            println!("✓ Vulkan backend initialized: {}", info.name);
            println!("  Memory: {:?} MB", info.memory_size.unwrap() / 1_000_000);
            println!("  Compute units: {:?}", info.compute_units.unwrap());
        }
        Err(e) => {
            println!("⚠ Vulkan not available: {}", e);
            // Skip test if Vulkan is not available
            return;
        }
    }
}

/// Test GPU memory allocation and data transfer
#[test]
fn test_gpu_memory_operations() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return, // Skip if Vulkan not available
    };

    // Test various buffer sizes
    let test_sizes = [1, 4, 64, 256, 1024, 4096];

    for &size in &test_sizes {
        // Allocate GPU memory
        let memory = backend.allocate(size).unwrap();
        assert_eq!(memory.size(), size * std::mem::size_of::<f32>());
        assert_eq!(memory.device_type(), DeviceType::Vulkan);

        // Create test data
        let test_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.5).collect();

        // Transfer to GPU
        backend.copy_to_device(&test_data, memory.as_ref()).unwrap();

        // Transfer back from GPU
        let mut result = vec![0.0f32; size];
        backend.copy_to_host(memory.as_ref(), &mut result).unwrap();

        // Verify data integrity
        for (i, (&original, &retrieved)) in test_data.iter().zip(result.iter()).enumerate() {
            assert!(
                (original - retrieved).abs() < 1e-6,
                "Data mismatch at index {}: expected {}, got {}",
                i,
                original,
                retrieved
            );
        }

        println!("✓ GPU memory test passed for size {}", size);
    }
}

/// Test elementwise addition on GPU
#[test]
fn test_gpu_elementwise_addition() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let size = 1024;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();

    // Allocate GPU buffers
    let mem_a = backend.allocate(size).unwrap();
    let mem_b = backend.allocate(size).unwrap();
    let mem_c = backend.allocate(size).unwrap();

    // Transfer data to GPU
    backend.copy_to_device(&a, mem_a.as_ref()).unwrap();
    backend.copy_to_device(&b, mem_b.as_ref()).unwrap();

    // Execute GPU kernel
    let kernel = VulkanKernel::elementwise("elementwise_add".to_string(), size as u32);
    backend
        .execute_kernel(
            &kernel,
            &[mem_a.as_ref(), mem_b.as_ref()],
            &[mem_c.as_ref()],
        )
        .unwrap();

    // Get result from GPU
    let mut result = vec![0.0f32; size];
    backend.copy_to_host(mem_c.as_ref(), &mut result).unwrap();

    // Verify results
    for i in 0..size {
        let expected = a[i] + b[i];
        assert!(
            (result[i] - expected).abs() < 1e-5,
            "Addition failed at index {}: {} + {} = {}, got {}",
            i,
            a[i],
            b[i],
            expected,
            result[i]
        );
    }

    println!("✓ GPU elementwise addition test passed");
}

/// Test all elementwise operations
#[test]
fn test_gpu_all_elementwise_operations() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let size = 256;
    let a: Vec<f32> = (1..=size).map(|i| i as f32).collect();
    let b: Vec<f32> = (1..=size).map(|i| (i as f32) * 0.5).collect();

    let operations = [
        ("elementwise_add", |x: f32, y: f32| x + y),
        ("elementwise_sub", |x: f32, y: f32| x - y),
        ("elementwise_mul", |x: f32, y: f32| x * y),
        ("elementwise_div", |x: f32, y: f32| x / y),
    ];

    for (op_name, op_fn) in &operations {
        // Allocate GPU buffers
        let mem_a = backend.allocate(size).unwrap();
        let mem_b = backend.allocate(size).unwrap();
        let mem_c = backend.allocate(size).unwrap();

        // Transfer data to GPU
        backend.copy_to_device(&a, mem_a.as_ref()).unwrap();
        backend.copy_to_device(&b, mem_b.as_ref()).unwrap();

        // Execute GPU kernel
        let kernel = VulkanKernel::elementwise(op_name.to_string(), size as u32);
        backend
            .execute_kernel(
                &kernel,
                &[mem_a.as_ref(), mem_b.as_ref()],
                &[mem_c.as_ref()],
            )
            .unwrap();

        // Get result from GPU
        let mut result = vec![0.0f32; size];
        backend.copy_to_host(mem_c.as_ref(), &mut result).unwrap();

        // Verify results
        for i in 0..size {
            let expected = op_fn(a[i], b[i]);
            let error = (result[i] - expected).abs();
            assert!(
                error < 1e-4,
                "{} failed at index {}: {}({}, {}) = {}, got {}, error: {}",
                op_name,
                i,
                op_name,
                a[i],
                b[i],
                expected,
                result[i],
                error
            );
        }

        println!("✓ GPU {} test passed", op_name);
    }
}

/// Test scalar operations on GPU
#[test]
fn test_gpu_scalar_operations() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let size = 512;
    let input: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
    let scalar_add = 5.0f32;
    let scalar_mul = 2.5f32;

    // Test scalar addition
    {
        let mem_input = backend.allocate(size).unwrap();
        let mem_output = backend.allocate(size).unwrap();
        let mem_uniform = backend.allocate_uniform(1).unwrap();

        backend.copy_to_device(&input, mem_input.as_ref()).unwrap();
        backend
            .copy_u32_to_device(&[scalar_add.to_bits()], mem_uniform.as_ref())
            .unwrap();

        let kernel = VulkanKernel::elementwise("scalar_add".to_string(), size as u32);
        backend
            .execute_kernel_with_uniform(
                &kernel,
                &[mem_input.as_ref()],
                &[mem_output.as_ref()],
                Some(mem_uniform.as_ref()),
            )
            .unwrap();

        let mut result = vec![0.0f32; size];
        backend
            .copy_to_host(mem_output.as_ref(), &mut result)
            .unwrap();

        for i in 0..size {
            let expected = input[i] + scalar_add;
            assert!(
                (result[i] - expected).abs() < 1e-5,
                "Scalar add failed at {}: {} + {} = {}, got {}",
                i,
                input[i],
                scalar_add,
                expected,
                result[i]
            );
        }

        println!("✓ GPU scalar addition test passed");
    }

    // Test scalar multiplication
    {
        let mem_input = backend.allocate(size).unwrap();
        let mem_output = backend.allocate(size).unwrap();
        let mem_uniform = backend.allocate_uniform(1).unwrap();

        backend.copy_to_device(&input, mem_input.as_ref()).unwrap();
        backend
            .copy_u32_to_device(&[scalar_mul.to_bits()], mem_uniform.as_ref())
            .unwrap();

        let kernel = VulkanKernel::elementwise("scalar_mul".to_string(), size as u32);
        backend
            .execute_kernel_with_uniform(
                &kernel,
                &[mem_input.as_ref()],
                &[mem_output.as_ref()],
                Some(mem_uniform.as_ref()),
            )
            .unwrap();

        let mut result = vec![0.0f32; size];
        backend
            .copy_to_host(mem_output.as_ref(), &mut result)
            .unwrap();

        for i in 0..size {
            let expected = input[i] * scalar_mul;
            assert!(
                (result[i] - expected).abs() < 1e-5,
                "Scalar mul failed at {}: {} * {} = {}, got {}",
                i,
                input[i],
                scalar_mul,
                expected,
                result[i]
            );
        }

        println!("✓ GPU scalar multiplication test passed");
    }
}

/// Test matrix multiplication on GPU
#[test]
fn test_gpu_matrix_multiplication() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    // Test different matrix sizes
    let test_cases = [(2, 3, 2), (4, 4, 4), (8, 6, 4)];

    for (m, k, n) in test_cases {
        println!("Testing {}x{} * {}x{} matrix multiplication", m, k, k, n);

        // Create test matrices
        let a: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32 * 0.5).collect();

        // Allocate GPU memory
        let mem_a = backend.allocate(m * k).unwrap();
        let mem_b = backend.allocate(k * n).unwrap();
        let mem_c = backend.allocate(m * n).unwrap();
        let mem_uniform = backend.allocate_uniform(3).unwrap();

        // Transfer data
        backend.copy_to_device(&a, mem_a.as_ref()).unwrap();
        backend.copy_to_device(&b, mem_b.as_ref()).unwrap();
        backend
            .copy_u32_to_device(&[m as u32, n as u32, k as u32], mem_uniform.as_ref())
            .unwrap();

        // Execute matrix multiplication
        let kernel = VulkanKernel::matrix("matrix_mul".to_string(), m as u32, n as u32);
        backend
            .execute_kernel_with_uniform(
                &kernel,
                &[mem_a.as_ref(), mem_b.as_ref()],
                &[mem_c.as_ref()],
                Some(mem_uniform.as_ref()),
            )
            .unwrap();

        // Get result
        let mut result = vec![0.0f32; m * n];
        backend.copy_to_host(mem_c.as_ref(), &mut result).unwrap();

        // Verify using CPU computation
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                expected[i * n + j] = sum;
            }
        }

        for i in 0..m * n {
            assert!(
                (result[i] - expected[i]).abs() < 1e-3,
                "Matrix mul failed at {}: expected {}, got {}",
                i,
                expected[i],
                result[i]
            );
        }

        println!("✓ GPU matrix multiplication {}x{}x{} test passed", m, k, n);
    }
}

/// Test activation functions on GPU
#[test]
fn test_gpu_activation_functions() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let size = 1000;
    let input: Vec<f32> = (0..size)
        .map(|i| (i as f32 - 500.0) * 0.01) // Range from -5.0 to 5.0
        .collect();

    let activations = [
        ("relu", |x: f32| x.max(0.0)),
        ("sigmoid", |x: f32| 1.0 / (1.0 + (-x).exp())),
        ("tanh", |x: f32| x.tanh()),
    ];

    for (activation_name, activation_fn) in &activations {
        let mem_input = backend.allocate(size).unwrap();
        let mem_output = backend.allocate(size).unwrap();

        backend.copy_to_device(&input, mem_input.as_ref()).unwrap();

        let kernel = VulkanKernel::elementwise(activation_name.to_string(), size as u32);
        backend
            .execute_kernel(&kernel, &[mem_input.as_ref()], &[mem_output.as_ref()])
            .unwrap();

        let mut result = vec![0.0f32; size];
        backend
            .copy_to_host(mem_output.as_ref(), &mut result)
            .unwrap();

        for i in 0..size {
            let expected = activation_fn(input[i]);
            let error = (result[i] - expected).abs();

            // Use different tolerances for different functions
            let tolerance = match *activation_name {
                "sigmoid" | "tanh" => 1e-4, // More lenient for transcendental functions
                _ => 1e-6,
            };

            assert!(
                error < tolerance,
                "{} failed at index {}: {}({}) = {}, got {}, error: {}",
                activation_name,
                i,
                activation_name,
                input[i],
                expected,
                result[i],
                error
            );
        }

        println!("✓ GPU {} activation test passed", activation_name);
    }
}

/// Test transpose operation on GPU
#[test]
fn test_gpu_transpose() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    let test_cases = [(3, 4), (8, 6), (16, 12)];

    for (rows, cols) in test_cases {
        println!("Testing {}x{} matrix transpose", rows, cols);

        // Create test matrix
        let input: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();

        // Allocate GPU memory
        let mem_input = backend.allocate(rows * cols).unwrap();
        let mem_output = backend.allocate(rows * cols).unwrap();
        let mem_uniform = backend.allocate_uniform(2).unwrap();

        // Transfer data
        backend.copy_to_device(&input, mem_input.as_ref()).unwrap();
        backend
            .copy_u32_to_device(&[rows as u32, cols as u32], mem_uniform.as_ref())
            .unwrap();

        // Execute transpose
        let kernel = VulkanKernel::matrix("transpose".to_string(), cols as u32, rows as u32);
        backend
            .execute_kernel_with_uniform(
                &kernel,
                &[mem_input.as_ref()],
                &[mem_output.as_ref()],
                Some(mem_uniform.as_ref()),
            )
            .unwrap();

        // Get result
        let mut result = vec![0.0f32; rows * cols];
        backend
            .copy_to_host(mem_output.as_ref(), &mut result)
            .unwrap();

        // Verify transpose
        for i in 0..rows {
            for j in 0..cols {
                let original_idx = i * cols + j;
                let transposed_idx = j * rows + i;
                assert_eq!(
                    result[transposed_idx], input[original_idx],
                    "Transpose failed: input[{}][{}] = {} should be output[{}][{}] = {}",
                    i, j, input[original_idx], j, i, result[transposed_idx]
                );
            }
        }

        println!("✓ GPU transpose {}x{} test passed", rows, cols);
    }
}

/// Test GPU device integration with high-level API
#[test]
fn test_gpu_device_integration() {
    let device = match Device::vulkan() {
        Ok(d) => d,
        Err(_) => {
            println!("⚠ Vulkan device not available, skipping integration test");
            return;
        }
    };

    assert_eq!(device.device_type(), DeviceType::Vulkan);
    assert!(device.memory_size().is_some());

    // Test synchronization
    device.synchronize().unwrap();

    println!("✓ GPU device integration test passed");
}

/// Benchmark GPU vs CPU performance
#[test]
fn benchmark_gpu_performance() {
    let gpu_backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => {
            println!("⚠ Vulkan not available, skipping benchmark");
            return;
        }
    };

    let size = 10000;
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();

    // GPU benchmark
    let start = std::time::Instant::now();

    let mem_a = gpu_backend.allocate(size).unwrap();
    let mem_b = gpu_backend.allocate(size).unwrap();
    let mem_c = gpu_backend.allocate(size).unwrap();

    gpu_backend.copy_to_device(&a, mem_a.as_ref()).unwrap();
    gpu_backend.copy_to_device(&b, mem_b.as_ref()).unwrap();

    let kernel = VulkanKernel::elementwise("elementwise_add".to_string(), size as u32);
    gpu_backend
        .execute_kernel(
            &kernel,
            &[mem_a.as_ref(), mem_b.as_ref()],
            &[mem_c.as_ref()],
        )
        .unwrap();

    let mut gpu_result = vec![0.0f32; size];
    gpu_backend
        .copy_to_host(mem_c.as_ref(), &mut gpu_result)
        .unwrap();

    let gpu_time = start.elapsed();

    // CPU benchmark
    let start = std::time::Instant::now();
    let cpu_result: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
    let cpu_time = start.elapsed();

    // Verify results match
    for i in 0..size {
        assert!((gpu_result[i] - cpu_result[i]).abs() < 1e-5);
    }

    println!("Performance Benchmark (size: {}):", size);
    println!("  GPU time: {:?}", gpu_time);
    println!("  CPU time: {:?}", cpu_time);

    if gpu_time < cpu_time {
        println!(
            "  ✓ GPU is {:.2}x faster",
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
        );
    } else {
        println!(
            "  ⚠ CPU is {:.2}x faster (GPU overhead for small operations)",
            gpu_time.as_secs_f64() / cpu_time.as_secs_f64()
        );
    }
}

/// Test error handling and edge cases
#[test]
fn test_gpu_error_handling() {
    let backend = match VulkanBackend::new() {
        Ok(b) => b,
        Err(_) => return,
    };

    // Test invalid shader name
    let result = backend.get_pipeline("invalid_shader");
    assert!(result.is_err());

    // Test buffer size mismatch
    let mem = backend.allocate(10).unwrap();
    let wrong_size_data = vec![1.0f32; 20];
    let result = backend.copy_to_device(&wrong_size_data, mem.as_ref());
    assert!(result.is_err());

    println!("✓ GPU error handling test passed");
}

/// Run all GPU tests
pub fn run_all_gpu_tests() {
    println!("Running comprehensive Vulkan GPU backend tests...\n");

    test_vulkan_backend_initialization();
    test_gpu_memory_operations();
    test_gpu_elementwise_addition();
    test_gpu_all_elementwise_operations();
    test_gpu_scalar_operations();
    test_gpu_matrix_multiplication();
    test_gpu_activation_functions();
    test_gpu_transpose();
    test_gpu_device_integration();
    benchmark_gpu_performance();
    test_gpu_error_handling();

    println!("\n🎉 All Vulkan GPU backend tests completed successfully!");
}
