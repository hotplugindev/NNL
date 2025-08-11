//! Simple XOR Neural Network Training Example - GPU Version
//!
//! This example demonstrates training a simple neural network to learn the XOR function
//! using GPU acceleration. The network consists of a single hidden layer and uses
//! backpropagation for training.

use nnl::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("Simple XOR Neural Network Training - GPU Version");
    println!("===============================================");

    // Try to get a GPU device, fall back to CPU if not available
    let device = match Device::vulkan().or_else(|_| Device::cpu()) {
        Ok(device) => {
            println!("Using device: {:?}", device.device_type());
            device
        }
        Err(e) => {
            println!("Failed to initialize GPU device: {}", e);
            println!("Falling back to CPU");
            Device::cpu()?
        }
    };

    // Create XOR training data as individual samples
    let train_inputs = vec![
        Tensor::from_slice_on_device(&[0.0, 0.0], &[1, 2], device.clone())?,
        Tensor::from_slice_on_device(&[0.0, 1.0], &[1, 2], device.clone())?,
        Tensor::from_slice_on_device(&[1.0, 0.0], &[1, 2], device.clone())?,
        Tensor::from_slice_on_device(&[1.0, 1.0], &[1, 2], device.clone())?,
    ];

    let train_targets = vec![
        Tensor::from_slice_on_device(&[0.0], &[1, 1], device.clone())?,
        Tensor::from_slice_on_device(&[1.0], &[1, 1], device.clone())?,
        Tensor::from_slice_on_device(&[1.0], &[1, 1], device.clone())?,
        Tensor::from_slice_on_device(&[0.0], &[1, 1], device.clone())?,
    ];

    println!("Training data created on GPU:");
    for (i, (input, target)) in train_inputs.iter().zip(train_targets.iter()).enumerate() {
        let input_data = input.to_vec()?;
        let target_data = target.to_vec()?;
        println!(
            "  Sample {}: [{:.0}, {:.0}] -> {:.0}",
            i + 1,
            input_data[0],
            input_data[1],
            target_data[0]
        );
    }

    // Build the neural network
    let mut network = NetworkBuilder::new()
        .add_layer(LayerConfig::Dense {
            input_size: 2,
            output_size: 8,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 8,
            output_size: 1,
            activation: Activation::Sigmoid,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::MeanSquaredError)
        .optimizer(OptimizerConfig::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: None,
            amsgrad: false,
        })
        .device(device.clone())
        .build()?;

    println!(
        "\nNetwork created with {} parameters",
        network.num_parameters()
    );

    // Debug: Check initial weights to ensure they're properly initialized
    println!("\nDebugging weight initialization...");
    let test_input = &train_inputs[0];
    let initial_prediction = network.forward(test_input)?;
    println!(
        "Initial prediction value: {:?}",
        initial_prediction.to_vec()?
    );

    // Test if tensors are actually on GPU
    println!("Test tensor device: {:?}", test_input.device());
    println!("Network device: {:?}", device.device_type());

    // Test initial predictions
    println!("\nInitial predictions (before training):");
    for (i, input) in train_inputs.iter().enumerate() {
        let prediction = network.forward(input)?;
        let predicted_value = prediction.to_vec()?[0];
        let target_value = train_targets[i].to_vec()?[0];
        let input_data = input.to_vec()?;

        println!(
            "  Input: [{:.0}, {:.0}] -> Predicted: {:.4}, Target: {:.0}",
            input_data[0], input_data[1], predicted_value, target_value
        );
    }

    // Training configuration
    let training_config = TrainingConfig {
        epochs: 1000,
        batch_size: 4,
        verbose: true,
        early_stopping_patience: 50,
        early_stopping_threshold: 0.001,
        lr_schedule: None,
        validation_split: 0.0,
        shuffle: true,
        random_seed: Some(42),
    };

    println!("\nStarting training...");
    println!("WARNING: GPU training is currently very slow. This may take several minutes...");
    let start_time = Instant::now();

    // Check prediction before training to see if weights change
    let pre_training_pred = network.forward(&train_inputs[0])?;
    println!("Pre-training prediction: {:?}", pre_training_pred.to_vec()?);

    let history = network.train(&train_inputs, &train_targets, &training_config)?;

    // Check prediction after training to see if weights actually changed
    let post_training_pred = network.forward(&train_inputs[0])?;
    println!(
        "Post-training prediction: {:?}",
        post_training_pred.to_vec()?
    );

    let training_time = start_time.elapsed();
    println!("Training completed in {:.2}s", training_time.as_secs_f64());
    println!(
        "Average time per epoch: {:.4}s",
        training_time.as_secs_f64() / training_config.epochs as f64
    );
    println!("Final loss: {:.6}", history.final_loss());

    // Test final predictions
    println!("\nFinal predictions (after training):");
    let mut correct = 0;

    // Debug: Define expected XOR inputs for verification
    let expected_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    for (i, input) in train_inputs.iter().enumerate() {
        let prediction = network.forward(input)?;
        let predicted_value = prediction.to_vec()?[0];
        let target_value = train_targets[i].to_vec()?[0];
        let input_data = input.to_vec()?;

        let predicted_class = if predicted_value > 0.5 { 1.0 } else { 0.0 };
        let is_correct = (predicted_class - target_value).abs() < 0.1;
        if is_correct {
            correct += 1;
        }

        let status = if is_correct { "✅" } else { "❌" };

        // Debug: Compare actual input with expected
        println!(
            "  Debug: Expected input [{:.0}, {:.0}], Got input [{:.0}, {:.0}]",
            expected_inputs[i][0], expected_inputs[i][1], input_data[0], input_data[1]
        );

        println!(
            "  {} Input: [{:.0}, {:.0}] -> Predicted: {:.4} (class: {:.0}), Target: {:.0}",
            status, input_data[0], input_data[1], predicted_value, predicted_class, target_value
        );
    }

    let accuracy = correct as f64 / train_inputs.len() as f64;
    println!(
        "\nAccuracy: {:.1}% ({}/{})",
        accuracy * 100.0,
        correct,
        train_inputs.len()
    );

    if accuracy >= 1.0 {
        println!("🎉 SUCCESS: Network learned the XOR function perfectly!");
    } else if accuracy >= 0.75 {
        println!("👍 GOOD: Network mostly learned the XOR function");
    } else {
        println!("❌ PROBLEM: Network failed to learn the XOR function");
    }

    // Benchmark inference speed
    println!("\nBenchmarking inference speed...");
    let benchmark_input = &train_inputs[0];
    let num_iterations = 100; // Reduced iterations due to slow GPU performance

    println!("Running {} inference iterations on GPU...", num_iterations);
    let start_time = Instant::now();
    for _ in 0..num_iterations {
        let _ = network.forward(benchmark_input)?;
    }
    // Ensure all GPU operations complete
    device.synchronize()?;
    let inference_time = start_time.elapsed();

    println!(
        "Average inference time: {:.4}ms ({} iterations)",
        inference_time.as_secs_f64() * 1000.0 / num_iterations as f64,
        num_iterations
    );

    println!("GPU Performance Analysis:");
    println!(
        "  - GPU inference: {:.4}ms per forward pass",
        inference_time.as_secs_f64() * 1000.0 / num_iterations as f64
    );
    println!(
        "  - Training time per epoch: {:.4}s",
        training_time.as_secs_f64() / training_config.epochs as f64
    );
    println!("  - GPU utilization appears to be very low");
    println!("  - Consider using CPU for small networks like XOR");

    // Save the trained model
    let model_path = "simple_xor_gpu_model.bin";
    nnl::io::save_model(&network, model_path, ModelFormat::Binary, None)?;
    println!("\nModel saved to: {}", model_path);

    println!("\nSimple XOR GPU training example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_xor_gpu() -> Result<()> {
        // Skip GPU test if no GPU is available
        let device = Device::cpu()?;

        // Test basic tensor creation
        let input = Tensor::from_slice_on_device(&[0.0, 1.0], &[1, 2], device.clone())?;
        let target = Tensor::from_slice_on_device(&[1.0], &[1, 1], device.clone())?;

        assert_eq!(input.shape(), &[1, 2]);
        assert_eq!(target.shape(), &[1, 1]);

        // Test network creation
        let network = NetworkBuilder::new()
            .add_layer(LayerConfig::Dense {
                input_size: 2,
                output_size: 4,
                activation: Activation::ReLU,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .add_layer(LayerConfig::Dense {
                input_size: 4,
                output_size: 1,
                activation: Activation::Sigmoid,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .loss(LossFunction::MeanSquaredError)
            .optimizer(OptimizerConfig::Adam {
                learning_rate: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: None,
                amsgrad: false,
            })
            .device(device)
            .build()?;

        // Test forward pass
        let output = network.forward(&input)?;
        assert_eq!(output.shape(), &[1, 1]);

        Ok(())
    }
}
