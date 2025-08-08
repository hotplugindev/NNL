//! Simple XOR Neural Network Training Example - CPU Version
//!
//! This example demonstrates training a simple neural network to learn the XOR function
//! using CPU computation. The network consists of a single hidden layer and uses
//! backpropagation for training.

use nnl::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("Simple XOR Neural Network Training - CPU Version");
    println!("===============================================");

    // Use CPU device
    let device = Device::cpu()?;
    println!("Using device: {:?}", device.device_type());

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

    println!("Training data created:");
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
    let start_time = Instant::now();

    let history = network.train(&train_inputs, &train_targets, &training_config)?;

    let training_time = start_time.elapsed();
    println!("Training completed in {:.2}s", training_time.as_secs_f64());
    println!("Final loss: {:.6}", history.final_loss());

    // Test final predictions
    println!("\nFinal predictions (after training):");
    let mut correct = 0;
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
    let num_iterations = 1000;

    let start_time = Instant::now();
    for _ in 0..num_iterations {
        let _ = network.forward(benchmark_input)?;
    }
    let inference_time = start_time.elapsed();

    println!(
        "Average inference time: {:.4}ms ({} iterations)",
        inference_time.as_secs_f64() * 1000.0 / num_iterations as f64,
        num_iterations
    );

    // Save the trained model
    let model_path = "simple_xor_cpu_model.bin";
    nnl::io::save_model(&network, model_path, ModelFormat::Binary, None)?;
    println!("\nModel saved to: {}", model_path);

    println!("\nSimple XOR CPU training example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_xor_cpu() -> Result<()> {
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
