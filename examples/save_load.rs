use nnl::prelude::*;

fn main() -> Result<()> {
    println!("=== Neural Network Save/Load Test ===\n");

    // Select the device
    let device = Device::cpu()?;
    println!("Using device: {:?}", device.device_type());

    let model_path = "xor_model_test.bin";

    // Create XOR training data
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

    // Create test inputs for evaluation
    let test_input_00 = Tensor::from_slice_on_device(&[0.0, 0.0], &[1, 2], device.clone())?;
    let test_input_01 = Tensor::from_slice_on_device(&[0.0, 1.0], &[1, 2], device.clone())?;
    let test_input_10 = Tensor::from_slice_on_device(&[1.0, 0.0], &[1, 2], device.clone())?;
    let test_input_11 = Tensor::from_slice_on_device(&[1.0, 1.0], &[1, 2], device.clone())?;

    // Create and train a neural network
    println!("Creating and training neural network...");
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
            output_size: 16,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 16,
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

    // Configure training
    let training_config = TrainingConfig {
        epochs: 500,
        batch_size: 4,
        verbose: false,
        ..Default::default()
    };

    // Train the network
    println!("Training for {} epochs...", training_config.epochs);
    network.train(&train_inputs, &train_targets, &training_config)?;

    // Test predictions before saving
    println!("\n--- Predictions BEFORE saving ---");
    let pred_00_before = network.forward(&test_input_00)?.to_vec()?[0];
    let pred_01_before = network.forward(&test_input_01)?.to_vec()?[0];
    let pred_10_before = network.forward(&test_input_10)?.to_vec()?[0];
    let pred_11_before = network.forward(&test_input_11)?.to_vec()?[0];

    println!("XOR(0,0) = {:.4} (expected: 0.0)", pred_00_before);
    println!("XOR(0,1) = {:.4} (expected: 1.0)", pred_01_before);
    println!("XOR(1,0) = {:.4} (expected: 1.0)", pred_10_before);
    println!("XOR(1,1) = {:.4} (expected: 0.0)", pred_11_before);

    // Calculate accuracy before saving
    let correct_before = (if pred_00_before < 0.5 { 1 } else { 0 })
        + (if pred_01_before > 0.5 { 1 } else { 0 })
        + (if pred_10_before > 0.5 { 1 } else { 0 })
        + (if pred_11_before < 0.5 { 1 } else { 0 });

    println!(
        "Accuracy before saving: {}/4 ({:.1}%)",
        correct_before,
        (correct_before as f32 / 4.0) * 100.0
    );

    // Save the model
    println!("\nSaving model to '{}'...", model_path);
    nnl::io::save_model(&network, model_path, ModelFormat::Binary, None)?;
    println!("Model saved successfully!");

    // Load the model
    println!("\nLoading model from '{}'...", model_path);
    let mut loaded_network = nnl::io::load_network(model_path, ModelFormat::Binary)?;
    println!("Model loaded successfully!");

    // Test predictions after loading
    println!("\n--- Predictions AFTER loading ---");
    let pred_00_after = loaded_network.forward(&test_input_00)?.to_vec()?[0];
    let pred_01_after = loaded_network.forward(&test_input_01)?.to_vec()?[0];
    let pred_10_after = loaded_network.forward(&test_input_10)?.to_vec()?[0];
    let pred_11_after = loaded_network.forward(&test_input_11)?.to_vec()?[0];

    println!("XOR(0,0) = {:.4} (expected: 0.0)", pred_00_after);
    println!("XOR(0,1) = {:.4} (expected: 1.0)", pred_01_after);
    println!("XOR(1,0) = {:.4} (expected: 1.0)", pred_10_after);
    println!("XOR(1,1) = {:.4} (expected: 0.0)", pred_11_after);

    // Calculate accuracy after loading
    let correct_after = (if pred_00_after < 0.5 { 1 } else { 0 })
        + (if pred_01_after > 0.5 { 1 } else { 0 })
        + (if pred_10_after > 0.5 { 1 } else { 0 })
        + (if pred_11_after < 0.5 { 1 } else { 0 });

    println!(
        "Accuracy after loading: {}/4 ({:.1}%)",
        correct_after,
        (correct_after as f32 / 4.0) * 100.0
    );

    // Compare predictions to verify state preservation
    println!("\n--- State Preservation Verification ---");
    let diff_00 = (pred_00_before - pred_00_after).abs();
    let diff_01 = (pred_01_before - pred_01_after).abs();
    let diff_10 = (pred_10_before - pred_10_after).abs();
    let diff_11 = (pred_11_before - pred_11_after).abs();

    println!("Prediction differences (should be near 0.0):");
    println!("XOR(0,0): {:.8}", diff_00);
    println!("XOR(0,1): {:.8}", diff_01);
    println!("XOR(1,0): {:.8}", diff_10);
    println!("XOR(1,1): {:.8}", diff_11);

    let max_diff = diff_00.max(diff_01).max(diff_10).max(diff_11);
    println!("Maximum difference: {:.8}", max_diff);

    // Determine if the test passed
    let tolerance = 1e-6;
    if max_diff < tolerance {
        println!("\n✅ SUCCESS: Model state was preserved correctly!");
        println!(
            "   All predictions match within tolerance ({:.0e})",
            tolerance
        );
    } else {
        println!("\n❌ FAILURE: Model state was NOT preserved!");
        println!(
            "   Maximum difference ({:.8}) exceeds tolerance ({:.0e})",
            max_diff, tolerance
        );
        return Err(nnl::error::NnlError::invalid_input(
            "Model state not preserved during save/load",
        ));
    }

    // Test continued training on loaded model
    println!("\n--- Testing Continued Training ---");
    println!("Training loaded model for 100 more epochs...");

    let additional_training_config = TrainingConfig {
        epochs: 100,
        batch_size: 4,
        verbose: false,
        ..Default::default()
    };

    loaded_network.train(&train_inputs, &train_targets, &additional_training_config)?;

    // Test final predictions
    let pred_00_final = loaded_network.forward(&test_input_00)?.to_vec()?[0];
    let pred_01_final = loaded_network.forward(&test_input_01)?.to_vec()?[0];
    let pred_10_final = loaded_network.forward(&test_input_10)?.to_vec()?[0];
    let pred_11_final = loaded_network.forward(&test_input_11)?.to_vec()?[0];

    println!("Final predictions after additional training:");
    println!("XOR(0,0) = {:.4} (expected: 0.0)", pred_00_final);
    println!("XOR(0,1) = {:.4} (expected: 1.0)", pred_01_final);
    println!("XOR(1,0) = {:.4} (expected: 1.0)", pred_10_final);
    println!("XOR(1,1) = {:.4} (expected: 0.0)", pred_11_final);

    let correct_final = (if pred_00_final < 0.5 { 1 } else { 0 })
        + (if pred_01_final > 0.5 { 1 } else { 0 })
        + (if pred_10_final > 0.5 { 1 } else { 0 })
        + (if pred_11_final < 0.5 { 1 } else { 0 });

    println!(
        "Final accuracy: {}/4 ({:.1}%)",
        correct_final,
        (correct_final as f32 / 4.0) * 100.0
    );

    // Clean up
    if std::path::Path::new(model_path).exists() {
        std::fs::remove_file(model_path)?;
        println!("\nCleaned up test file: {}", model_path);
    }

    println!("\n🎉 Save/Load test completed successfully!");

    Ok(())
}
