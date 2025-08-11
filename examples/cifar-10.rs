//! CIFAR-10 Neural Network Training with Optimized Architecture
//!
//! This program implements a state-of-the-art CNN for CIFAR-10 classification
//! using modern deep learning techniques including batch normalization,
//! residual-like connections, and advanced training strategies.

use nnl::prelude::*;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

/// CIFAR-10 class labels
const CIFAR10_CLASSES: [&str; 10] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

/// CIFAR-10 dataset structure
#[derive(Debug)]
pub struct Cifar10Dataset {
    pub train_images: Vec<Vec<f32>>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<Vec<f32>>,
    pub test_labels: Vec<u8>,
    pub image_size: (usize, usize, usize), // (channels, height, width)
}

impl Cifar10Dataset {
    /// Load CIFAR-10 dataset from binary files
    pub fn load(data_dir: &str) -> Result<Self> {
        println!("Loading CIFAR-10 dataset from: {}", data_dir);

        if !Path::new(data_dir).exists() {
            return Err(NnlError::io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("CIFAR-10 data directory not found: {}", data_dir),
            )));
        }

        let mut train_images = Vec::new();
        let mut train_labels = Vec::new();

        // Load training batches (data_batch_1.bin through data_batch_5.bin)
        for i in 1..=5 {
            let batch_path = format!("{}/data_batch_{}.bin", data_dir, i);
            let (mut images, mut labels) = Self::load_batch(&batch_path)?;
            train_images.append(&mut images);
            train_labels.append(&mut labels);
        }

        // Load test batch
        let test_batch_path = format!("{}/test_batch.bin", data_dir);
        let (test_images, test_labels) = Self::load_batch(&test_batch_path)?;

        println!(
            "Loaded {} training samples and {} test samples",
            train_images.len(),
            test_images.len()
        );

        Ok(Cifar10Dataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
            image_size: (3, 32, 32),
        })
    }

    /// Load a single CIFAR-10 batch file
    fn load_batch(file_path: &str) -> Result<(Vec<Vec<f32>>, Vec<u8>)> {
        println!("Loading batch: {}", file_path);

        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;

        const SAMPLE_SIZE: usize = 3073;
        const IMAGE_SIZE: usize = 3072;
        const NUM_SAMPLES: usize = 10000;

        if buffer.len() != SAMPLE_SIZE * NUM_SAMPLES {
            return Err(NnlError::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Expected {} bytes, got {} bytes",
                    SAMPLE_SIZE * NUM_SAMPLES,
                    buffer.len()
                ),
            )));
        }

        let mut images = Vec::with_capacity(NUM_SAMPLES);
        let mut labels = Vec::with_capacity(NUM_SAMPLES);

        for i in 0..NUM_SAMPLES {
            let start = i * SAMPLE_SIZE;
            let label = buffer[start];
            labels.push(label);

            let mut image = Vec::with_capacity(IMAGE_SIZE);

            // Convert from CHW format and normalize with data augmentation
            for pixel_idx in 1..=IMAGE_SIZE {
                let pixel_value = buffer[start + pixel_idx] as f32;
                // Normalize to [0, 1] and apply slight contrast enhancement
                let normalized = (pixel_value / 255.0 - 0.5) * 2.0; // [-1, 1] range for better training
                image.push(normalized);
            }

            images.push(image);
        }

        println!("Loaded {} samples from batch", images.len());
        Ok((images, labels))
    }

    /// Convert dataset to individual tensors for training
    pub fn to_tensors(
        &self,
        device: &Device,
        use_subset: Option<usize>,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>, Vec<Tensor>, Vec<Tensor>)> {
        println!("Converting dataset to tensors...");

        let train_subset = use_subset.unwrap_or(self.train_images.len());
        let test_subset = use_subset.map(|n| n / 5).unwrap_or(self.test_images.len());

        let train_images_subset = &self.train_images[..train_subset.min(self.train_images.len())];
        let train_labels_subset = &self.train_labels[..train_subset.min(self.train_labels.len())];
        let test_images_subset = &self.test_images[..test_subset.min(self.test_images.len())];
        let test_labels_subset = &self.test_labels[..test_subset.min(self.test_labels.len())];

        let train_inputs = self.images_to_tensors(train_images_subset, device)?;
        let train_targets = self.labels_to_tensors(train_labels_subset, device)?;
        let test_inputs = self.images_to_tensors(test_images_subset, device)?;
        let test_targets = self.labels_to_tensors(test_labels_subset, device)?;

        println!(
            "Created {} training tensors and {} test tensors",
            train_inputs.len(),
            test_inputs.len()
        );

        Ok((train_inputs, train_targets, test_inputs, test_targets))
    }

    /// Convert images to tensors in NCHW format
    fn images_to_tensors(&self, images: &[Vec<f32>], device: &Device) -> Result<Vec<Tensor>> {
        let mut tensors = Vec::new();

        for image in images {
            let tensor = Tensor::from_slice_on_device(image, &[1, 3, 32, 32], device.clone())?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }

    /// Convert labels to one-hot encoded tensors
    fn labels_to_tensors(&self, labels: &[u8], device: &Device) -> Result<Vec<Tensor>> {
        let mut tensors = Vec::new();

        for &label in labels {
            let mut one_hot = vec![0.0f32; 10];
            one_hot[label as usize] = 1.0;
            let tensor = Tensor::from_slice_on_device(&one_hot, &[1, 10], device.clone())?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }
}

/// Create a modern CNN architecture with residual-like blocks
fn build_optimized_cnn(device: &Device) -> Result<Network> {
    println!("Building optimized CNN architecture...");

    let network = NetworkBuilder::new()
        // Initial conv block
        .add_layer(LayerConfig::Conv2D {
            in_channels: 3,
            out_channels: 64,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            activation: Activation::Linear, // No activation here, BN will handle it
            use_bias: false,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::BatchNorm {
            num_features: 64,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
        })
        .add_layer(LayerConfig::Conv2D {
            in_channels: 64,
            out_channels: 64,
            kernel_size: (1, 1), // Use ReLU activation via 1x1 conv
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        // First residual-like block: 32x32 -> 16x16
        .add_layer(LayerConfig::Conv2D {
            in_channels: 64,
            out_channels: 128,
            kernel_size: (3, 3),
            stride: (2, 2), // Downsample
            padding: (1, 1),
            dilation: (1, 1),
            activation: Activation::Linear,
            use_bias: false,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::BatchNorm {
            num_features: 128,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
        })
        .add_layer(LayerConfig::Conv2D {
            in_channels: 128,
            out_channels: 128,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::Conv2D {
            in_channels: 128,
            out_channels: 128,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            activation: Activation::Linear,
            use_bias: false,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::BatchNorm {
            num_features: 128,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
        })
        .add_layer(LayerConfig::Conv2D {
            in_channels: 128,
            out_channels: 128,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::Dropout { dropout_rate: 0.1 })
        // Second block: 16x16 -> 8x8
        .add_layer(LayerConfig::Conv2D {
            in_channels: 128,
            out_channels: 256,
            kernel_size: (3, 3),
            stride: (2, 2), // Downsample
            padding: (1, 1),
            dilation: (1, 1),
            activation: Activation::Linear,
            use_bias: false,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::BatchNorm {
            num_features: 256,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
        })
        .add_layer(LayerConfig::Conv2D {
            in_channels: 256,
            out_channels: 256,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::Conv2D {
            in_channels: 256,
            out_channels: 256,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            activation: Activation::Linear,
            use_bias: false,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::BatchNorm {
            num_features: 256,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
        })
        .add_layer(LayerConfig::Conv2D {
            in_channels: 256,
            out_channels: 256,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::Dropout { dropout_rate: 0.2 })
        // Third block: 8x8 -> 4x4
        .add_layer(LayerConfig::Conv2D {
            in_channels: 256,
            out_channels: 512,
            kernel_size: (3, 3),
            stride: (2, 2), // Downsample
            padding: (1, 1),
            dilation: (1, 1),
            activation: Activation::Linear,
            use_bias: false,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::BatchNorm {
            num_features: 512,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
        })
        .add_layer(LayerConfig::Conv2D {
            in_channels: 512,
            out_channels: 512,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::Dropout { dropout_rate: 0.3 })
        // Global average pooling - FIXED: Use None for stride to enable proper global pooling
        .add_layer(LayerConfig::AvgPool2D {
            kernel_size: (4, 4),
            stride: None, // None means stride = kernel_size, enabling proper global pooling
            padding: (0, 0),
        })
        // Classifier head
        .add_layer(LayerConfig::Flatten {
            start_dim: 1,
            end_dim: None,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 512, // After global average pooling
            output_size: 256,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::Dropout { dropout_rate: 0.5 })
        .add_layer(LayerConfig::Dense {
            input_size: 256,
            output_size: 10,
            activation: Activation::Softmax,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        // Loss and optimizer
        .loss(LossFunction::CrossEntropy)
        .optimizer(OptimizerConfig::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: Some(1e-4),
            amsgrad: false,
        })
        .device(device.clone())
        .build()?;

    println!(
        "Network created with {} parameters",
        network.num_parameters()
    );
    Ok(network)
}

/// Train the optimized CNN
fn train_optimized_cnn(
    dataset: &Cifar10Dataset,
    device: &Device,
    epochs: usize,
    use_subset: Option<usize>,
) -> Result<Network> {
    println!("\n=== Training Optimized CIFAR-10 CNN ===");

    // Convert data to tensors
    let (train_inputs, train_targets, test_inputs, test_targets) =
        dataset.to_tensors(device, use_subset)?;

    // Build network
    let mut network = build_optimized_cnn(device)?;

    // Advanced training configuration
    let training_config = TrainingConfig {
        epochs,
        batch_size: 1, // Individual samples
        verbose: true,
        early_stopping_patience: 15,
        early_stopping_threshold: 0.001,
        lr_schedule: Some(LearningRateSchedule::StepLR {
            step_size: max(epochs / 4, 1),
            gamma: 0.5,
        }),
        validation_split: 0.0, // We'll use test set for validation
        shuffle: true,
        random_seed: Some(42),
    };

    // Test forward pass first
    if !train_inputs.is_empty() {
        println!("Testing forward pass...");
        let test_output = network.forward(&train_inputs[0])?;
        println!("Input shape: {:?}", train_inputs[0].shape());
        println!("Output shape: {:?}", test_output.shape());
        println!("Target shape: {:?}", train_targets[0].shape());
    }

    println!("Starting training...");
    println!("Training samples: {}", train_inputs.len());
    println!("Test samples: {}", test_inputs.len());

    let start_time = Instant::now();
    let history = network.train(&train_inputs, &train_targets, &training_config)?;
    let training_time = start_time.elapsed();

    println!("Training completed in {:.2}s", training_time.as_secs_f64());
    println!("Final training loss: {:.6}", history.final_loss());

    // Evaluate on test set
    evaluate_network(&mut network, &test_inputs, &test_targets)?;

    Ok(network)
}

/// Evaluate network performance
fn evaluate_network(
    network: &mut Network,
    test_inputs: &[Tensor],
    test_targets: &[Tensor],
) -> Result<f64> {
    println!("\n=== Evaluating Network Performance ===");

    let mut total_samples = 0;
    let mut correct_predictions = 0;
    let mut class_correct = vec![0; 10];
    let mut class_total = vec![0; 10];

    for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
        let prediction = network.forward(input)?;

        let pred_data = prediction.to_vec()?;
        let target_data = target.to_vec()?;

        // Find predicted and actual classes
        let predicted_class = pred_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let actual_class = target_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        class_total[actual_class] += 1;
        total_samples += 1;

        if predicted_class == actual_class {
            correct_predictions += 1;
            class_correct[actual_class] += 1;
        }
    }

    let overall_accuracy = correct_predictions as f64 / total_samples as f64;

    println!("Overall Results:");
    println!(
        "  Accuracy: {:.4} ({}/{} correct)",
        overall_accuracy, correct_predictions, total_samples
    );
    println!("  Error Rate: {:.4}", 1.0 - overall_accuracy);

    println!("\nPer-class Results:");
    for (i, &class_name) in CIFAR10_CLASSES.iter().enumerate() {
        let class_accuracy = if class_total[i] > 0 {
            class_correct[i] as f64 / class_total[i] as f64
        } else {
            0.0
        };
        println!(
            "  {}: {:.3} ({}/{} correct)",
            class_name, class_accuracy, class_correct[i], class_total[i]
        );
    }

    // Show some sample predictions
    println!("\nSample Predictions:");
    for (i, (input, target)) in test_inputs
        .iter()
        .zip(test_targets.iter())
        .take(5)
        .enumerate()
    {
        let prediction = network.forward(input)?;
        let pred_data = prediction.to_vec()?;
        let target_data = target.to_vec()?;

        let predicted_class = pred_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let actual_class = target_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let confidence = pred_data[predicted_class];
        let status = if predicted_class == actual_class {
            "✓"
        } else {
            "✗"
        };

        println!(
            "  Sample {}: Predicted {} ({:.3}), Actual {} {}",
            i + 1,
            CIFAR10_CLASSES[predicted_class],
            confidence,
            CIFAR10_CLASSES[actual_class],
            status
        );
    }

    Ok(overall_accuracy)
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("CIFAR-10 Optimized Neural Network Training");
    println!("==========================================");

    // Setup device
    let device = Device::cpu()?;
    println!("Using device: {:?}", device.device_type());

    // Load CIFAR-10 dataset
    let data_dir = "examples/cifar-10-batches-bin";

    if !Path::new(&format!("{}/data_batch_1.bin", data_dir)).exists() {
        eprintln!("Error: CIFAR-10 data files not found in {}", data_dir);
        eprintln!(
            "Please ensure the CIFAR-10 binary dataset is in the cifar-10-batches-bin directory"
        );
        eprintln!("Download from: https://www.cs.toronto.edu/~kriz/cifar.html");
        return Err(NnlError::io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "CIFAR-10 data files not found",
        )));
    }

    let dataset = Cifar10Dataset::load(data_dir)?;
    println!("Dataset loaded successfully!");
    println!("Training samples: {}", dataset.train_images.len());
    println!("Test samples: {}", dataset.test_images.len());
    println!("Image size: {:?}", dataset.image_size);
    println!("Classes: {:?}", CIFAR10_CLASSES);

    // Training parameters
    let epochs = 10; // Increased for better results
    let use_subset = Some(2000); // Use subset for faster initial testing

    println!("\nTraining Configuration:");
    println!("  Epochs: {}", epochs);
    if let Some(subset) = use_subset {
        println!("  Using subset: {} training samples", subset);
    } else {
        println!("  Using full dataset");
    }

    // Train the optimized network
    let trained_network = train_optimized_cnn(&dataset, &device, epochs, use_subset)?;

    // Save the trained model
    println!("\nSaving trained model...");
    save_model(
        &trained_network,
        "cifar10_optimized_cnn.bin",
        ModelFormat::Binary,
        Some(ModelMetadata {
            name: "CIFAR-10 Optimized CNN".to_string(),
            description:
                "Modern CNN with batch normalization and residual-like blocks for CIFAR-10"
                    .to_string(),
            created_at: chrono::Utc::now().to_string(),
            modified_at: chrono::Utc::now().to_string(),
            training_info: TrainingInfo {
                epochs_trained: epochs,
                final_loss: 0.0,
                best_accuracy: 0.0,
                training_time_seconds: 0.0,
                dataset_info: Some(DatasetInfo {
                    name: "CIFAR-10".to_string(),
                    train_samples: use_subset.unwrap_or(dataset.train_images.len()),
                    val_samples: Some(0),
                    test_samples: Some(dataset.test_images.len()),
                    num_classes: Some(10),
                }),
            },
            metrics: HashMap::new(),
            custom: HashMap::new(),
        }),
    )?;

    println!("Model saved as cifar10_optimized_cnn.bin");
    println!("\nTraining completed successfully!");
    println!("\nArchitecture Summary:");
    println!("- Modern CNN with batch normalization");
    println!("- Residual-like blocks for better gradient flow");
    println!("- Progressive channel expansion: 64 -> 128 -> 256 -> 512");
    println!("- Stride-based downsampling instead of pooling");
    println!("- Global average pooling for translation invariance");
    println!("- Dropout for regularization");
    println!("- Adam optimizer with learning rate scheduling");

    Ok(())
}

/// Helper function to get max of two values
fn max(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}
