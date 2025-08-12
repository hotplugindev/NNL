//! Network builder module for fluent API
//!
//! This module provides a builder pattern for constructing neural networks
//! with a fluent, easy-to-use API that allows chaining layer additions and
//! configuration settings.

use crate::device::Device;
use crate::error::{NnlError, Result};
use crate::layers::{Layer, LayerConfig, create_layer};
use crate::losses::LossFunction;
use crate::network::Network;
use crate::optimizers::{OptimizerConfig, create_optimizer};

/// Builder for constructing neural networks
#[derive(Debug)]
pub struct NetworkBuilder {
    /// List of layer configurations
    layers: Vec<LayerConfig>,
    /// Loss function configuration
    loss_function: Option<LossFunction>,
    /// Optimizer configuration
    optimizer: Option<OptimizerConfig>,
    /// Target device for the network
    device: Option<Device>,
    /// Network name
    name: Option<String>,
    /// Network description
    description: Option<String>,
}

impl NetworkBuilder {
    /// Create a new network builder
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            loss_function: None,
            optimizer: None,
            device: None,
            name: None,
            description: None,
        }
    }

    /// Add a layer to the network
    pub fn add_layer(mut self, layer_config: LayerConfig) -> Self {
        self.layers.push(layer_config);
        self
    }

    /// Add multiple layers to the network
    pub fn add_layers(mut self, layer_configs: Vec<LayerConfig>) -> Self {
        self.layers.extend(layer_configs);
        self
    }

    /// Set the loss function
    pub fn loss(mut self, loss_function: LossFunction) -> Self {
        self.loss_function = Some(loss_function);
        self
    }

    /// Set the optimizer
    pub fn optimizer(mut self, optimizer: OptimizerConfig) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    /// Set the target device
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the network name
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the network description
    pub fn description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Build the network
    pub fn build(self) -> Result<Network> {
        // Validate configuration
        if self.layers.is_empty() {
            return Err(NnlError::network("Network must have at least one layer"));
        }

        let loss_function = self
            .loss_function
            .ok_or_else(|| NnlError::network("Loss function must be specified"))?;

        let optimizer_config = self
            .optimizer
            .ok_or_else(|| NnlError::network("Optimizer must be specified"))?;

        // Get or auto-select device
        let device = if let Some(device) = self.device {
            device
        } else {
            Device::auto_select()?
        };

        // Create layers
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        for layer_config in &self.layers {
            let layer = create_layer(layer_config.clone(), device.clone())?;
            layers.push(layer);
        }

        // Validate layer compatibility
        NetworkBuilder::validate_layer_compatibility(&self.layers, &layers)?;

        // Create optimizer
        let optimizer = create_optimizer(optimizer_config.clone())?;

        // Build network
        Network::new_with_configs(
            layers,
            self.layers,
            loss_function,
            optimizer_config,
            optimizer,
            device,
        )
    }

    /// Validate that layers are compatible with each other
    fn validate_layer_compatibility(
        _layer_configs: &[LayerConfig],
        layers: &[Box<dyn Layer>],
    ) -> Result<()> {
        // This is a simplified validation - would need actual shape checking
        // For now, just ensure we have layers
        if layers.is_empty() {
            return Err(NnlError::network("No layers created"));
        }

        Ok(())
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent API extensions for common network architectures
impl NetworkBuilder {
    /// Create a simple feedforward network
    pub fn feedforward(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize) -> Self {
        let mut builder = Self::new();

        // Add input to first hidden layer
        if let Some(&first_hidden) = hidden_sizes.first() {
            builder = builder.add_layer(LayerConfig::dense_relu(input_size, first_hidden));
        }

        // Add hidden layers
        for window in hidden_sizes.windows(2) {
            builder = builder.add_layer(LayerConfig::dense_relu(window[0], window[1]));
        }

        // Add output layer
        if let Some(&last_hidden) = hidden_sizes.last() {
            builder = builder.add_layer(LayerConfig::dense_linear(last_hidden, output_size));
        } else {
            // No hidden layers, direct input to output
            builder = builder.add_layer(LayerConfig::dense_linear(input_size, output_size));
        }

        builder
    }

    /// Create a simple CNN for image classification
    pub fn cnn_classifier(input_channels: usize, num_classes: usize, image_size: usize) -> Self {
        let mut builder = Self::new();

        // Convolutional layers
        builder = builder
            .add_layer(LayerConfig::conv2d_3x3(input_channels, 32))
            .add_layer(LayerConfig::max_pool2d())
            .add_layer(LayerConfig::conv2d_3x3(32, 64))
            .add_layer(LayerConfig::max_pool2d())
            .add_layer(LayerConfig::conv2d_3x3(64, 128))
            .add_layer(LayerConfig::max_pool2d());

        // Calculate flattened size (simplified calculation)
        let pooled_size = image_size / 8; // 3 max pooling layers
        let flattened_size = 128 * pooled_size * pooled_size;

        // Fully connected layers
        builder = builder
            .add_layer(LayerConfig::flatten())
            .add_layer(LayerConfig::dense_relu(flattened_size, 512))
            .add_layer(LayerConfig::dropout(0.5))
            .add_layer(LayerConfig::dense_linear(512, num_classes));

        builder
    }

    /// Create a binary classifier
    pub fn binary_classifier(input_size: usize, hidden_sizes: Vec<usize>) -> Self {
        let mut builder = Self::feedforward(input_size, hidden_sizes, 1);

        // Set appropriate loss function for binary classification
        builder = builder.loss(LossFunction::BinaryCrossEntropy);

        // Use a good optimizer for classification
        builder = builder.optimizer(OptimizerConfig::adam(0.001));

        builder
    }

    /// Create a multi-class classifier
    pub fn multiclass_classifier(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        num_classes: usize,
    ) -> Self {
        let mut builder = Self::feedforward(input_size, hidden_sizes, num_classes);

        // Set appropriate loss function for multi-class classification
        builder = builder.loss(LossFunction::CrossEntropy);

        // Use a good optimizer for classification
        builder = builder.optimizer(OptimizerConfig::adam(0.001));

        builder
    }

    /// Create a regression network
    pub fn regressor(input_size: usize, hidden_sizes: Vec<usize>, output_size: usize) -> Self {
        let mut builder = Self::feedforward(input_size, hidden_sizes, output_size);

        // Set appropriate loss function for regression
        builder = builder.loss(LossFunction::MeanSquaredError);

        // Use a good optimizer for regression
        builder = builder.optimizer(OptimizerConfig::adam(0.001));

        builder
    }

    /// Create an autoencoder network
    pub fn autoencoder(input_size: usize, encoding_size: usize) -> Self {
        let mut builder = Self::new();

        // Encoder
        let mut current_size = input_size;
        while current_size > encoding_size {
            let next_size = (current_size + encoding_size) / 2;
            builder = builder.add_layer(LayerConfig::dense_relu(current_size, next_size));
            current_size = next_size;
        }

        // Decoder
        while current_size < input_size {
            let next_size = if current_size * 2 > input_size {
                input_size
            } else {
                current_size * 2
            };
            builder = builder.add_layer(LayerConfig::dense_relu(current_size, next_size));
            current_size = next_size;
        }

        // Set appropriate loss and optimizer for autoencoder
        builder = builder
            .loss(LossFunction::MeanSquaredError)
            .optimizer(OptimizerConfig::adam(0.001));

        builder
    }
}

/// Preset network configurations
pub mod presets {
    use super::*;
    use crate::activations::Activation;
    use crate::layers::WeightInit;

    /// Create a LeNet-5 style CNN for MNIST
    pub fn lenet5() -> NetworkBuilder {
        NetworkBuilder::new()
            .add_layer(LayerConfig::Conv2D {
                in_channels: 1,
                out_channels: 6,
                kernel_size: (5, 5),
                stride: (1, 1),
                padding: (0, 0),
                dilation: (1, 1),
                activation: Activation::Tanh,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .add_layer(LayerConfig::MaxPool2D {
                kernel_size: (2, 2),
                stride: Some((2, 2)),
                padding: (0, 0),
            })
            .add_layer(LayerConfig::Conv2D {
                in_channels: 6,
                out_channels: 16,
                kernel_size: (5, 5),
                stride: (1, 1),
                padding: (0, 0),
                dilation: (1, 1),
                activation: Activation::Tanh,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .add_layer(LayerConfig::MaxPool2D {
                kernel_size: (2, 2),
                stride: Some((2, 2)),
                padding: (0, 0),
            })
            .add_layer(LayerConfig::Flatten {
                start_dim: 1,
                end_dim: None,
            })
            .add_layer(LayerConfig::Dense {
                input_size: 16 * 5 * 5,
                output_size: 120,
                activation: Activation::Tanh,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .add_layer(LayerConfig::Dense {
                input_size: 120,
                output_size: 84,
                activation: Activation::Tanh,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .add_layer(LayerConfig::Dense {
                input_size: 84,
                output_size: 10,
                activation: Activation::Linear,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .loss(LossFunction::CrossEntropy)
            .optimizer(OptimizerConfig::sgd(0.01))
    }

    /// Create a simple MLP for XOR problem
    pub fn xor_network() -> NetworkBuilder {
        NetworkBuilder::new()
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
            .loss(LossFunction::BinaryCrossEntropy)
            .optimizer(OptimizerConfig::adam(0.01))
    }

    /// Create a simple MNIST classifier
    pub fn mnist_classifier() -> NetworkBuilder {
        NetworkBuilder::new()
            .add_layer(LayerConfig::Dense {
                input_size: 784,
                output_size: 128,
                activation: Activation::ReLU,
                use_bias: true,
                weight_init: WeightInit::HeNormal,
            })
            .add_layer(LayerConfig::Dropout { dropout_rate: 0.2 })
            .add_layer(LayerConfig::Dense {
                input_size: 128,
                output_size: 64,
                activation: Activation::ReLU,
                use_bias: true,
                weight_init: WeightInit::HeNormal,
            })
            .add_layer(LayerConfig::Dropout { dropout_rate: 0.2 })
            .add_layer(LayerConfig::Dense {
                input_size: 64,
                output_size: 10,
                activation: Activation::Softmax,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .loss(LossFunction::CrossEntropy)
            .optimizer(OptimizerConfig::adam(0.001))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let network = NetworkBuilder::new()
            .add_layer(LayerConfig::dense_relu(2, 4))
            .add_layer(LayerConfig::dense_sigmoid(4, 1))
            .loss(LossFunction::MeanSquaredError)
            .optimizer(OptimizerConfig::sgd(0.1))
            .build();

        assert!(network.is_ok());
        let network = network.unwrap();
        assert_eq!(network.num_layers(), 2);
    }

    #[test]
    fn test_builder_missing_loss() {
        let result = NetworkBuilder::new()
            .add_layer(LayerConfig::dense_relu(2, 1))
            .optimizer(OptimizerConfig::sgd(0.1))
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_optimizer() {
        let result = NetworkBuilder::new()
            .add_layer(LayerConfig::dense_relu(2, 1))
            .loss(LossFunction::MeanSquaredError)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_builder_no_layers() {
        let result = NetworkBuilder::new()
            .loss(LossFunction::MeanSquaredError)
            .optimizer(OptimizerConfig::sgd(0.1))
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_feedforward_builder() {
        let network = NetworkBuilder::feedforward(784, vec![128, 64], 10)
            .loss(LossFunction::CrossEntropy)
            .optimizer(OptimizerConfig::adam(0.001))
            .build();

        assert!(network.is_ok());
        let network = network.unwrap();
        assert_eq!(network.num_layers(), 3); // 2 hidden + 1 output
    }

    #[test]
    fn test_binary_classifier_builder() {
        let network = NetworkBuilder::binary_classifier(10, vec![5, 3]).build();

        assert!(network.is_ok());
        let network = network.unwrap();
        assert_eq!(network.num_layers(), 3);
    }

    #[test]
    fn test_multiclass_classifier_builder() {
        let network = NetworkBuilder::multiclass_classifier(784, vec![128], 10).build();

        assert!(network.is_ok());
        let network = network.unwrap();
        assert_eq!(network.num_layers(), 2);
    }

    #[test]
    fn test_regressor_builder() {
        let network = NetworkBuilder::regressor(5, vec![10, 5], 1).build();

        assert!(network.is_ok());
        let network = network.unwrap();
        assert_eq!(network.num_layers(), 3);
    }

    #[test]
    fn test_preset_xor_network() {
        let network = presets::xor_network().build();

        assert!(network.is_ok());
        let network = network.unwrap();
        assert_eq!(network.num_layers(), 2);
    }

    #[test]
    fn test_preset_mnist_classifier() {
        let network = presets::mnist_classifier().build();

        assert!(network.is_ok());
        let network = network.unwrap();
        assert_eq!(network.num_layers(), 5); // 3 dense + 2 dropout
    }

    #[test]
    fn test_builder_fluent_api() {
        let network = NetworkBuilder::new()
            .name("Test Network")
            .description("A test neural network")
            .add_layer(LayerConfig::dense_relu(10, 5))
            .add_layer(LayerConfig::dense_linear(5, 1))
            .loss(LossFunction::MeanSquaredError)
            .optimizer(OptimizerConfig::adam(0.001))
            .build();

        assert!(network.is_ok());
    }

    #[test]
    fn test_autoencoder_builder() {
        let network = NetworkBuilder::autoencoder(784, 32).build();

        assert!(network.is_ok());
        let network = network.unwrap();
        assert!(network.num_layers() > 2); // Should have encoder and decoder layers
    }
}
