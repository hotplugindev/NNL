//! I/O module for model serialization and persistence
//!
//! This module provides functionality for saving and loading neural network
//! models to and from disk in various formats, supporting both binary and
//! text-based serialization with compression options.

use crate::error::{NnlError, Result};
use crate::network::Network;
use crate::tensor::SerializableTensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Supported model file formats
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// Binary format using bincode (fast, compact)
    Binary,
    /// JSON format (human-readable, larger)
    Json,
    /// MessagePack format (compact, fast)
    MessagePack,
}

/// Serializable model structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableModel {
    /// Model architecture information
    pub architecture: ModelArchitecture,
    /// Model parameters (weights and biases)
    pub parameters: Vec<SerializableTensor>,
    /// Optimizer state
    pub optimizer_state: HashMap<String, SerializableTensor>,
    /// Training metadata
    pub metadata: ModelMetadata,
    /// Format version for compatibility
    pub version: String,
}

/// Model architecture information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// Layer configurations
    pub layers: Vec<crate::layers::LayerConfig>,
    /// Loss function type
    pub loss_function: crate::losses::LossFunction,
    /// Optimizer configuration
    pub optimizer_config: crate::optimizers::OptimizerConfig,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Device type
    pub device_type: String,
}

/// Individual layer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    /// Layer type (Dense, Conv2D, etc.)
    pub layer_type: String,
    /// Layer configuration parameters
    pub config: HashMap<String, serde_json::Value>,
    /// Number of parameters in this layer
    pub num_parameters: usize,
    /// Input shape for this layer
    pub input_shape: Vec<usize>,
    /// Output shape for this layer
    pub output_shape: Vec<usize>,
}

/// Model training and creation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name or identifier
    pub name: String,
    /// Model description
    pub description: String,
    /// Creation timestamp
    pub created_at: String,
    /// Last modified timestamp
    pub modified_at: String,
    /// Training history summary
    pub training_info: TrainingInfo,
    /// Model performance metrics
    pub metrics: HashMap<String, f32>,
    /// Additional custom metadata
    pub custom: HashMap<String, serde_json::Value>,
}

/// Training information summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingInfo {
    /// Number of epochs trained
    pub epochs_trained: usize,
    /// Final training loss
    pub final_loss: f32,
    /// Best validation accuracy achieved
    pub best_accuracy: f32,
    /// Total training time in seconds
    pub training_time_seconds: f32,
    /// Dataset information
    pub dataset_info: Option<DatasetInfo>,
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset name
    pub name: String,
    /// Number of training samples
    pub train_samples: usize,
    /// Number of validation samples
    pub val_samples: Option<usize>,
    /// Number of test samples
    pub test_samples: Option<usize>,
    /// Number of classes (for classification)
    pub num_classes: Option<usize>,
}

/// Save a model to disk
pub fn save_model<P: AsRef<Path>>(
    model: &Network,
    path: P,
    format: ModelFormat,
    metadata: Option<ModelMetadata>,
) -> Result<()> {
    let serializable = serialize_model(model, metadata)?;

    match format {
        ModelFormat::Binary => save_binary(&serializable, path),
        ModelFormat::Json => save_json(&serializable, path),
        ModelFormat::MessagePack => save_messagepack(&serializable, path),
    }
}

/// Load a model from disk
pub fn load_model<P: AsRef<Path>>(path: P, format: ModelFormat) -> Result<SerializableModel> {
    match format {
        ModelFormat::Binary => load_binary(path),
        ModelFormat::Json => load_json(path),
        ModelFormat::MessagePack => load_messagepack(path),
    }
}

/// Auto-detect format from file extension and load
pub fn load_model_auto<P: AsRef<Path>>(path: P) -> Result<SerializableModel> {
    let path = path.as_ref();
    let format = detect_format_from_extension(path)?;
    load_model(path, format)
}

/// Load a model and convert it back to a Network
pub fn load_network<P: AsRef<Path>>(path: P, format: ModelFormat) -> Result<Network> {
    let serializable = load_model(path, format)?;
    deserialize_model(serializable)
}

/// Auto-detect format and load model as Network
pub fn load_network_auto<P: AsRef<Path>>(path: P) -> Result<Network> {
    let serializable = load_model_auto(path)?;
    deserialize_model(serializable)
}

/// Save model in binary format
fn save_binary<P: AsRef<Path>>(model: &SerializableModel, path: P) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, model)?;
    Ok(())
}

/// Load model from binary format
fn load_binary<P: AsRef<Path>>(path: P) -> Result<SerializableModel> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let model = bincode::deserialize_from(reader)?;
    Ok(model)
}

/// Save model in JSON format
fn save_json<P: AsRef<Path>>(model: &SerializableModel, path: P) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, model)?;
    Ok(())
}

/// Load model from JSON format
fn load_json<P: AsRef<Path>>(path: P) -> Result<SerializableModel> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let model = serde_json::from_reader(reader)?;
    Ok(model)
}

/// Save model in MessagePack format
fn save_messagepack<P: AsRef<Path>>(model: &SerializableModel, path: P) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    rmp_serde::encode::write(&mut writer, model)
        .map_err(|e| NnlError::io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
    Ok(())
}

/// Load model from MessagePack format
fn load_messagepack<P: AsRef<Path>>(path: P) -> Result<SerializableModel> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let model = rmp_serde::decode::from_read(reader)
        .map_err(|e| NnlError::io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
    Ok(model)
}

/// Detect format from file extension
fn detect_format_from_extension<P: AsRef<Path>>(path: P) -> Result<ModelFormat> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| {
            NnlError::io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "No file extension found",
            ))
        })?;

    match extension.to_lowercase().as_str() {
        "bin" | "model" => Ok(ModelFormat::Binary),
        "json" => Ok(ModelFormat::Json),
        "msgpack" | "mp" => Ok(ModelFormat::MessagePack),
        _ => Err(NnlError::io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Unsupported file extension: {}", extension),
        ))),
    }
}

/// Convert Network to SerializableModel
fn serialize_model(
    network: &Network,
    metadata: Option<ModelMetadata>,
) -> Result<SerializableModel> {
    // Extract parameters
    let parameters = extract_parameters(network)?;

    // Extract optimizer state
    let optimizer_state = extract_optimizer_state(network)?;

    // Build architecture description
    let architecture = build_architecture_info(network)?;

    // Use provided metadata or create default
    let metadata = metadata.unwrap_or_else(|| create_default_metadata(network));

    Ok(SerializableModel {
        architecture,
        parameters,
        optimizer_state,
        metadata,
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Extract parameters from network
fn extract_parameters(network: &Network) -> Result<Vec<SerializableTensor>> {
    Ok(network.get_parameters())
}

/// Extract optimizer state from network
fn extract_optimizer_state(network: &Network) -> Result<HashMap<String, SerializableTensor>> {
    Ok(network.get_optimizer_state())
}

/// Convert SerializableModel back to Network
fn deserialize_model(model: SerializableModel) -> Result<Network> {
    use crate::network::NetworkBuilder;

    // Build network from architecture
    let mut builder = NetworkBuilder::new();

    // Add layers from architecture
    if !model.architecture.layers.is_empty() {
        for layer_config in &model.architecture.layers {
            builder = builder.add_layer(layer_config.clone());
        }
    } else {
        return Err(crate::error::NnlError::invalid_input(
            "No layers found in saved model",
        ));
    }

    // Set loss function and optimizer
    builder = builder
        .loss(model.architecture.loss_function.clone())
        .optimizer(model.architecture.optimizer_config.clone());

    // Build the network
    let mut network = builder.build()?;

    // Restore parameters if available
    if !model.parameters.is_empty() {
        network.set_parameters(model.parameters)?;
    }

    // Restore optimizer state if available
    if !model.optimizer_state.is_empty() {
        network.set_optimizer_state(model.optimizer_state)?;
    }

    Ok(network)
}

/// Build architecture information from network
fn build_architecture_info(network: &Network) -> Result<ModelArchitecture> {
    let layers = network.get_layer_configs().to_vec();
    let loss_function = network.get_loss_function().clone();
    let optimizer_config = network.get_optimizer_config().clone();

    // Determine input and output shapes from first and last layers
    let input_shape = if let Some(first_layer) = layers.first() {
        match first_layer {
            crate::layers::LayerConfig::Dense { input_size, .. } => vec![*input_size],
            crate::layers::LayerConfig::Conv2D { in_channels, .. } => vec![*in_channels],
            _ => vec![],
        }
    } else {
        vec![]
    };

    let output_shape = if let Some(last_layer) = layers.last() {
        match last_layer {
            crate::layers::LayerConfig::Dense { output_size, .. } => vec![*output_size],
            crate::layers::LayerConfig::Conv2D { out_channels, .. } => vec![*out_channels],
            _ => vec![],
        }
    } else {
        vec![]
    };

    let device_type = match network.get_device().device_type() {
        crate::device::DeviceType::Vulkan => "Vulkan".to_string(),
        crate::device::DeviceType::Cpu => "CPU".to_string(),
    };

    Ok(ModelArchitecture {
        layers,
        loss_function,
        optimizer_config,
        input_shape,
        output_shape,
        device_type,
    })
}

/// Create default metadata
fn create_default_metadata(network: &Network) -> ModelMetadata {
    let now = chrono::Utc::now().to_rfc3339();

    ModelMetadata {
        name: "Unnamed Model".to_string(),
        description: "Neural network model".to_string(),
        created_at: now.clone(),
        modified_at: now,
        training_info: TrainingInfo {
            epochs_trained: network.metrics().epochs_trained,
            final_loss: network.metrics().best_loss,
            best_accuracy: network.metrics().best_accuracy,
            training_time_seconds: network.metrics().training_time_ms / 1000.0,
            dataset_info: None,
        },
        metrics: HashMap::new(),
        custom: HashMap::new(),
    }
}

/// Model validation and compatibility checking
pub mod validation {
    use super::*;

    /// Validate model compatibility
    pub fn validate_model(model: &SerializableModel) -> Result<()> {
        // Check version compatibility
        validate_version(&model.version)?;

        // Validate architecture consistency
        validate_architecture(&model.architecture)?;

        // Validate parameter shapes
        validate_parameters(&model.parameters, &model.architecture)?;

        Ok(())
    }

    /// Check version compatibility
    fn validate_version(version: &str) -> Result<()> {
        let current_version = env!("CARGO_PKG_VERSION");

        // Simple version check - in practice, you'd want semantic versioning
        if version != current_version {
            log::warn!(
                "Model version {} differs from current version {}",
                version,
                current_version
            );
        }

        Ok(())
    }

    /// Validate architecture consistency
    fn validate_architecture(architecture: &ModelArchitecture) -> Result<()> {
        if architecture.layers.is_empty() {
            return Err(NnlError::network("Model must have at least one layer"));
        }

        // TODO: Implement shape consistency validation
        // Would require extracting shape information from LayerConfig

        Ok(())
    }

    /// Validate parameter shapes match architecture
    fn validate_parameters(
        _parameters: &[SerializableTensor],
        _architecture: &ModelArchitecture,
    ) -> Result<()> {
        // TODO: Implement parameter validation
        // Would require calculating expected parameter count from LayerConfig
        Ok(())
    }
}

/// Checkpoint management utilities
pub mod checkpoint {
    use super::*;
    use std::fs;

    /// Save a training checkpoint
    pub fn save_checkpoint<P: AsRef<Path>>(
        model: &Network,
        epoch: usize,
        loss: f32,
        checkpoint_dir: P,
    ) -> Result<()> {
        let checkpoint_dir = checkpoint_dir.as_ref();
        fs::create_dir_all(checkpoint_dir)?;

        let filename = format!("checkpoint_epoch_{:04}_loss_{:.6}.bin", epoch, loss);
        let path = checkpoint_dir.join(filename);

        let metadata = ModelMetadata {
            name: format!("Checkpoint Epoch {}", epoch),
            description: format!(
                "Training checkpoint at epoch {} with loss {:.6}",
                epoch, loss
            ),
            created_at: chrono::Utc::now().to_rfc3339(),
            modified_at: chrono::Utc::now().to_rfc3339(),
            training_info: TrainingInfo {
                epochs_trained: epoch,
                final_loss: loss,
                best_accuracy: 0.0,
                training_time_seconds: 0.0,
                dataset_info: None,
            },
            metrics: HashMap::new(),
            custom: HashMap::new(),
        };

        save_model(model, path, ModelFormat::Binary, Some(metadata))
    }

    /// Load the latest checkpoint from directory
    pub fn load_latest_checkpoint<P: AsRef<Path>>(
        checkpoint_dir: P,
    ) -> Result<Option<SerializableModel>> {
        let checkpoint_dir = checkpoint_dir.as_ref();

        if !checkpoint_dir.exists() {
            return Ok(None);
        }

        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                    if filename.starts_with("checkpoint_epoch_") {
                        checkpoints.push(path);
                    }
                }
            }
        }

        if checkpoints.is_empty() {
            return Ok(None);
        }

        // Sort by modification time, newest first
        checkpoints.sort_by_key(|path| {
            fs::metadata(path)
                .and_then(|meta| meta.modified())
                .unwrap_or(std::time::UNIX_EPOCH)
        });
        checkpoints.reverse();

        let latest = &checkpoints[0];
        let model = load_model(latest, ModelFormat::Binary)?;
        Ok(Some(model))
    }

    /// Clean up old checkpoints, keeping only the N most recent
    pub fn cleanup_checkpoints<P: AsRef<Path>>(
        checkpoint_dir: P,
        keep_count: usize,
    ) -> Result<usize> {
        let checkpoint_dir = checkpoint_dir.as_ref();

        if !checkpoint_dir.exists() {
            return Ok(0);
        }

        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                    if filename.starts_with("checkpoint_epoch_") {
                        let modified = fs::metadata(&path)
                            .and_then(|meta| meta.modified())
                            .unwrap_or(std::time::UNIX_EPOCH);
                        checkpoints.push((path, modified));
                    }
                }
            }
        }

        if checkpoints.len() <= keep_count {
            return Ok(0);
        }

        // Sort by modification time, newest first
        checkpoints.sort_by_key(|(_, time)| *time);
        checkpoints.reverse();

        // Remove old checkpoints
        let mut removed = 0;
        for (path, _) in checkpoints.iter().skip(keep_count) {
            if fs::remove_file(path).is_ok() {
                removed += 1;
            }
        }

        Ok(removed)
    }
}

/// Model export utilities for different frameworks
pub mod export {
    use super::*;

    /// Export model in ONNX-like format (simplified)
    pub fn export_onnx<P: AsRef<Path>>(model: &SerializableModel, path: P) -> Result<()> {
        // This would be a complex implementation to convert to ONNX format
        // For now, just save as JSON with ONNX-like structure
        let onnx_model = OnnxLikeModel {
            ir_version: 7,
            producer_name: "nnl".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            model_version: 1,
            graph: GraphProto {
                name: model.metadata.name.clone(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                nodes: convert_layers_to_nodes(&model.architecture.layers),
                initializers: Vec::new(),
            },
        };

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &onnx_model)?;
        Ok(())
    }

    #[derive(Serialize)]
    struct OnnxLikeModel {
        ir_version: i32,
        producer_name: String,
        producer_version: String,
        model_version: i32,
        graph: GraphProto,
    }

    #[derive(Serialize)]
    struct GraphProto {
        name: String,
        inputs: Vec<ValueInfoProto>,
        outputs: Vec<ValueInfoProto>,
        nodes: Vec<NodeProto>,
        initializers: Vec<TensorProto>,
    }

    #[derive(Serialize)]
    struct ValueInfoProto {
        name: String,
        type_info: TypeProto,
    }

    #[derive(Serialize)]
    struct TypeProto {
        tensor_type: TensorTypeProto,
    }

    #[derive(Serialize)]
    struct TensorTypeProto {
        elem_type: i32,
        shape: Vec<i64>,
    }

    #[derive(Serialize)]
    struct NodeProto {
        name: String,
        op_type: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: HashMap<String, serde_json::Value>,
    }

    #[derive(Serialize)]
    struct TensorProto {
        name: String,
        data_type: i32,
        dims: Vec<i64>,
        raw_data: Vec<u8>,
    }

    fn convert_layers_to_nodes(layers: &[crate::layers::LayerConfig]) -> Vec<NodeProto> {
        layers
            .iter()
            .enumerate()
            .map(|(i, layer)| NodeProto {
                name: format!("layer_{}", i),
                op_type: match layer {
                    crate::layers::LayerConfig::Dense { .. } => "Dense".to_string(),
                    crate::layers::LayerConfig::Conv2D { .. } => "Conv2D".to_string(),
                    crate::layers::LayerConfig::MaxPool2D { .. } => "MaxPool2D".to_string(),
                    crate::layers::LayerConfig::AvgPool2D { .. } => "AvgPool2D".to_string(),
                    crate::layers::LayerConfig::Flatten { .. } => "Flatten".to_string(),
                    crate::layers::LayerConfig::Reshape { .. } => "Reshape".to_string(),
                    crate::layers::LayerConfig::Dropout { .. } => "Dropout".to_string(),
                    crate::layers::LayerConfig::BatchNorm { .. } => "BatchNorm".to_string(),
                    crate::layers::LayerConfig::LayerNorm { .. } => "LayerNorm".to_string(),
                },
                inputs: vec![format!("input_{}", i)],
                outputs: vec![format!("output_{}", i)],
                attributes: HashMap::new(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_format_detection() {
        assert_eq!(
            detect_format_from_extension("model.bin").unwrap(),
            ModelFormat::Binary
        );
        assert_eq!(
            detect_format_from_extension("model.json").unwrap(),
            ModelFormat::Json
        );
        assert_eq!(
            detect_format_from_extension("model.msgpack").unwrap(),
            ModelFormat::MessagePack
        );

        assert!(detect_format_from_extension("model.txt").is_err());
    }

    #[test]
    fn test_serializable_model_creation() {
        let metadata = ModelMetadata {
            name: "Test Model".to_string(),
            description: "A test model".to_string(),
            created_at: "2023-01-01T00:00:00Z".to_string(),
            modified_at: "2023-01-01T00:00:00Z".to_string(),
            training_info: TrainingInfo {
                epochs_trained: 100,
                final_loss: 0.1,
                best_accuracy: 0.95,
                training_time_seconds: 300.0,
                dataset_info: None,
            },
            metrics: HashMap::new(),
            custom: HashMap::new(),
        };

        let architecture = ModelArchitecture {
            layers: Vec::new(),
            loss_function: crate::losses::LossFunction::MeanSquaredError,
            optimizer_config: crate::optimizers::OptimizerConfig::SGD {
                learning_rate: 0.01,
                momentum: None,
                weight_decay: None,
                nesterov: false,
            },
            input_shape: vec![784],
            output_shape: vec![10],
            device_type: "CPU".to_string(),
        };

        let model = SerializableModel {
            architecture,
            parameters: Vec::new(),
            optimizer_state: HashMap::new(),
            metadata,
            version: "0.1.0".to_string(),
        };

        assert_eq!(model.metadata.name, "Test Model");
        assert_eq!(model.version, "0.1.0");
    }

    #[test]
    fn test_json_serialization() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test_model.json");

        let model = SerializableModel {
            architecture: ModelArchitecture {
                layers: Vec::new(),
                loss_function: crate::losses::LossFunction::MeanSquaredError,
                optimizer_config: crate::optimizers::OptimizerConfig::SGD {
                    learning_rate: 0.01,
                    momentum: None,
                    weight_decay: None,
                    nesterov: false,
                },
                input_shape: vec![784],
                output_shape: vec![10],
                device_type: "CPU".to_string(),
            },
            parameters: Vec::new(),
            optimizer_state: HashMap::new(),
            metadata: ModelMetadata {
                name: "Test".to_string(),
                description: "Test model".to_string(),
                created_at: "2023-01-01T00:00:00Z".to_string(),
                modified_at: "2023-01-01T00:00:00Z".to_string(),
                training_info: TrainingInfo {
                    epochs_trained: 10,
                    final_loss: 0.5,
                    best_accuracy: 0.8,
                    training_time_seconds: 60.0,
                    dataset_info: None,
                },
                metrics: HashMap::new(),
                custom: HashMap::new(),
            },
            version: "0.1.0".to_string(),
        };

        save_json(&model, &path)?;
        let loaded = load_json(&path)?;

        assert_eq!(loaded.metadata.name, model.metadata.name);
        assert_eq!(loaded.version, model.version);

        Ok(())
    }

    #[test]
    fn test_model_validation() {
        let architecture = ModelArchitecture {
            layers: Vec::new(),
            loss_function: crate::losses::LossFunction::MeanSquaredError,
            optimizer_config: crate::optimizers::OptimizerConfig::SGD {
                learning_rate: 0.01,
                momentum: None,
                weight_decay: None,
                nesterov: false,
            },
            input_shape: vec![10],
            output_shape: vec![1],
            device_type: "CPU".to_string(),
        };

        let model = SerializableModel {
            architecture,
            parameters: Vec::new(),
            optimizer_state: HashMap::new(),
            metadata: ModelMetadata {
                name: "Test".to_string(),
                description: "Test".to_string(),
                created_at: "2023-01-01T00:00:00Z".to_string(),
                modified_at: "2023-01-01T00:00:00Z".to_string(),
                training_info: TrainingInfo {
                    epochs_trained: 0,
                    final_loss: 0.0,
                    best_accuracy: 0.0,
                    training_time_seconds: 0.0,
                    dataset_info: None,
                },
                metrics: HashMap::new(),
                custom: HashMap::new(),
            },
            version: "0.1.0".to_string(),
        };

        assert!(validation::validate_model(&model).is_ok());
    }
}
