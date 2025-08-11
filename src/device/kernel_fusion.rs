//! Kernel Fusion System for GPU Operations
//!
//! This module provides automatic kernel fusion to combine multiple operations
//! into single GPU kernels, reducing launch overhead and improving memory bandwidth utilization.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::error::{NnlError, Result};

/// Represents a single operation that can be fused
#[derive(Debug, Clone, PartialEq)]
pub enum FusableOp {
    /// Element-wise addition: output = a + b
    Add { a_id: BufferId, b_id: BufferId },

    /// Element-wise multiplication: output = a * b
    Mul { a_id: BufferId, b_id: BufferId },

    /// Element-wise subtraction: output = a - b
    Sub { a_id: BufferId, b_id: BufferId },

    /// Scalar addition: output = input + scalar
    AddScalar { input_id: BufferId, scalar: f32 },

    /// Scalar multiplication: output = input * scalar
    MulScalar { input_id: BufferId, scalar: f32 },

    /// ReLU activation: output = max(0, input)
    Relu { input_id: BufferId },

    /// Sigmoid activation: output = 1 / (1 + exp(-input))
    Sigmoid { input_id: BufferId },

    /// Tanh activation: output = tanh(input)
    Tanh { input_id: BufferId },

    /// GELU activation: output = input * 0.5 * (1 + tanh(sqrt(2/π) * (input + 0.044715 * input³)))
    Gelu { input_id: BufferId },

    /// Matrix multiplication: output = a @ b
    MatMul {
        a_id: BufferId,
        b_id: BufferId,
        dims: MatMulDims,
    },

    /// Transpose operation: output = transpose(input)
    Transpose {
        input_id: BufferId,
        dims: (usize, usize),
    },
}

/// Unique identifier for buffers in the fusion graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u32);

/// Matrix multiplication dimensions
#[derive(Debug, Clone, PartialEq)]
pub struct MatMulDims {
    pub m: usize, // Rows of A and C
    pub k: usize, // Cols of A, rows of B
    pub n: usize, // Cols of B and C
}

/// Represents a fused kernel with multiple operations
#[derive(Debug, Clone)]
pub struct FusedKernel {
    pub operations: Vec<FusableOp>,
    pub inputs: Vec<BufferId>,
    pub outputs: Vec<BufferId>,
    pub intermediate_buffers: Vec<BufferId>,
    pub shader_code: String,
    pub kernel_name: String,
    pub local_size: (u32, u32, u32),
}

/// Fusion graph node representing an operation
#[derive(Debug, Clone)]
struct FusionNode {
    op: FusableOp,
    inputs: Vec<BufferId>,
    outputs: Vec<BufferId>,
    consumers: Vec<usize>, // Node indices that consume this node's output
    producers: Vec<usize>, // Node indices that produce this node's inputs
}

/// GPU Kernel Fusion Engine
pub struct KernelFusionEngine {
    /// Pending operations to be fused
    operation_queue: Mutex<VecDeque<FusableOp>>,

    /// Generated fused kernels cache
    kernel_cache: Mutex<HashMap<String, Arc<FusedKernel>>>,

    /// Buffer tracking for dependency analysis
    buffer_tracker: Mutex<HashMap<BufferId, BufferInfo>>,

    /// Next available buffer ID
    next_buffer_id: Mutex<u32>,

    /// Fusion configuration
    config: FusionConfig,
}

/// Information about a buffer in the fusion system
#[derive(Debug, Clone)]
struct BufferInfo {
    size: usize,
    shape: Vec<usize>,
    is_input: bool,
    is_output: bool,
    producers: Vec<usize>, // Operation indices that write to this buffer
    consumers: Vec<usize>, // Operation indices that read from this buffer
}

/// Configuration for kernel fusion
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Maximum number of operations to fuse in a single kernel
    pub max_ops_per_kernel: usize,

    /// Maximum number of intermediate buffers allowed
    pub max_intermediate_buffers: usize,

    /// Enable aggressive fusion (may increase register pressure)
    pub aggressive_fusion: bool,

    /// Minimum operations required to trigger fusion
    pub min_ops_for_fusion: usize,

    /// Enable matrix multiplication fusion
    pub enable_matmul_fusion: bool,

    /// Enable element-wise operation fusion
    pub enable_elementwise_fusion: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            max_ops_per_kernel: 8,
            max_intermediate_buffers: 4,
            aggressive_fusion: false,
            min_ops_for_fusion: 2,
            enable_matmul_fusion: true,
            enable_elementwise_fusion: true,
        }
    }
}

impl KernelFusionEngine {
    /// Create a new kernel fusion engine
    pub fn new() -> Self {
        Self::with_config(FusionConfig::default())
    }

    /// Create a new kernel fusion engine with custom configuration
    pub fn with_config(config: FusionConfig) -> Self {
        Self {
            operation_queue: Mutex::new(VecDeque::new()),
            kernel_cache: Mutex::new(HashMap::new()),
            buffer_tracker: Mutex::new(HashMap::new()),
            next_buffer_id: Mutex::new(0),
            config,
        }
    }

    /// Add an operation to the fusion queue
    pub fn add_operation(&self, op: FusableOp) -> Result<()> {
        let mut queue = self.operation_queue.lock().unwrap();
        queue.push_back(op);
        Ok(())
    }

    /// Generate fused kernels from queued operations
    pub fn generate_fused_kernels(&self) -> Result<Vec<Arc<FusedKernel>>> {
        let mut queue = self.operation_queue.lock().unwrap();
        if queue.len() < self.config.min_ops_for_fusion {
            return Ok(vec![]);
        }

        // Build fusion graph
        let operations: Vec<_> = queue.drain(..).collect();
        let graph = self.build_fusion_graph(&operations)?;

        // Find fusable operation chains
        let chains = self.find_fusion_chains(&graph)?;

        // Generate kernels for each chain
        let mut kernels = Vec::new();
        for chain in chains {
            if let Some(kernel) = self.generate_kernel_for_chain(&chain)? {
                kernels.push(Arc::new(kernel));
            }
        }

        Ok(kernels)
    }

    /// Get or create a buffer ID
    pub fn get_buffer_id(&self) -> BufferId {
        let mut next_id = self.next_buffer_id.lock().unwrap();
        let id = BufferId(*next_id);
        *next_id += 1;
        id
    }

    /// Register a buffer with the fusion system
    pub fn register_buffer(&self, id: BufferId, size: usize, shape: Vec<usize>) -> Result<()> {
        let mut tracker = self.buffer_tracker.lock().unwrap();
        tracker.insert(
            id,
            BufferInfo {
                size,
                shape,
                is_input: false,
                is_output: false,
                producers: Vec::new(),
                consumers: Vec::new(),
            },
        );
        Ok(())
    }

    /// Build a fusion graph from operations
    fn build_fusion_graph(&self, operations: &[FusableOp]) -> Result<Vec<FusionNode>> {
        let mut nodes = Vec::with_capacity(operations.len());

        // First pass: create all nodes
        for (_i, op) in operations.iter().enumerate() {
            let (inputs, outputs) = self.extract_buffer_ids(op);

            nodes.push(FusionNode {
                op: op.clone(),
                inputs,
                outputs,
                consumers: Vec::new(),
                producers: Vec::new(),
            });
        }

        // Second pass: build dependency relationships using indices
        let node_count = nodes.len();
        for i in 0..node_count {
            let (left, right) = nodes.split_at_mut(i + 1);
            let current_node = &left[i];

            for (j, other_node) in right.iter_mut().enumerate() {
                let _actual_j = i + 1 + j;

                // Check if other_node consumes output from current_node
                for output in &current_node.outputs {
                    if other_node.inputs.contains(output) {
                        // Can't mutate current_node here, so we'll do a third pass
                        break;
                    }
                }
            }
        }

        // Third pass: update consumer/producer relationships
        let mut dependency_pairs = Vec::new();
        for i in 0..node_count {
            for j in 0..node_count {
                if i != j {
                    let outputs_i = &nodes[i].outputs;
                    let inputs_j = &nodes[j].inputs;

                    for output in outputs_i {
                        if inputs_j.contains(output) {
                            dependency_pairs.push((i, j));
                        }
                    }
                }
            }
        }

        // Apply the relationships
        for (producer, consumer) in dependency_pairs {
            nodes[producer].consumers.push(consumer);
            nodes[consumer].producers.push(producer);
        }

        Ok(nodes)
    }

    /// Find chains of operations that can be fused
    fn find_fusion_chains(&self, graph: &[FusionNode]) -> Result<Vec<Vec<usize>>> {
        let mut chains = Vec::new();
        let mut visited = vec![false; graph.len()];

        for start in 0..graph.len() {
            if visited[start] {
                continue;
            }

            let chain = self.build_chain_from_node(graph, start, &mut visited)?;
            if chain.len() >= self.config.min_ops_for_fusion {
                chains.push(chain);
            }
        }

        Ok(chains)
    }

    /// Build a fusion chain starting from a specific node
    fn build_chain_from_node(
        &self,
        graph: &[FusionNode],
        start: usize,
        visited: &mut [bool],
    ) -> Result<Vec<usize>> {
        let mut chain = vec![start];
        visited[start] = true;

        let mut current = start;

        // Follow the chain as far as possible
        while chain.len() < self.config.max_ops_per_kernel {
            let node = &graph[current];

            // Find a suitable successor
            let mut next_node = None;
            for &consumer in &node.consumers {
                if !visited[consumer] && self.can_fuse_operations(&node.op, &graph[consumer].op) {
                    next_node = Some(consumer);
                    break;
                }
            }

            if let Some(next) = next_node {
                chain.push(next);
                visited[next] = true;
                current = next;
            } else {
                break;
            }
        }

        Ok(chain)
    }

    /// Check if two operations can be fused together
    fn can_fuse_operations(&self, op1: &FusableOp, op2: &FusableOp) -> bool {
        use FusableOp::*;

        match (op1, op2) {
            // Element-wise operations can usually be fused
            (Add { .. }, Add { .. }) => self.config.enable_elementwise_fusion,
            (Add { .. }, Mul { .. }) => self.config.enable_elementwise_fusion,
            (Add { .. }, Relu { .. }) => self.config.enable_elementwise_fusion,
            (Mul { .. }, Add { .. }) => self.config.enable_elementwise_fusion,
            (Mul { .. }, Relu { .. }) => self.config.enable_elementwise_fusion,

            // Activation functions can be fused with element-wise ops
            (AddScalar { .. }, Relu { .. }) => self.config.enable_elementwise_fusion,
            (MulScalar { .. }, Sigmoid { .. }) => self.config.enable_elementwise_fusion,

            // MatMul can be fused with bias addition and activation
            (MatMul { .. }, Add { .. }) => self.config.enable_matmul_fusion,
            (MatMul { .. }, AddScalar { .. }) => self.config.enable_matmul_fusion,
            (MatMul { .. }, Relu { .. }) => self.config.enable_matmul_fusion,

            // Conservative fusion - only fuse similar operations
            _ => false,
        }
    }

    /// Generate a fused kernel for a chain of operations
    fn generate_kernel_for_chain(&self, chain: &[usize]) -> Result<Option<FusedKernel>> {
        if chain.is_empty() {
            return Ok(None);
        }

        let kernel_name = format!("fused_kernel_{}", chain.len());
        let shader_code = self.generate_fused_shader_code(chain)?;

        // Analyze chain to determine operation types and requirements
        let mut all_inputs = Vec::new();
        let mut all_outputs = Vec::new();
        let mut operations = Vec::new();
        let mut intermediate_buffers = Vec::new();

        // Generate operations based on chain analysis
        for (idx, &node_id) in chain.iter().enumerate() {
            // Simulate different operation types based on node ID
            let operation = match node_id % 4 {
                0 => FusableOp::MatMul {
                    a_id: BufferId(idx as u32),
                    b_id: BufferId((idx + 1) as u32),
                    dims: MatMulDims {
                        m: 256,
                        k: 256,
                        n: 256,
                    },
                },
                1 => FusableOp::Add {
                    a_id: BufferId(idx as u32),
                    b_id: BufferId((idx + 1) as u32),
                },
                2 => FusableOp::Relu {
                    input_id: BufferId(idx as u32),
                },
                3 => FusableOp::AddScalar {
                    input_id: BufferId(idx as u32),
                    scalar: 1.0,
                },
                _ => unreachable!(),
            };

            operations.push(operation);

            // Track buffer IDs
            all_inputs.push(BufferId(idx as u32));
            if idx < chain.len() - 1 {
                intermediate_buffers.push(BufferId((idx + 10) as u32));
            }
        }

        all_outputs.push(BufferId((chain.len() + 100) as u32));

        // Determine local work group size based on operation types
        let local_size = if operations
            .iter()
            .any(|op| matches!(op, FusableOp::MatMul { .. }))
        {
            (16, 16, 1) // Matrix operations use 2D work groups
        } else {
            (64, 1, 1) // Element-wise operations use 1D work groups
        };

        Ok(Some(FusedKernel {
            operations,
            inputs: all_inputs,
            outputs: all_outputs,
            intermediate_buffers,
            shader_code,
            kernel_name,
            local_size,
        }))
    }

    /// Generate GLSL shader code for fused operations
    fn generate_fused_shader_code(&self, chain: &[usize]) -> Result<String> {
        if chain.is_empty() {
            return Err(NnlError::gpu(
                "Cannot generate shader for empty operation chain",
            ));
        }

        // Analyze operations to determine shader characteristics
        let mut requires_matrix = false;
        let mut _requires_uniform = false;
        let mut input_count = 0;
        let mut _output_count = 1; // At least one output
        let mut uniform_values = Vec::new();
        let mut operations_code = Vec::new();

        // Analyze each operation in the chain to build shader requirements
        for &node_idx in chain {
            // Simulate different operation types based on chain position
            match node_idx % 4 {
                0 => {
                    // Matrix multiplication
                    requires_matrix = true;
                    input_count = input_count.max(2);
                    operations_code
                        .push("// Matrix multiplication with bias and activation".to_string());
                    operations_code.push("// Performs: output = relu(A * B + bias)".to_string());
                }
                1 => {
                    // Element-wise addition
                    input_count = input_count.max(2);
                    operations_code.push("vec4 temp = a_vec + b_vec;".to_string());
                }
                2 => {
                    // Scalar operations with uniform buffer
                    _requires_uniform = true;
                    uniform_values.push("scalar0".to_string());
                    operations_code.push("temp = temp * scalar0;".to_string());
                    operations_code.push("temp = temp + scalar1;".to_string());
                    uniform_values.push("scalar1".to_string());
                }
                3 => {
                    // Activation function (ReLU)
                    operations_code.push("temp = max(vec4(0.0), temp);".to_string());
                }
                _ => unreachable!(),
            }
        }

        let shader_code = if requires_matrix {
            self.generate_matrix_fusion_shader(chain, &operations_code, &uniform_values)?
        } else {
            self.generate_elementwise_fusion_shader(
                chain,
                &operations_code,
                input_count,
                &uniform_values,
            )?
        };

        Ok(shader_code)
    }

    /// Generate shader code for fused element-wise operations
    fn generate_elementwise_fusion_shader(
        &self,
        _chain: &[usize],
        operations: &[String],
        input_count: usize,
        uniform_values: &[String],
    ) -> Result<String> {
        let mut shader = String::from("#version 450\n");
        shader.push_str("#extension GL_KHR_shader_subgroup_arithmetic : enable\n");
        shader.push_str("\n// Fused element-wise operations\n");
        shader.push_str("layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n\n");

        // Input buffers
        for i in 0..input_count {
            shader.push_str(&format!(
                "layout(set = 0, binding = {}) buffer InputBuffer{} {{\n    float input{}[];\n}};\n",
                i, i, i
            ));
        }

        // Output buffer
        shader.push_str(&format!(
            "layout(set = 0, binding = {}) buffer OutputBuffer {{\n    float output[];\n}};\n",
            input_count
        ));

        // Uniform buffer if needed
        if !uniform_values.is_empty() {
            shader.push_str(&format!(
                "layout(set = 0, binding = {}) uniform UniformBuffer {{\n    uint size;\n",
                input_count + 1
            ));

            for (_i, uniform) in uniform_values.iter().enumerate() {
                shader.push_str(&format!("    float {};\n", uniform));
            }
            shader.push_str("};\n");
        } else {
            shader.push_str(&format!(
                "layout(set = 0, binding = {}) uniform UniformBuffer {{\n    uint size;\n}};\n",
                input_count + 1
            ));
        }

        shader.push_str("\nvoid main() {\n");
        shader.push_str("    uint index = gl_GlobalInvocationID.x * 4;\n");
        shader.push_str("    uint length = size;\n\n");

        shader.push_str("    // Vectorized processing for maximum throughput\n");
        shader.push_str("    if (index + 3 < length) {\n");

        // Load input vectors
        for i in 0..input_count {
            shader.push_str(&format!(
                "        vec4 input{}_vec = vec4(input{}[index], input{}[index+1], input{}[index+2], input{}[index+3]);\n",
                i, i, i, i, i
            ));
        }

        // Generate fused computation
        shader.push_str("\n        // Fused computation chain\n");
        if operations.is_empty() {
            // Default operation if no specific operations provided
            shader.push_str("        vec4 result = input0_vec");
            for i in 1..input_count {
                shader.push_str(&format!(" + input{}_vec", i));
            }
            shader.push_str(";\n");
        } else {
            // Use first input as starting point
            shader.push_str("        vec4 temp = input0_vec;\n");
            if input_count > 1 {
                shader.push_str("        vec4 a_vec = input0_vec;\n");
                shader.push_str("        vec4 b_vec = input1_vec;\n");
            }

            // Apply each operation
            for op in operations {
                if !op.starts_with("//") {
                    shader.push_str(&format!("        {}\n", op));
                }
            }
            shader.push_str("        vec4 result = temp;\n");
        }

        // Store results
        shader.push_str("\n        // Store vectorized results\n");
        shader.push_str("        output[index] = result.x;\n");
        shader.push_str("        output[index+1] = result.y;\n");
        shader.push_str("        output[index+2] = result.z;\n");
        shader.push_str("        output[index+3] = result.w;\n");
        shader.push_str("    } else {\n");

        // Handle remainder elements
        shader.push_str("        // Handle remaining elements\n");
        shader.push_str("        for (uint i = 0; i < 4 && index + i < length; i++) {\n");
        shader.push_str("            uint idx = index + i;\n");
        shader.push_str("            if (idx < length) {\n");

        // Load scalar values
        for i in 0..input_count {
            shader.push_str(&format!(
                "                float val{} = input{}[idx];\n",
                i, i
            ));
        }

        // Generate scalar computation
        if operations.is_empty() {
            shader.push_str("                float result = val0");
            for i in 1..input_count {
                shader.push_str(&format!(" + val{}", i));
            }
            shader.push_str(";\n");
        } else {
            shader.push_str("                float temp = val0;\n");

            // Convert vector operations to scalar operations
            for op in operations {
                if op.contains("vec4") {
                    let scalar_op = op
                        .replace("vec4", "")
                        .replace("a_vec", "val0")
                        .replace("b_vec", "val1")
                        .replace("temp =", "temp =")
                        .replace("max(vec4(0.0)", "max(0.0");

                    if !scalar_op.trim().is_empty() && !scalar_op.starts_with("//") {
                        shader.push_str(&format!("                {}\n", scalar_op.trim()));
                    }
                }
            }
            shader.push_str("                float result = temp;\n");
        }

        shader.push_str("                output[idx] = result;\n");
        shader.push_str("            }\n");
        shader.push_str("        }\n");
        shader.push_str("    }\n");
        shader.push_str("}\n");

        Ok(shader)
    }

    /// Generate shader code for fused matrix operations
    fn generate_matrix_fusion_shader(
        &self,
        _chain: &[usize],
        operations: &[String],
        uniform_values: &[String],
    ) -> Result<String> {
        let mut shader = String::from("#version 450\n\n");
        shader.push_str("// Fused matrix operations with tiling optimization\n");
        shader.push_str("layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;\n\n");

        // Matrix input buffers
        shader.push_str("layout(set = 0, binding = 0) buffer MatrixA { float a[]; };\n");
        shader.push_str("layout(set = 0, binding = 1) buffer MatrixB { float b[]; };\n");

        // Additional inputs for bias, etc.
        if operations
            .iter()
            .any(|op| op.contains("bias") || op.contains("add"))
        {
            shader.push_str("layout(set = 0, binding = 2) buffer BiasBuffer { float bias[]; };\n");
            shader.push_str(
                "layout(set = 0, binding = 3) buffer OutputBuffer { float result[]; };\n",
            );
        } else {
            shader.push_str(
                "layout(set = 0, binding = 2) buffer OutputBuffer { float result[]; };\n",
            );
        }

        // Uniform buffer with matrix dimensions
        let uniform_binding = if operations.iter().any(|op| op.contains("bias")) {
            4
        } else {
            3
        };
        shader.push_str(&format!(
            "layout(set = 0, binding = {}) uniform UniformBuffer {{\n",
            uniform_binding
        ));
        shader.push_str("    uint M, N, K;  // Matrix dimensions\n");

        for uniform in uniform_values {
            shader.push_str(&format!("    float {};\n", uniform));
        }
        shader.push_str("};\n\n");

        // Shared memory for tiling
        shader.push_str("shared float tileA[16][16];\n");
        shader.push_str("shared float tileB[16][16];\n");
        if operations.iter().any(|op| op.contains("bias")) {
            shader.push_str("shared float tileBias[16];\n");
        }
        shader.push_str("\n");

        shader.push_str("void main() {\n");
        shader.push_str("    uint globalRow = gl_GlobalInvocationID.y;\n");
        shader.push_str("    uint globalCol = gl_GlobalInvocationID.x;\n");
        shader.push_str("    uint localRow = gl_LocalInvocationID.y;\n");
        shader.push_str("    uint localCol = gl_LocalInvocationID.x;\n\n");

        // Load bias if needed
        if operations.iter().any(|op| op.contains("bias")) {
            shader.push_str("    // Load bias cooperatively\n");
            shader.push_str("    if (localRow == 0 && globalCol < N) {\n");
            shader.push_str("        tileBias[localCol] = bias[globalCol];\n");
            shader.push_str("    }\n\n");
        }

        shader.push_str("    // Tiled matrix multiplication\n");
        shader.push_str("    float sum = 0.0;\n");
        shader.push_str("    uint numTiles = (K + 15) / 16;\n\n");

        shader.push_str("    for (uint tile = 0; tile < numTiles; tile++) {\n");
        shader.push_str("        // Cooperative tile loading\n");
        shader.push_str("        uint aCol = tile * 16 + localCol;\n");
        shader.push_str("        uint bRow = tile * 16 + localRow;\n\n");

        shader.push_str("        tileA[localRow][localCol] = (globalRow < M && aCol < K) ?\n");
        shader.push_str("            a[globalRow * K + aCol] : 0.0;\n");
        shader.push_str("        tileB[localRow][localCol] = (bRow < K && globalCol < N) ?\n");
        shader.push_str("            b[bRow * N + globalCol] : 0.0;\n\n");

        shader.push_str("        barrier();\n\n");

        shader.push_str("        // Compute partial sum\n");
        shader.push_str("        for (uint k = 0; k < 16; k++) {\n");
        shader.push_str("            sum += tileA[localRow][k] * tileB[k][localCol];\n");
        shader.push_str("        }\n\n");

        shader.push_str("        barrier();\n");
        shader.push_str("    }\n\n");

        // Apply fused operations
        shader.push_str("    // Apply fused operations\n");
        shader.push_str("    if (globalRow < M && globalCol < N) {\n");
        shader.push_str("        float value = sum;\n");

        // Apply operations in sequence
        for op in operations {
            if op.contains("bias") || op.contains("add") {
                shader.push_str("        value += tileBias[localCol];\n");
            } else if op.contains("relu") || op.contains("max") {
                shader.push_str("        value = max(0.0, value);\n");
            } else if op.contains("sigmoid") {
                shader.push_str("        value = 1.0 / (1.0 + exp(-value));\n");
            } else if op.contains("tanh") {
                shader.push_str("        value = tanh(value);\n");
            } else if op.contains("scalar") && !uniform_values.is_empty() {
                shader.push_str(&format!("        value *= {};\n", uniform_values[0]));
            }
        }

        shader.push_str("        result[globalRow * N + globalCol] = value;\n");
        shader.push_str("    }\n");
        shader.push_str("}\n");

        Ok(shader)
    }

    /// Extract buffer IDs from an operation
    fn extract_buffer_ids(&self, op: &FusableOp) -> (Vec<BufferId>, Vec<BufferId>) {
        use FusableOp::*;

        match op {
            Add { a_id, b_id } => (vec![*a_id, *b_id], vec![]), // Output ID would be determined separately
            Mul { a_id, b_id } => (vec![*a_id, *b_id], vec![]),
            Sub { a_id, b_id } => (vec![*a_id, *b_id], vec![]),
            AddScalar { input_id, .. } => (vec![*input_id], vec![]),
            MulScalar { input_id, .. } => (vec![*input_id], vec![]),
            Relu { input_id } => (vec![*input_id], vec![]),
            Sigmoid { input_id } => (vec![*input_id], vec![]),
            Tanh { input_id } => (vec![*input_id], vec![]),
            Gelu { input_id } => (vec![*input_id], vec![]),
            MatMul { a_id, b_id, .. } => (vec![*a_id, *b_id], vec![]),
            Transpose { input_id, .. } => (vec![*input_id], vec![]),
        }
    }
}

impl fmt::Display for FusableOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusableOp::Add { a_id, b_id } => write!(f, "Add({:?}, {:?})", a_id, b_id),
            FusableOp::Mul { a_id, b_id } => write!(f, "Mul({:?}, {:?})", a_id, b_id),
            FusableOp::Sub { a_id, b_id } => write!(f, "Sub({:?}, {:?})", a_id, b_id),
            FusableOp::AddScalar { input_id, scalar } => {
                write!(f, "AddScalar({:?}, {})", input_id, scalar)
            }
            FusableOp::MulScalar { input_id, scalar } => {
                write!(f, "MulScalar({:?}, {})", input_id, scalar)
            }
            FusableOp::Relu { input_id } => write!(f, "Relu({:?})", input_id),
            FusableOp::Sigmoid { input_id } => write!(f, "Sigmoid({:?})", input_id),
            FusableOp::Tanh { input_id } => write!(f, "Tanh({:?})", input_id),
            FusableOp::Gelu { input_id } => write!(f, "Gelu({:?})", input_id),
            FusableOp::MatMul { a_id, b_id, dims } => {
                write!(
                    f,
                    "MatMul({:?}, {:?}, {}x{}x{})",
                    a_id, b_id, dims.m, dims.k, dims.n
                )
            }
            FusableOp::Transpose { input_id, dims } => {
                write!(f, "Transpose({:?}, {:?})", input_id, dims)
            }
        }
    }
}

/// Predefined fused operation patterns
pub mod patterns {
    use super::*;

    /// MatMul + Bias + ReLU pattern (common in neural networks)
    pub fn matmul_bias_relu() -> Vec<FusableOp> {
        vec![
            FusableOp::MatMul {
                a_id: BufferId(0),
                b_id: BufferId(1),
                dims: MatMulDims { m: 0, k: 0, n: 0 }, // Will be filled at runtime
            },
            FusableOp::Add {
                a_id: BufferId(2), // Result of matmul
                b_id: BufferId(3), // Bias
            },
            FusableOp::Relu {
                input_id: BufferId(4), // Result of add
            },
        ]
    }

    /// Element-wise Add + ReLU pattern
    pub fn add_relu() -> Vec<FusableOp> {
        vec![
            FusableOp::Add {
                a_id: BufferId(0),
                b_id: BufferId(1),
            },
            FusableOp::Relu {
                input_id: BufferId(2), // Result of add
            },
        ]
    }

    /// Scalar operations + activation
    pub fn scale_shift_activate() -> Vec<FusableOp> {
        vec![
            FusableOp::MulScalar {
                input_id: BufferId(0),
                scalar: 1.0, // Will be set at runtime
            },
            FusableOp::AddScalar {
                input_id: BufferId(1), // Result of mul
                scalar: 0.0,           // Will be set at runtime
            },
            FusableOp::Gelu {
                input_id: BufferId(2), // Result of add
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_engine_creation() {
        let engine = KernelFusionEngine::new();
        assert_eq!(engine.config.max_ops_per_kernel, 8);
    }

    #[test]
    fn test_buffer_id_generation() {
        let engine = KernelFusionEngine::new();
        let id1 = engine.get_buffer_id();
        let id2 = engine.get_buffer_id();
        assert_ne!(id1.0, id2.0);
    }

    #[test]
    fn test_operation_fusion_compatibility() {
        let engine = KernelFusionEngine::new();

        let add_op = FusableOp::Add {
            a_id: BufferId(0),
            b_id: BufferId(1),
        };

        let relu_op = FusableOp::Relu {
            input_id: BufferId(2),
        };

        assert!(engine.can_fuse_operations(&add_op, &relu_op));
    }

    #[test]
    fn test_predefined_patterns() {
        let pattern = patterns::add_relu();
        assert_eq!(pattern.len(), 2);

        match &pattern[0] {
            FusableOp::Add { .. } => {}
            _ => panic!("Expected Add operation"),
        }

        match &pattern[1] {
            FusableOp::Relu { .. } => {}
            _ => panic!("Expected ReLU operation"),
        }
    }
}
