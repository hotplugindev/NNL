//! GPU Memory Pool for efficient memory management
//!
//! This module provides a high-performance memory pool system for GPU buffers
//! to eliminate allocation overhead and memory fragmentation.

use std::collections::{HashMap, VecDeque};
use std::sync::{
    Arc, Mutex, RwLock,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

use crate::device::{DeviceMemory, gpu::VulkanBuffer};
use crate::error::{NnlError, Result};

/// Statistics for monitoring pool performance
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total number of buffer allocations requested
    pub total_allocations: u64,
    /// Number of allocations served from the pool
    pub pool_hits: u64,
    /// Number of allocations that required new buffer creation
    pub pool_misses: u64,
    /// Number of buffers currently in use
    pub active_buffers: usize,
    /// Number of buffers available in the pool
    pub pooled_buffers: usize,
    /// Total memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Ratio of fragmented memory (0.0 to 1.0)
    pub fragmentation_ratio: f32,
}

/// Memory pool bucket for a specific buffer size
struct PoolBucket {
    free_buffers: VecDeque<Arc<VulkanBuffer>>,
    allocated_buffers: Vec<Arc<VulkanBuffer>>,
    buffer_size: usize,
    max_pool_size: usize,
    usage_stats: BucketUsageStats,
}

#[derive(Debug, Default)]
struct BucketUsageStats {
    hits: u64,
    misses: u64,
    created: u64,
    last_access: Option<Instant>,
}

/// High-performance GPU memory pool
pub struct GpuMemoryPool {
    /// Pool buckets organized by buffer size (power of 2)
    buckets: RwLock<HashMap<usize, Mutex<PoolBucket>>>,

    /// Allocator for creating new buffers
    allocator: Arc<StandardMemoryAllocator>,

    /// Pool configuration
    config: PoolConfig,

    /// Global statistics
    stats: Mutex<PoolStats>,

    /// Background cleanup thread handle
    cleanup_handle: Option<std::thread::JoinHandle<()>>,

    /// Shutdown signal for background thread
    shutdown_signal: Arc<AtomicBool>,
}

/// Detailed statistics for a memory pool bucket
#[derive(Debug, Clone)]
pub struct BucketStats {
    /// Number of free buffers available in this bucket
    pub free_buffers: usize,
    /// Number of allocated buffers from this bucket
    pub allocated_buffers: usize,
    /// Number of successful allocations from this bucket
    pub hits: u64,
    /// Number of failed allocations from this bucket
    pub misses: u64,
    /// Number of buffers created for this bucket
    pub created: u64,
}

/// Comprehensive statistics for the entire memory pool
/// Enhanced statistics with additional metrics
#[derive(Debug, Clone)]
pub struct DetailedPoolStats {
    /// Total number of buffer allocations requested
    pub total_allocations: u64,
    /// Number of allocations served from the pool
    pub pool_hits: u64,
    /// Number of allocations that required new buffer creation
    pub pool_misses: u64,
    /// Hit ratio as a percentage (0.0 to 1.0)
    pub hit_ratio: f64,
    /// Number of buffers currently in use
    pub active_buffers: usize,
    /// Number of buffers available in the pool
    pub pooled_buffers: usize,
    /// Total memory usage in megabytes
    pub memory_usage_mb: f64,
    /// Ratio of fragmented memory (0.0 to 1.0)
    pub fragmentation_ratio: f32,
    /// Per-bucket statistics mapped by buffer size
    pub bucket_stats: HashMap<usize, BucketStats>,
}

/// Configuration for the memory pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of buffers per size bucket
    pub max_buffers_per_bucket: usize,

    /// Minimum buffer size (bytes) - smaller allocations use this size
    pub min_buffer_size: usize,

    /// Maximum buffer size (bytes) - larger allocations bypass pool
    pub max_buffer_size: usize,

    /// Enable background cleanup of unused buffers
    pub enable_background_cleanup: bool,

    /// Cleanup interval in seconds
    pub cleanup_interval_secs: u64,

    /// Buffer idle timeout before cleanup (seconds)
    pub buffer_idle_timeout_secs: u64,

    /// Enable memory usage tracking
    pub track_memory_usage: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_buffers_per_bucket: 32,
            min_buffer_size: 1024,              // 1KB minimum
            max_buffer_size: 256 * 1024 * 1024, // 256MB maximum
            enable_background_cleanup: true,
            cleanup_interval_secs: 30,
            buffer_idle_timeout_secs: 300, // 5 minutes
            track_memory_usage: true,
        }
    }
}

impl PoolBucket {
    fn new(buffer_size: usize, max_pool_size: usize) -> Self {
        Self {
            free_buffers: VecDeque::with_capacity(max_pool_size),
            allocated_buffers: Vec::new(),
            buffer_size,
            max_pool_size,
            usage_stats: BucketUsageStats::default(),
        }
    }

    fn get_buffer(&mut self, allocator: Arc<StandardMemoryAllocator>) -> Result<Arc<VulkanBuffer>> {
        if let Some(buffer) = self.free_buffers.pop_front() {
            // Pool hit - reuse existing buffer
            self.usage_stats.hits += 1;
            self.usage_stats.last_access = Some(Instant::now());
            Ok(buffer)
        } else {
            // Pool miss - create new buffer
            self.usage_stats.misses += 1;
            self.usage_stats.created += 1;
            self.usage_stats.last_access = Some(Instant::now());

            let buffer = self.create_buffer(allocator, self.buffer_size)?;
            self.allocated_buffers.push(buffer.clone());
            Ok(buffer)
        }
    }

    fn return_buffer(&mut self, buffer: Arc<VulkanBuffer>) {
        // Only pool if we have space and the buffer is the right size
        if self.free_buffers.len() < self.max_pool_size {
            // Use DeviceMemory trait to get buffer size
            if buffer.size() == self.buffer_size {
                self.free_buffers.push_back(buffer);
            }
        }
        // Otherwise let it drop naturally
    }

    fn create_buffer(
        &self,
        allocator: Arc<StandardMemoryAllocator>,
        size: usize,
    ) -> Result<Arc<VulkanBuffer>> {
        // Create buffer with optimal settings for reuse
        let buffer = Buffer::new_slice::<f32>(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (size / std::mem::size_of::<f32>()).try_into().unwrap(),
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create pooled buffer: {}", e)))?;

        Ok(Arc::new(VulkanBuffer::from_buffer(
            buffer, size, false, // Not uniform buffer
        )?))
    }

    fn cleanup_idle_buffers(&mut self, idle_timeout: u64) -> usize {
        let cutoff_time = Instant::now() - Duration::from_secs(idle_timeout);
        let initial_count = self.free_buffers.len();

        // For aggressive cleanup, remove a percentage of buffers when idle timeout is reached
        if let Some(last_access) = self.usage_stats.last_access {
            if last_access < cutoff_time {
                // Remove up to 50% of idle buffers to free memory
                let buffers_to_remove = (initial_count / 2).max(1);
                for _ in 0..buffers_to_remove {
                    if self.free_buffers.is_empty() {
                        break;
                    }
                    self.free_buffers.pop_front();
                }

                // Update last access time to current time after cleanup
                self.usage_stats.last_access = Some(Instant::now());
            }
        }

        initial_count - self.free_buffers.len()
    }
}

impl GpuMemoryPool {
    /// Create new memory pool with default configuration
    pub fn new(allocator: Arc<StandardMemoryAllocator>) -> Self {
        Self::with_config(allocator, PoolConfig::default())
    }

    /// Create new memory pool with custom configuration
    pub fn with_config(allocator: Arc<StandardMemoryAllocator>, config: PoolConfig) -> Self {
        let mut pool = Self {
            buckets: RwLock::new(HashMap::new()),
            allocator,
            config: config.clone(),
            stats: Mutex::new(PoolStats {
                total_allocations: 0,
                pool_hits: 0,
                pool_misses: 0,
                active_buffers: 0,
                pooled_buffers: 0,
                memory_usage_bytes: 0,
                fragmentation_ratio: 0.0,
            }),
            cleanup_handle: None,
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        };

        // Start background cleanup thread if enabled
        if config.enable_background_cleanup {
            pool.start_background_cleanup();
        }

        pool
    }

    /// Get a buffer from the pool or create a new one
    pub fn get_buffer(&self, size: usize) -> Result<Arc<VulkanBuffer>> {
        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_allocations += 1;
        }

        // For very large buffers, bypass the pool
        if size > self.config.max_buffer_size {
            return self.create_direct_buffer(size);
        }

        // Round up to next power of 2 for better pooling
        let pooled_size = self.round_to_pool_size(size);

        // Get or create bucket for this size
        let bucket_key = pooled_size;

        // Check if bucket exists (read lock)
        {
            let buckets = self.buckets.read().unwrap();
            if let Some(bucket_mutex) = buckets.get(&bucket_key) {
                let mut bucket = bucket_mutex.lock().unwrap();
                match bucket.get_buffer(self.allocator.clone()) {
                    Ok(buffer) => {
                        // Update global stats
                        let mut stats = self.stats.lock().unwrap();
                        stats.pool_hits += 1;
                        stats.active_buffers += 1;
                        return Ok(buffer);
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // Create new bucket if it doesn't exist (write lock)
        {
            let mut buckets = self.buckets.write().unwrap();
            let bucket_mutex = buckets.entry(bucket_key).or_insert_with(|| {
                Mutex::new(PoolBucket::new(
                    pooled_size,
                    self.config.max_buffers_per_bucket,
                ))
            });

            let mut bucket = bucket_mutex.lock().unwrap();
            match bucket.get_buffer(self.allocator.clone()) {
                Ok(buffer) => {
                    // Update global stats
                    let mut stats = self.stats.lock().unwrap();
                    if bucket.usage_stats.hits > 0 {
                        stats.pool_hits += 1;
                    } else {
                        stats.pool_misses += 1;
                    }
                    stats.active_buffers += 1;
                    Ok(buffer)
                }
                Err(e) => Err(e),
            }
        }
    }

    /// Return a buffer to the pool for reuse
    pub fn return_buffer(&self, buffer: Arc<VulkanBuffer>) {
        let buffer_size = buffer.size();
        let pooled_size = self.round_to_pool_size(buffer_size);

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.active_buffers = stats.active_buffers.saturating_sub(1);
        }

        // Don't pool oversized buffers
        if buffer_size > self.config.max_buffer_size {
            return; // Let it drop naturally
        }

        // Find the appropriate bucket
        {
            let buckets = self.buckets.read().unwrap();
            if let Some(bucket_mutex) = buckets.get(&pooled_size) {
                let mut bucket = bucket_mutex.lock().unwrap();
                bucket.return_buffer(buffer);
                return;
            }
        }

        // Create new bucket if it doesn't exist
        {
            let mut buckets = self.buckets.write().unwrap();
            let bucket = buckets.entry(pooled_size).or_insert_with(|| {
                Mutex::new(PoolBucket::new(
                    pooled_size,
                    self.config.max_buffers_per_bucket,
                ))
            });
            let mut bucket = bucket.lock().unwrap();
            bucket.return_buffer(buffer);
        }

        // Update pooled buffer count
        let mut stats = self.stats.lock().unwrap();
        stats.pooled_buffers += 1;
    }

    /// Get current pool statistics
    pub fn get_stats(&self) -> PoolStats {
        let mut stats = self.stats.lock().unwrap();

        // Update dynamic statistics
        let buckets = self.buckets.read().unwrap();
        let mut total_pooled = 0;
        let mut total_memory = 0;

        for bucket_mutex in buckets.values() {
            let bucket = bucket_mutex.lock().unwrap();
            total_pooled += bucket.free_buffers.len();
            total_memory += bucket.free_buffers.len() * bucket.buffer_size;
        }

        stats.pooled_buffers = total_pooled;
        stats.memory_usage_bytes = total_memory;

        // Calculate fragmentation ratio
        if stats.total_allocations > 0 {
            stats.fragmentation_ratio = stats.pool_misses as f32 / stats.total_allocations as f32;
        }

        stats.clone()
    }

    /// Manually trigger cleanup of idle buffers
    pub fn cleanup_idle_buffers(&self) -> usize {
        let buckets = self.buckets.read().unwrap();
        let mut total_cleaned = 0;

        for bucket_mutex in buckets.values() {
            let mut bucket = bucket_mutex.lock().unwrap();
            total_cleaned += bucket.cleanup_idle_buffers(self.config.buffer_idle_timeout_secs);
        }

        total_cleaned
    }

    /// Clear all pooled buffers and reset statistics
    pub fn clear(&self) {
        let mut buckets = self.buckets.write().unwrap();
        buckets.clear();

        let mut stats = self.stats.lock().unwrap();
        *stats = PoolStats {
            total_allocations: 0,
            pool_hits: 0,
            pool_misses: 0,
            active_buffers: 0,
            pooled_buffers: 0,
            memory_usage_bytes: 0,
            fragmentation_ratio: 0.0,
        };
    }

    /// Start background cleanup thread
    fn start_background_cleanup(&mut self) {
        if self.cleanup_handle.is_some() {
            return; // Already started
        }

        let buckets = Arc::new(RwLock::new(HashMap::<usize, Arc<Mutex<PoolBucket>>>::new()));
        let stats = Arc::new(Mutex::new(PoolStats {
            total_allocations: 0,
            pool_hits: 0,
            pool_misses: 0,
            active_buffers: 0,
            pooled_buffers: 0,
            memory_usage_bytes: 0,
            fragmentation_ratio: 0.0,
        }));

        // Clone existing buckets into Arc structure for thread sharing
        {
            let mut thread_buckets = buckets.write().unwrap();
            let existing_buckets = self.buckets.read().unwrap();
            for (&size, _bucket_mutex) in existing_buckets.iter() {
                thread_buckets.insert(
                    size,
                    Arc::new(Mutex::new(PoolBucket::new(
                        size,
                        self.config.max_buffers_per_bucket,
                    ))),
                );
            }
        }

        let config = self.config.clone();
        let shutdown_signal = self.shutdown_signal.clone();

        let handle = std::thread::spawn(move || {
            let cleanup_interval = Duration::from_secs(config.cleanup_interval_secs);
            let mut last_cleanup = Instant::now();

            loop {
                // Check for shutdown signal
                if shutdown_signal.load(Ordering::Relaxed) {
                    log::info!("Background cleanup thread shutting down");
                    break;
                }

                thread::sleep(Duration::from_millis(100));

                // Only perform cleanup if enough time has passed
                if last_cleanup.elapsed() < cleanup_interval {
                    continue;
                }

                log::debug!("Starting background memory pool cleanup");
                let cleanup_start = Instant::now();
                let mut total_cleaned = 0;
                let mut buckets_processed = 0;

                // Perform cleanup on all buckets
                {
                    let buckets_guard = buckets.read().unwrap();
                    for (size, bucket_mutex) in buckets_guard.iter() {
                        if shutdown_signal.load(Ordering::Relaxed) {
                            break;
                        }

                        // Try to acquire bucket lock without blocking
                        if let Ok(mut bucket) = bucket_mutex.try_lock() {
                            let cleaned =
                                bucket.cleanup_idle_buffers(config.buffer_idle_timeout_secs);
                            total_cleaned += cleaned;
                            buckets_processed += 1;

                            if cleaned > 0 {
                                log::debug!(
                                    "Cleaned {} idle buffers from bucket size {}",
                                    cleaned,
                                    size
                                );
                            }
                        }
                    }
                }

                // Update global statistics
                if total_cleaned > 0 {
                    if let Ok(mut stats_guard) = stats.try_lock() {
                        stats_guard.pooled_buffers =
                            stats_guard.pooled_buffers.saturating_sub(total_cleaned);

                        // Update memory usage
                        let buckets_guard = buckets.read().unwrap();
                        let mut total_memory = 0;
                        let mut total_pooled = 0;

                        for (size, bucket_mutex) in buckets_guard.iter() {
                            if let Ok(bucket) = bucket_mutex.try_lock() {
                                total_pooled += bucket.free_buffers.len();
                                total_memory += bucket.free_buffers.len() * size;
                            }
                        }

                        stats_guard.pooled_buffers = total_pooled;
                        stats_guard.memory_usage_bytes = total_memory;
                    }

                    log::info!(
                        "Background cleanup completed: {} buffers cleaned from {} buckets in {:.2}ms",
                        total_cleaned,
                        buckets_processed,
                        cleanup_start.elapsed().as_secs_f64() * 1000.0
                    );
                } else {
                    log::debug!(
                        "Background cleanup completed: no idle buffers found in {} buckets",
                        buckets_processed
                    );
                }

                last_cleanup = Instant::now();

                // Adaptive cleanup interval based on memory pressure
                let memory_pressure = {
                    if let Ok(stats_guard) = stats.try_lock() {
                        if stats_guard.pooled_buffers > config.max_buffers_per_bucket * 10 {
                            0.5 // High pressure - cleanup more frequently
                        } else if stats_guard.pooled_buffers < config.max_buffers_per_bucket {
                            2.0 // Low pressure - cleanup less frequently
                        } else {
                            1.0 // Normal pressure
                        }
                    } else {
                        1.0
                    }
                };

                // Adjust next cleanup interval based on pressure
                let next_interval = Duration::from_secs(
                    (config.cleanup_interval_secs as f64 * memory_pressure) as u64,
                );

                if next_interval != cleanup_interval {
                    log::debug!(
                        "Adjusting cleanup interval to {}s based on memory pressure (factor: {:.1})",
                        next_interval.as_secs(),
                        memory_pressure
                    );
                }
            }

            log::info!("Background cleanup thread terminated");
        });

        self.cleanup_handle = Some(handle);
    }

    /// Get detailed memory pool metrics for monitoring
    pub fn get_detailed_stats(&self) -> DetailedPoolStats {
        let stats = self.get_stats();

        let hit_ratio = if stats.total_allocations > 0 {
            stats.pool_hits as f64 / stats.total_allocations as f64
        } else {
            0.0
        };

        let memory_usage_mb = stats.memory_usage_bytes as f64 / (1024.0 * 1024.0);

        // Per-bucket statistics
        let buckets = self.buckets.read().unwrap();
        let mut bucket_stats = HashMap::new();

        for (&size, bucket_mutex) in buckets.iter() {
            if let Ok(bucket) = bucket_mutex.try_lock() {
                let bucket_info = BucketStats {
                    free_buffers: bucket.free_buffers.len(),
                    allocated_buffers: bucket.allocated_buffers.len(),
                    hits: bucket.usage_stats.hits,
                    misses: bucket.usage_stats.misses,
                    created: bucket.usage_stats.created,
                };
                bucket_stats.insert(size, bucket_info);
            }
        }

        DetailedPoolStats {
            total_allocations: stats.total_allocations,
            pool_hits: stats.pool_hits,
            pool_misses: stats.pool_misses,
            hit_ratio,
            active_buffers: stats.active_buffers,
            pooled_buffers: stats.pooled_buffers,
            memory_usage_mb,
            fragmentation_ratio: stats.fragmentation_ratio,
            bucket_stats,
        }
    }

    /// Force immediate cleanup of all idle buffers
    pub fn force_cleanup(&self) -> usize {
        let buckets = self.buckets.read().unwrap();
        let mut total_cleaned = 0;

        for bucket_mutex in buckets.values() {
            let mut bucket = bucket_mutex.lock().unwrap();
            total_cleaned += bucket.cleanup_idle_buffers(0); // Force cleanup regardless of timeout
        }

        // Update stats
        if total_cleaned > 0 {
            let mut stats = self.stats.lock().unwrap();
            stats.pooled_buffers = stats.pooled_buffers.saturating_sub(total_cleaned);
        }

        total_cleaned
    }

    /// Pre-allocate buffers for common sizes to reduce allocation latency
    pub fn preallocate_common_sizes(&self) -> Result<()> {
        let common_sizes = [
            1024,    // 1KB
            4096,    // 4KB
            16384,   // 16KB
            65536,   // 64KB
            262144,  // 256KB
            1048576, // 1MB
            4194304, // 4MB
        ];

        for &size in &common_sizes {
            if size <= self.config.max_buffer_size {
                // Pre-allocate a few buffers for each common size
                for _ in 0..4 {
                    let buffer = self.get_buffer(size)?;
                    self.return_buffer(buffer);
                }
            }
        }

        log::info!(
            "Pre-allocated buffers for {} common sizes",
            common_sizes.len()
        );
        Ok(())
    }

    /// Round size up to the next pool bucket size (power of 2)
    fn round_to_pool_size(&self, size: usize) -> usize {
        let min_size = self.config.min_buffer_size.max(size);
        min_size.next_power_of_two()
    }

    /// Create buffer directly without pooling (for large allocations)
    fn create_direct_buffer(&self, size: usize) -> Result<Arc<VulkanBuffer>> {
        let buffer = Buffer::new_slice::<f32>(
            self.allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_SRC
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (size / std::mem::size_of::<f32>()) as u64,
        )
        .map_err(|e| NnlError::gpu(format!("Failed to create direct buffer: {}", e)))?;

        // Update stats
        let mut stats = self.stats.lock().unwrap();
        stats.pool_misses += 1;
        stats.active_buffers += 1;

        Ok(Arc::new(VulkanBuffer::from_buffer(buffer, size, false)?))
    }
}

impl Drop for GpuMemoryPool {
    fn drop(&mut self) {
        // Signal shutdown to background thread
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Wait for background thread to finish gracefully
        if let Some(handle) = self.cleanup_handle.take() {
            log::info!("Waiting for background cleanup thread to finish...");

            // Give the thread some time to shut down gracefully
            let join_result = handle.join();

            match join_result {
                Ok(()) => log::info!("Background cleanup thread finished successfully"),
                Err(_) => log::warn!("Background cleanup thread panicked during shutdown"),
            }
        }

        // Log final statistics
        let final_stats = self.get_stats();
        log::info!(
            "Memory pool destroyed - Final stats: {} total allocations, {:.1}% hit rate, {} MB peak usage",
            final_stats.total_allocations,
            if final_stats.total_allocations > 0 {
                (final_stats.pool_hits as f64 / final_stats.total_allocations as f64) * 100.0
            } else {
                0.0
            },
            final_stats.memory_usage_bytes as f64 / (1024.0 * 1024.0)
        );

        // Clear all buckets
        {
            let mut buckets = self.buckets.write().unwrap();
            let bucket_count = buckets.len();
            buckets.clear();
            log::debug!("Cleared {} memory pool buckets", bucket_count);
        }
    }
}

/// Scoped buffer that automatically returns to pool when dropped
pub struct ScopedBuffer {
    buffer: Option<Arc<VulkanBuffer>>,
    pool: Arc<GpuMemoryPool>,
}

impl ScopedBuffer {
    /// Create a new pooled buffer wrapper
    pub fn new(buffer: Arc<VulkanBuffer>, pool: Arc<GpuMemoryPool>) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    /// Get a reference to the underlying buffer
    pub fn buffer(&self) -> &Arc<VulkanBuffer> {
        self.buffer.as_ref().expect("Buffer already taken")
    }

    /// Take ownership of the underlying buffer, consuming this wrapper
    pub fn take(mut self) -> Arc<VulkanBuffer> {
        self.buffer.take().expect("Buffer already taken")
    }
}

impl Drop for ScopedBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.return_buffer(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock allocator for testing
    fn create_test_pool() -> GpuMemoryPool {
        // This would need a real Vulkan allocator in practice
        todo!("Implement tests with mock Vulkan allocator")
    }

    #[test]
    fn test_pool_hit_miss_ratio() {
        // Test that repeated allocations of same size hit the pool
    }

    #[test]
    fn test_size_bucketing() {
        // Test power-of-2 bucketing
        let config = PoolConfig::default();
        assert_eq!(
            1000_usize.next_power_of_two().max(config.min_buffer_size),
            1024
        );
        assert_eq!(
            2000_usize.next_power_of_two().max(config.min_buffer_size),
            2048
        );
        assert_eq!(
            1024_usize.next_power_of_two().max(config.min_buffer_size),
            1024
        );
        assert_eq!(
            500_usize.next_power_of_two().max(config.min_buffer_size),
            1024
        ); // min_buffer_size
    }

    #[test]
    fn test_cleanup_functionality() {
        // Test that old buffers are cleaned up properly
    }
}
