# GPU Initialization Fix - Testing Guide

## Problem Fixed
The training script was crashing with the following error when TensorFlow attempted to initialize the GPU:
```
Failed to initialize GPU device #0: shared object symbol not found
Floating point exception (core dumped)
```

## Solution Implemented
Added comprehensive GPU error handling with automatic CPU fallback:

1. **Early GPU Testing**: Added a GPU initialization test in `configure_gpu()` that creates a simple tensor operation on the GPU before model building
2. **Graceful Degradation**: If GPU initialization fails, the script automatically disables GPU and falls back to CPU execution
3. **Model Building Protection**: Wrapped model building in try-catch blocks with retry logic on CPU if GPU fails
4. **Informative Messages**: Provides clear error messages explaining potential causes

## Testing Instructions

### On EC2 with GPU Issues
Run the training script as before:
```bash
python3 src/train.py --mode multimodal
```

**Expected behavior with this fix:**
- GPU will be detected
- GPU initialization test will fail (catching the error early)
- Script will automatically disable GPU and force CPU execution
- Training will continue successfully on CPU
- No crash or "Floating point exception"

### Expected Output with GPU Failure
```
======================================================================
GPU Configuration
======================================================================
Found 1 GPU(s):
  GPU 0: /physical_device:GPU:0

✓ GPU memory growth enabled
  (GPU memory will be allocated as needed)
✓ Mixed precision training enabled (float16)
======================================================================

======================================================================
⚠ GPU Initialization Failed
======================================================================
Error: [error details]
This may be caused by:
  - Missing or incompatible CUDA libraries
  - Driver version mismatch
  - Corrupted TensorFlow installation

Forcing CPU execution for stability...
======================================================================

✓ Successfully disabled GPU, using CPU

1. Preparing multimodal data...
   [continues with training on CPU]
```

### On Systems with Working GPU
If GPU is working properly:
```
======================================================================
GPU Configuration
======================================================================
Found 1 GPU(s):
  GPU 0: /physical_device:GPU:0

✓ GPU memory growth enabled
  (GPU memory will be allocated as needed)
✓ Mixed precision training enabled (float16)
======================================================================

✓ GPU initialization test passed

1. Preparing multimodal data...
   [continues with training on GPU]
```

### On CPU-Only Systems
If no GPU is available:
```
======================================================================
No GPU detected - using CPU
======================================================================

1. Preparing multimodal data...
   [continues with training on CPU]
```

## Manual Verification Steps

1. **Test GPU Fallback (simulated)**:
   ```bash
   # Force CPU-only execution to test the fallback path
   CUDA_VISIBLE_DEVICES="" python3 src/train.py --mode multimodal
   ```

2. **Test Normal GPU Path** (if GPU is working):
   ```bash
   python3 src/train.py --mode multimodal
   ```

3. **Test Tabular Mode**:
   ```bash
   python3 src/train.py --mode tabular
   ```

## Files Modified
- `src/train.py`: Added GPU error handling and CPU fallback logic

## Key Changes in Code

### Enhanced `configure_gpu()` Function
- Returns `bool` indicating GPU availability
- Tests GPU with actual tensor operation before model building
- Catches GPU failures early with helpful error messages
- Automatically disables GPU and forces CPU on failure

### Protected Model Building
Both `train_multimodal_model()` and `train_tabular_model()` now:
- Wrap model building in try-catch blocks
- Retry on CPU if GPU fails during model initialization
- Provide clear status messages

## Troubleshooting

### If training still crashes:
1. Check TensorFlow installation: `python3 -c "import tensorflow as tf; print(tf.__version__)"`
2. Check CUDA libraries: `ldconfig -p | grep cuda`
3. Check GPU drivers: `nvidia-smi`
4. Try forcing CPU-only: `CUDA_VISIBLE_DEVICES="" python3 src/train.py --mode multimodal`

### If you want to investigate the GPU issue further:
The fix allows training to continue, but you may want to resolve the underlying GPU issue:
- Update NVIDIA drivers
- Reinstall TensorFlow with proper CUDA compatibility
- Check for CUDA library version mismatches

## Performance Impact
- **With working GPU**: No performance impact
- **With failed GPU (now using CPU)**: Training will be slower but functional
- The fix ensures training can complete even with GPU hardware/driver issues
