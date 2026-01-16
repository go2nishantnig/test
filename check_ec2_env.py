#!/usr/bin/env python3
"""
Check EC2 environment for fraud detection model training
Verifies GPU, directories, dependencies, and configuration
"""
import os
import sys

def check_python():
    """Check Python version"""
    print("\n" + "="*70)
    print("Python Environment")
    print("="*70)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 8:
        print("✓ Python version is compatible (3.8+)")
        return True
    else:
        print("✗ Python version should be 3.8 or higher")
        return False


def check_tensorflow():
    """Check TensorFlow and GPU"""
    print("\n" + "="*70)
    print("TensorFlow and GPU")
    print("="*70)
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Check GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            
            # Try to get GPU details
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    print("\nGPU Details:")
                    for line in result.stdout.strip().split('\n'):
                        print(f"  {line}")
            except:
                pass
            
            return True
        else:
            print("✗ No GPU detected")
            print("For EC2 G5 XLarge, ensure NVIDIA drivers are installed:")
            print("  sudo yum install -y nvidia-driver-latest-dkms")
            return False
            
    except ImportError:
        print("✗ TensorFlow not installed")
        print("Install with: pip install tensorflow>=2.13.0")
        return False
    except Exception as e:
        print(f"✗ Error checking TensorFlow: {e}")
        return False


def check_dependencies():
    """Check other required dependencies"""
    print("\n" + "="*70)
    print("Required Dependencies")
    print("="*70)
    
    dependencies = {
        'numpy': 'Data processing',
        'pandas': 'Data manipulation',
        'sklearn': 'Machine learning utilities',
        'matplotlib': 'Visualization',
        'seaborn': 'Statistical visualization',
        'PIL': 'Image processing',
        'jupyter': 'Jupyter notebooks',
    }
    
    all_installed = True
    for module, description in dependencies.items():
        try:
            if module == 'sklearn':
                import sklearn
            elif module == 'PIL':
                from PIL import Image
            else:
                __import__(module)
            print(f"✓ {module:15} - {description}")
        except ImportError:
            print(f"✗ {module:15} - {description} (NOT INSTALLED)")
            all_installed = False
    
    return all_installed


def check_directories():
    """Check EC2 directories"""
    print("\n" + "="*70)
    print("EC2 Directories")
    print("="*70)
    
    directories = {
        'QR Data': '/home/ec2-user/qrdata',
        'CSV Data': '/home/ec2-user/csv-data',
        'Models': '/home/ec2-user/model',
        'Logs': '/home/ec2-user/model/logs',
    }
    
    all_exist = True
    for name, path in directories.items():
        exists = os.path.exists(path)
        writable = os.access(path, os.W_OK) if exists else False
        
        if exists and writable:
            print(f"✓ {name:12} - {path} (writable)")
        elif exists:
            print(f"⚠ {name:12} - {path} (not writable)")
            all_exist = False
        else:
            print(f"✗ {name:12} - {path} (does not exist)")
            all_exist = False
    
    if not all_exist:
        print("\nTo create directories, run:")
        print("  bash setup_ec2_jupyter.sh")
    
    return all_exist


def check_virtual_env():
    """Check if running in virtual environment"""
    print("\n" + "="*70)
    print("Virtual Environment")
    print("="*70)
    
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print(f"✓ Running in virtual environment")
        print(f"  Location: {sys.prefix}")
        return True
    else:
        print("⚠ Not running in virtual environment")
        print("  Recommended to use virtual environment")
        print("  Activate with: source /home/ec2-user/fraud_detection_env/bin/activate")
        return False


def check_config():
    """Check if EC2 config is accessible"""
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from config import ec2_config
        print("✓ EC2 configuration loaded")
        print(f"  QR Data Path: {ec2_config.EC2_QRDATA_DIR}")
        print(f"  CSV Data Path: {ec2_config.EC2_CSV_DATA_DIR}")
        print(f"  Model Path: {ec2_config.EC2_MODEL_DIR}")
        print(f"  Mixed Precision: {ec2_config.GPU_CONFIG.get('mixed_precision', False)}")
        print(f"  Memory Growth: {ec2_config.GPU_CONFIG.get('memory_growth', False)}")
        return True
    except ImportError as e:
        print(f"✗ Cannot load EC2 configuration: {e}")
        print("  Make sure you're in the repository directory")
        return False
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False


def main():
    """Run all checks"""
    print("="*70)
    print("EC2 Environment Check for Multimodal Fraud Detection")
    print("="*70)
    
    checks = {
        'Python': check_python(),
        'TensorFlow & GPU': check_tensorflow(),
        'Dependencies': check_dependencies(),
        'Directories': check_directories(),
        'Virtual Environment': check_virtual_env(),
        'Configuration': check_config(),
    }
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    for name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {name}")
    
    all_passed = all(checks.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All checks passed! Your environment is ready.")
        print("\nNext steps:")
        print("  1. Place your data in /home/ec2-user/csv-data and /home/ec2-user/qrdata")
        print("  2. Open Jupyter notebook: ec2_g5_xlarge_pipeline.ipynb")
        print("  3. Or run: python ec2_quick_start.py")
    else:
        print("⚠ Some checks failed. Please fix the issues above.")
        print("\nRun setup script to fix common issues:")
        print("  bash setup_ec2_jupyter.sh")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
