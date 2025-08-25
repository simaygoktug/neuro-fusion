# neurofusion/__init__.py
"""
NeuroFusion Controller: Hybrid Brain-AI Control System
=======================================================

A revolutionary real-time control system that combines human neural signals 
with artificial intelligence for seamless human-machine collaboration.

Key Components:
- Core System: Signal processing, prediction, and control
- Edge Deployment: Optimization and validation tools  
- Hardware Integration: Real sensor and robot interfaces

Author: Goktug Can Simay
University of Essex - MSc Intelligent Systems and Robotics
"""

__version__ = "1.0.0"
__author__ = "Goktug Can Simay"
__email__ = "simaygoktug@gmail.com"

# Import main classes for easy access
from .core_system import (
    EEGEMGPreprocessor,
    MotorIntentionPredictor, 
    BayesianUncertaintyEstimator,
    ModelPredictiveController,
    TransferLearningAdapter,
    NeuroFusionController
)

from .deployment_and_testing import (
    SystemConfig,
    EdgeOptimizer,
    RealTimeDataSimulator,
    SystemValidator,
    ProductionDeployment
)

from .hardware_integration import (
    HardwareConfig,
    OpenBCIInterface,
    DelsysEMGInterface, 
    RobotControlInterface,
    RealTimeNeuroFusionSystem,
    DissertationReportGenerator
)

__all__ = [
    # Core System
    'EEGEMGPreprocessor',
    'MotorIntentionPredictor',
    'BayesianUncertaintyEstimator', 
    'ModelPredictiveController',
    'TransferLearningAdapter',
    'NeuroFusionController',
    
    # Deployment & Testing
    'SystemConfig',
    'EdgeOptimizer',
    'RealTimeDataSimulator',
    'SystemValidator', 
    'ProductionDeployment',
    
    # Hardware Integration
    'HardwareConfig',
    'OpenBCIInterface',
    'DelsysEMGInterface',
    'RobotControlInterface', 
    'RealTimeNeuroFusionSystem',
    'DissertationReportGenerator',
]

# Convenience functions
def create_default_system():
    """Create a NeuroFusion system with default configuration"""
    return NeuroFusionController()

def create_hardware_system(hardware_config=None):
    """Create a hardware-integrated NeuroFusion system"""
    if hardware_config is None:
        hardware_config = HardwareConfig()
    return RealTimeNeuroFusionSystem(hardware_config)

def run_system_demo():
    """Run a complete system demonstration"""
    print("NeuroFusion Controller Demo")
    print("=" * 40)
    
    # Create and test core system
    controller = create_default_system()
    print("✓ Core system initialized")
    
    # Run basic functionality test
    import numpy as np
    target = np.array([1.0, 0.5, 0.8, 0.1, 0.2, 0.0])
    controller.set_target(target)
    print("✓ Target set successfully")
    
    print("Demo completed successfully!")
    return controller

# Version check function
def check_dependencies():
    """Check if all required dependencies are installed"""
    import importlib
    required_packages = [
        'numpy', 'torch', 'scipy', 'matplotlib', 
        'seaborn', 'sklearn', 'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("All dependencies satisfied!")
        return True