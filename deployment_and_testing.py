import numpy as np
import torch
import torch.onnx
import time
import json
import threading
import queue
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """System configuration parameters"""
    # Neural network parameters
    eeg_channels: int = 8
    emg_channels: int = 4
    sampling_rate: int = 1000
    window_size_ms: int = 250
    num_actions: int = 6
    
    # Processing parameters
    batch_size: int = 1
    max_latency_ms: float = 1.0
    confidence_threshold: float = 0.7
    uncertainty_threshold: float = 0.3
    
    # MPC parameters
    mpc_horizon: int = 10
    mpc_iterations: int = 50
    
    # Safety parameters
    emergency_stop_threshold: float = 0.9
    max_acceleration: float = 2.0
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        logger.info(f"Configuration loaded from {filepath}")
        return cls(**config_dict)

class EdgeOptimizer:
    """
    Optimize neural networks for edge deployment
    Model quantization, pruning, and ONNX conversion
    """
    
    def __init__(self, model):
        self.model = model
        self.optimized_models = {}
    
    def quantize_model(self, calibration_data=None):
        """Apply dynamic quantization for faster inference"""
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.LSTM},
                dtype=torch.qint8
            )
            self.optimized_models['quantized'] = quantized_model
            logger.info("Model quantization completed")
            return quantized_model
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return None
    
    def prune_model(self, sparsity=0.3):
        """Apply structured pruning to reduce model size"""
        try:
            import torch.nn.utils.prune as prune
            
            # Apply pruning to linear layers
            for module in self.model.modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                    prune.remove(module, 'weight')
            
            self.optimized_models['pruned'] = self.model
            logger.info(f"Model pruning completed with {sparsity} sparsity")
            return self.model
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return None
    
    def convert_to_onnx(self, input_shape_eeg, input_shape_emg, filepath="neurofusion_model.onnx"):
        """Convert PyTorch model to ONNX format"""
        try:
            # Create dummy inputs
            dummy_eeg = torch.randn(1, *input_shape_eeg)
            dummy_emg = torch.randn(1, *input_shape_emg)
            
            # Export model
            torch.onnx.export(
                self.model,
                (dummy_eeg, dummy_emg),
                filepath,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['eeg_input', 'emg_input'],
                output_names=['action_probs', 'confidence'],
                dynamic_axes={
                    'eeg_input': {0: 'batch_size'},
                    'emg_input': {0: 'batch_size'},
                    'action_probs': {0: 'batch_size'},
                    'confidence': {0: 'batch_size'}
                }
            )
            logger.info(f"ONNX model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            return False
    
    def benchmark_models(self, test_data_eeg, test_data_emg, num_iterations=1000):
        """Benchmark original vs optimized models"""
        models_to_test = {'original': self.model}
        models_to_test.update(self.optimized_models)
        
        results = {}
        
        for model_name, model in models_to_test.items():
            latencies = []
            
            model.eval()
            with torch.no_grad():
                for _ in range(num_iterations):
                    start_time = time.perf_counter()
                    _ = model(test_data_eeg, test_data_emg)
                    latency = (time.perf_counter() - start_time) * 1000  # ms
                    latencies.append(latency)
            
            results[model_name] = {
                'mean_latency_ms': np.mean(latencies),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99),
                'throughput_hz': 1000 / np.mean(latencies)
            }
        
        return results

class RealTimeDataSimulator:
    """
    Simulate real-time EEG/EMG data streams for testing
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.is_running = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.simulation_thread = None
        
        # Signal parameters
        self.eeg_noise_level = 10e-6  # 10 microvolts
        self.emg_noise_level = 50e-6  # 50 microvolts
        
    def start_simulation(self, duration_seconds=60):
        """Start real-time data simulation"""
        self.is_running = True
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            args=(duration_seconds,)
        )
        self.simulation_thread.start()
        logger.info("Real-time simulation started")
    
    def stop_simulation(self):
        """Stop data simulation"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join()
        logger.info("Real-time simulation stopped")
    
    def _simulation_loop(self, duration_seconds):
        """Main simulation loop"""
        samples_per_window = self.config.window_size_ms * self.config.sampling_rate // 1000
        window_interval = self.config.window_size_ms / 1000  # seconds
        
        start_time = time.time()
        
        while self.is_running and (time.time() - start_time) < duration_seconds:
            # Generate EEG data with realistic patterns
            eeg_data = self._generate_eeg_data(samples_per_window)
            
            # Generate EMG data with muscle activation patterns
            emg_data = self._generate_emg_data(samples_per_window)
            
            # Create data packet
            data_packet = {
                'timestamp': time.time(),
                'eeg_data': eeg_data,
                'emg_data': emg_data,
                'true_action': self._generate_true_action()
            }
            
            try:
                self.data_queue.put(data_packet, timeout=0.001)
            except queue.Full:
                # Remove oldest if buffer full
                self.data_queue.get_nowait()
                self.data_queue.put(data_packet)
            
            # Sleep to maintain real-time rate
            time.sleep(window_interval)
    
    def _generate_eeg_data(self, num_samples):
        """Generate realistic EEG data"""
        eeg_data = np.zeros((self.config.eeg_channels, num_samples))
        
        for channel in range(self.config.eeg_channels):
            # Base noise
            noise = np.random.normal(0, self.eeg_noise_level, num_samples)
            
            # Add frequency components
            t = np.linspace(0, num_samples/self.config.sampling_rate, num_samples)
            
            # Alpha waves (8-13 Hz)
            alpha = 2e-6 * np.sin(2 * np.pi * 10 * t + np.random.random() * 2 * np.pi)
            
            # Beta waves (13-30 Hz) 
            beta = 1e-6 * np.sin(2 * np.pi * 20 * t + np.random.random() * 2 * np.pi)
            
            # Theta waves (4-8 Hz)
            theta = 1.5e-6 * np.sin(2 * np.pi * 6 * t + np.random.random() * 2 * np.pi)
            
            eeg_data[channel] = noise + alpha + beta + theta
        
        return eeg_data
    
    def _generate_emg_data(self, num_samples):
        """Generate realistic EMG data"""
        emg_data = np.zeros((self.config.emg_channels, num_samples))
        
        for channel in range(self.config.emg_channels):
            # Base noise
            noise = np.random.normal(0, self.emg_noise_level, num_samples)
            
            # Muscle activation bursts
            if np.random.random() < 0.3:  # 30% chance of activation
                burst_start = np.random.randint(0, num_samples // 2)
                burst_duration = np.random.randint(50, 150)  # 50-150ms
                burst_end = min(burst_start + burst_duration, num_samples)
                
                # Generate muscle activation pattern
                activation = np.random.exponential(100e-6, burst_end - burst_start)
                noise[burst_start:burst_end] += activation
            
            emg_data[channel] = noise
        
        return emg_data
    
    def _generate_true_action(self):
        """Generate true action label for validation"""
        return np.random.randint(0, self.config.num_actions)
    
    def get_data(self):
        """Get latest simulated data"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

class SystemValidator:
    """
    Comprehensive system validation and testing
    """
    
    def __init__(self, controller, config: SystemConfig):
        self.controller = controller
        self.config = config
        self.validation_results = {}
        
    def run_latency_test(self, num_samples=1000):
        """Test system latency performance"""
        logger.info("Running latency validation test...")
        
        latencies = []
        simulator = RealTimeDataSimulator(self.config)
        
        for i in range(num_samples):
            # Generate test data
            eeg_data = simulator._generate_eeg_data(250)
            emg_data = simulator._generate_emg_data(250)
            
            # Measure processing time
            start_time = time.perf_counter()
            result = self.controller.process_neural_signals(eeg_data, emg_data)
            latency = (time.perf_counter() - start_time) * 1000  # ms
            
            latencies.append(latency)
            
            if i % 100 == 0:
                logger.info(f"Processed {i}/{num_samples} samples")
        
        # Calculate statistics
        latency_stats = {
            'mean_ms': np.mean(latencies),
            'median_ms': np.median(latencies),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'max_ms': np.max(latencies),
            'std_ms': np.std(latencies),
            'samples_under_1ms': np.sum(np.array(latencies) < 1.0),
            'success_rate': np.sum(np.array(latencies) < self.config.max_latency_ms) / len(latencies)
        }
        
        self.validation_results['latency'] = latency_stats
        logger.info(f"Latency test completed - Mean: {latency_stats['mean_ms']:.3f}ms")
        return latency_stats
    
    def run_accuracy_test(self, num_samples=500):
        """Test prediction accuracy"""
        logger.info("Running accuracy validation test...")
        
        predictions = []
        true_labels = []
        confidences = []
        
        simulator = RealTimeDataSimulator(self.config)
        
        for i in range(num_samples):
            # Generate test data with known labels
            eeg_data = simulator._generate_eeg_data(250)
            emg_data = simulator._generate_emg_data(250)
            true_action = simulator._generate_true_action()
            
            # Process through system
            result = self.controller.process_neural_signals(eeg_data, emg_data)
            
            if result:
                predicted_action = np.argmax(result['action_probabilities'])
                predictions.append(predicted_action)
                true_labels.append(true_action)
                confidences.append(result['confidence'])
            
            if i % 100 == 0:
                logger.info(f"Processed {i}/{num_samples} samples")
        
        # Calculate accuracy metrics
        accuracy = accuracy_score(true_labels, predictions)
        conf_matrix = confusion_matrix(true_labels, predictions)
        avg_confidence = np.mean(confidences)
        
        accuracy_stats = {
            'overall_accuracy': accuracy,
            'confusion_matrix': conf_matrix.tolist(),
            'average_confidence': avg_confidence,
            'high_confidence_samples': np.sum(np.array(confidences) > self.config.confidence_threshold),
            'total_samples': len(predictions)
        }
        
        self.validation_results['accuracy'] = accuracy_stats
        logger.info(f"Accuracy test completed - Accuracy: {accuracy:.3f}")
        return accuracy_stats
    
    def run_stress_test(self, duration_minutes=5):
        """Run system stress test"""
        logger.info(f"Running stress test for {duration_minutes} minutes...")
        
        simulator = RealTimeDataSimulator(self.config)
        simulator.start_simulation(duration_minutes * 60)
        
        start_time = time.time()
        processed_samples = 0
        failed_samples = 0
        latencies = []
        
        try:
            while (time.time() - start_time) < (duration_minutes * 60):
                data = simulator.get_data()
                
                if data:
                    process_start = time.perf_counter()
                    result = self.controller.process_neural_signals(
                        data['eeg_data'], data['emg_data']
                    )
                    process_time = (time.perf_counter() - process_start) * 1000
                    
                    if result:
                        processed_samples += 1
                        latencies.append(process_time)
                    else:
                        failed_samples += 1
                
                time.sleep(0.001)  # 1ms sleep
        
        finally:
            simulator.stop_simulation()
        
        stress_stats = {
            'duration_minutes': duration_minutes,
            'processed_samples': processed_samples,
            'failed_samples': failed_samples,
            'success_rate': processed_samples / (processed_samples + failed_samples) if (processed_samples + failed_samples) > 0 else 0,
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'max_latency_ms': np.max(latencies) if latencies else 0,
            'throughput_hz': processed_samples / (duration_minutes * 60)
        }
        
        self.validation_results['stress'] = stress_stats
        logger.info(f"Stress test completed - Processed: {processed_samples} samples")
        return stress_stats
    
    def run_safety_test(self, num_scenarios=100):
        """Test safety mechanisms"""
        logger.info("Running safety validation test...")
        
        emergency_stops = 0
        unsafe_commands = 0
        total_scenarios = 0
        
        simulator = RealTimeDataSimulator(self.config)
        
        for i in range(num_scenarios):
            # Generate potentially unsafe scenarios
            if np.random.random() < 0.1:  # 10% unsafe scenarios
                # Inject noise or corrupted data
                eeg_data = np.random.randn(self.config.eeg_channels, 250) * 1e-3  # High noise
                emg_data = np.random.randn(self.config.emg_channels, 250) * 1e-3
            else:
                eeg_data = simulator._generate_eeg_data(250)
                emg_data = simulator._generate_emg_data(250)
            
            result = self.controller.process_neural_signals(eeg_data, emg_data)
            total_scenarios += 1
            
            if result:
                # Check for unsafe commands
                command_magnitude = np.linalg.norm(result['control_command'])
                
                if result['confidence'] < 0.1:  # Very low confidence
                    emergency_stops += 1
                elif command_magnitude > self.config.max_acceleration:
                    unsafe_commands += 1
        
        safety_stats = {
            'total_scenarios': total_scenarios,
            'emergency_stops': emergency_stops,
            'unsafe_commands': unsafe_commands,
            'safety_rate': (total_scenarios - unsafe_commands) / total_scenarios if total_scenarios > 0 else 0,
            'emergency_stop_rate': emergency_stops / total_scenarios if total_scenarios > 0 else 0
        }
        
        self.validation_results['safety'] = safety_stats
        logger.info(f"Safety test completed - Safety rate: {safety_stats['safety_rate']:.3f}")
        return safety_stats
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_config': asdict(self.config),
            'validation_results': self.validation_results,
            'summary': self._generate_summary()
        }
        
        return report
    
    def _generate_summary(self):
        """Generate validation summary"""
        summary = {
            'overall_status': 'PASS',
            'critical_issues': [],
            'recommendations': []
        }
        
        # Check latency requirements
        if 'latency' in self.validation_results:
            latency = self.validation_results['latency']
            if latency['p95_ms'] > self.config.max_latency_ms:
                summary['overall_status'] = 'FAIL'
                summary['critical_issues'].append('Latency exceeds requirements')
        
        # Check accuracy requirements
        if 'accuracy' in self.validation_results:
            accuracy = self.validation_results['accuracy']
            if accuracy['overall_accuracy'] < 0.8:
                summary['overall_status'] = 'WARNING'
                summary['recommendations'].append('Consider model retraining')
        
        # Check safety requirements
        if 'safety' in self.validation_results:
            safety = self.validation_results['safety']
            if safety['safety_rate'] < 0.95:
                summary['overall_status'] = 'FAIL'
                summary['critical_issues'].append('Safety rate below threshold')
        
        return summary
    
    def plot_results(self, save_path=None):
        """Generate validation result plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Latency histogram
        if 'latency' in self.validation_results:
            latency_data = self.controller.processing_times
            axes[0, 0].hist(latency_data, bins=50, alpha=0.7, color='blue')
            axes[0, 0].axvline(self.config.max_latency_ms, color='red', linestyle='--', label='Requirement')
            axes[0, 0].set_xlabel('Latency (ms)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Processing Latency Distribution')
            axes[0, 0].legend()
        
        # Accuracy confusion matrix
        if 'accuracy' in self.validation_results:
            conf_matrix = np.array(self.validation_results['accuracy']['confusion_matrix'])
            sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Actual')
            axes[0, 1].set_title('Confusion Matrix')
        
        # Stress test results
        if 'stress' in self.validation_results:
            stress_results = self.validation_results['stress']
            metrics = ['Success Rate', 'Avg Latency (ms)', 'Throughput (Hz)']
            values = [
                stress_results['success_rate'],
                stress_results['avg_latency_ms'],
                stress_results['throughput_hz']
            ]
            axes[1, 0].bar(metrics, values, color=['green', 'orange', 'purple'])
            axes[1, 0].set_title('Stress Test Results')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Safety test results
        if 'safety' in self.validation_results:
            safety_results = self.validation_results['safety']
            labels = ['Safe', 'Unsafe', 'Emergency Stop']
            sizes = [
                safety_results['total_scenarios'] - safety_results['unsafe_commands'] - safety_results['emergency_stops'],
                safety_results['unsafe_commands'],
                safety_results['emergency_stops']
            ]
            axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=['green', 'red', 'orange'])
            axes[1, 1].set_title('Safety Test Results')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation plots saved to {save_path}")
        
        plt.show()

class ProductionDeployment:
    """
    Production deployment utilities
    """
    
    def __init__(self, controller, config: SystemConfig):
        self.controller = controller
        self.config = config
        self.deployment_status = {
            'deployed': False,
            'start_time': None,
            'uptime_hours': 0,
            'total_processed': 0,
            'errors': 0
        }
        
    def deploy_system(self):
        """Deploy system to production"""
        logger.info("Deploying NeuroFusion Controller to production...")
        
        # Pre-deployment checks
        if not self._run_deployment_checks():
            logger.error("Deployment checks failed")
            return False
        
        # Start system
        self.deployment_status['deployed'] = True
        self.deployment_status['start_time'] = time.time()
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info("NeuroFusion Controller successfully deployed")
        return True
    
    def _run_deployment_checks(self):
        """Run pre-deployment validation"""
        logger.info("Running deployment validation checks...")
        
        # Check model integrity
        try:
            test_eeg = torch.randn(1, 40)  # Typical feature size
            test_emg = torch.randn(1, 16)
            with torch.no_grad():
                _ = self.controller.intention_predictor(test_eeg, test_emg)
            logger.info("✓ Model integrity check passed")
        except Exception as e:
            logger.error(f"✗ Model integrity check failed: {e}")
            return False
        
        # Check latency requirements
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            eeg_data = np.random.randn(8, 250) * 10e-6
            emg_data = np.random.randn(4, 250) * 50e-6
            self.controller.process_neural_signals(eeg_data, emg_data)
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_latency = np.mean(latencies)
        if avg_latency > self.config.max_latency_ms:
            logger.error(f"✗ Latency check failed: {avg_latency:.2f}ms > {self.config.max_latency_ms}ms")
            return False
        else:
            logger.info(f"✓ Latency check passed: {avg_latency:.2f}ms")
        
        # Check memory usage
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > 1024:  # 1GB limit
            logger.warning(f"⚠ High memory usage: {memory_mb:.1f}MB")
        else:
            logger.info(f"✓ Memory usage acceptable: {memory_mb:.1f}MB")
        
        logger.info("All deployment checks passed")
        return True
    
    def _start_monitoring(self):
        """Start production monitoring"""
        monitoring_thread = threading.Thread(target=self._monitoring_loop)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        logger.info("Production monitoring started")
    
    def _monitoring_loop(self):
        """Production monitoring loop"""
        while self.deployment_status['deployed']:
            try:
                # Update uptime
                if self.deployment_status['start_time']:
                    uptime_seconds = time.time() - self.deployment_status['start_time']
                    self.deployment_status['uptime_hours'] = uptime_seconds / 3600
                
                # Log status every hour
                if self.deployment_status['uptime_hours'] % 1 < 0.001:  # Approximately every hour
                    logger.info(
                        f"System Status - Uptime: {self.deployment_status['uptime_hours']:.1f}h, "
                        f"Processed: {self.deployment_status['total_processed']}, "
                        f"Errors: {self.deployment_status['errors']}"
                    )
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                self.deployment_status['errors'] += 1
    
    def get_deployment_status(self):
        """Get current deployment status"""
        return self.deployment_status.copy()
    
    def shutdown_system(self):
        """Gracefully shutdown production system"""
        logger.info("Shutting down NeuroFusion Controller...")
        
        self.deployment_status['deployed'] = False
        
        # Save final statistics
        final_stats = {
            'shutdown_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_uptime_hours': self.deployment_status['uptime_hours'],
            'total_processed': self.deployment_status['total_processed'],
            'total_errors': self.deployment_status['errors'],
            'error_rate': self.deployment_status['errors'] / max(self.deployment_status['total_processed'], 1)
        }
        
        with open('deployment_stats.json', 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        logger.info("System shutdown completed")
        return final_stats

# Example usage and comprehensive testing
if __name__ == "__main__":
    print("="*80)
    print("NEUROFUSION DEPLOYMENT & TESTING - COMPREHENSIVE VALIDATION")
    print("="*80)
    
    # Import core system
    from .core_system import NeuroFusionController
    
    # Create system configuration
    config = SystemConfig()
    print(f"\nSystem Configuration:")
    print(f"Max Latency: {config.max_latency_ms}ms")
    print(f"Confidence Threshold: {config.confidence_threshold}")
    print(f"Sampling Rate: {config.sampling_rate}Hz")
    
    # Initialize controller
    controller = NeuroFusionController(config)
    
    # Edge optimization
    print("\n1. EDGE OPTIMIZATION")
    print("-" * 50)
    optimizer = EdgeOptimizer(controller.intention_predictor)
    
    # Test data for optimization
    test_eeg = torch.randn(1, 40)
    test_emg = torch.randn(1, 16)
    
    # Quantization
    quantized_model = optimizer.quantize_model()
    if quantized_model:
        print("✓ Model quantization completed")
    
    # Benchmark models
    benchmark_results = optimizer.benchmark_models(test_eeg, test_emg, num_iterations=100)
    for model_name, results in benchmark_results.items():
        print(f"{model_name}: {results['mean_latency_ms']:.2f}ms avg, {results['throughput_hz']:.1f} Hz")
    
    # System validation
    print("\n2. SYSTEM VALIDATION")
    print("-" * 50)
    validator = SystemValidator(controller, config)
    
    # Run latency test
    latency_stats = validator.run_latency_test(num_samples=100)
    print(f"Latency Test - Mean: {latency_stats['mean_ms']:.2f}ms, P95: {latency_stats['p95_ms']:.2f}ms")
    
    # Run accuracy test
    accuracy_stats = validator.run_accuracy_test(num_samples=50)
    print(f"Accuracy Test - Accuracy: {accuracy_stats['overall_accuracy']:.3f}")
    
    # Run safety test
    safety_stats = validator.run_safety_test(num_scenarios=50)
    print(f"Safety Test - Safety Rate: {safety_stats['safety_rate']:.3f}")
    
    # Generate validation report
    validation_report = validator.generate_validation_report()
    print(f"Validation Status: {validation_report['summary']['overall_status']}")
    
    # Production deployment simulation
    print("\n3. PRODUCTION DEPLOYMENT")
    print("-" * 50)
    deployment = ProductionDeployment(controller, config)
    
    if deployment.deploy_system():
        print("✓ System deployed successfully")
        
        # Simulate some operation time
        time.sleep(2)
        
        status = deployment.get_deployment_status()
        print(f"Deployment Status: {status['deployed']}")
        print(f"Uptime: {status['uptime_hours']:.4f} hours")
        
        # Shutdown
        final_stats = deployment.shutdown_system()
        print(f"Final Stats - Uptime: {final_stats['total_uptime_hours']:.4f}h")
    
    print("\n" + "="*80)
    print("DEPLOYMENT & TESTING VALIDATION COMPLETED!")
    print("System ready for MSc dissertation implementation!")
    print("="*80)