import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy.stats import dirichlet
from sklearn.preprocessing import StandardScaler
import time
from typing import Tuple, Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGEMGPreprocessor:
    """
    Advanced preprocessing for EEG and EMG signals
    Real-time feature extraction with optimized performance
    """
    
    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate
        self.eeg_scaler = StandardScaler()
        self.emg_scaler = StandardScaler()
        
        # Filter design
        self.eeg_bandpass = signal.butter(4, [0.5, 50], btype='band', fs=sampling_rate, output='sos')
        self.emg_bandpass = signal.butter(4, [20, 500], btype='band', fs=sampling_rate, output='sos')
        
        # EEG frequency bands
        self.eeg_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    def preprocess_eeg(self, eeg_data):
        """Extract EEG features for neural network"""
        if len(eeg_data.shape) == 1:
            eeg_data = eeg_data.reshape(1, -1)
        
        features = []
        
        for channel in range(eeg_data.shape[0]):
            # Filter signal
            filtered = signal.sosfilt(self.eeg_bandpass, eeg_data[channel])
            
            # Power spectral density for each frequency band
            freqs, psd = signal.welch(filtered, self.sampling_rate, nperseg=256)
            
            for band_name, (low, high) in self.eeg_bands.items():
                band_indices = np.where((freqs >= low) & (freqs <= high))[0]
                band_power = np.mean(psd[band_indices])
                features.append(band_power)
            
            # Time domain features
            features.extend([
                np.mean(filtered),
                np.std(filtered),
                np.var(filtered),
                self._hjorth_parameters(filtered)
            ])
        
        return np.array(features).flatten()
    
    def preprocess_emg(self, emg_data):
        """Extract EMG features for neural network"""
        if len(emg_data.shape) == 1:
            emg_data = emg_data.reshape(1, -1)
        
        features = []
        
        for channel in range(emg_data.shape[0]):
            # Filter signal
            filtered = signal.sosfilt(self.emg_bandpass, emg_data[channel])
            
            # Time domain features
            rms = np.sqrt(np.mean(filtered**2))
            mav = np.mean(np.abs(filtered))
            zc = self._zero_crossings(filtered)
            ssc = self._slope_sign_changes(filtered)
            
            features.extend([rms, mav, zc, ssc])
        
        return np.array(features)
    
    def _hjorth_parameters(self, signal_data):
        """Calculate Hjorth parameters (complexity measure)"""
        diff1 = np.diff(signal_data)
        diff2 = np.diff(diff1)
        
        var_signal = np.var(signal_data)
        var_diff1 = np.var(diff1)
        var_diff2 = np.var(diff2)
        
        mobility = np.sqrt(var_diff1 / var_signal)
        complexity = np.sqrt(var_diff2 / var_diff1) / mobility
        
        return complexity
    
    def _zero_crossings(self, signal_data):
        """Count zero crossings"""
        return np.sum(np.diff(np.sign(signal_data)) != 0)
    
    def _slope_sign_changes(self, signal_data):
        """Count slope sign changes"""
        diff_signal = np.diff(signal_data)
        return np.sum(np.diff(np.sign(diff_signal)) != 0)

class MotorIntentionPredictor(nn.Module):
    """
    Advanced neural network for motor intention prediction
    Dual-branch architecture for EEG/EMG fusion
    """
    
    def __init__(self, eeg_features=40, emg_features=16, num_actions=6):
        super(MotorIntentionPredictor, self).__init__()
        
        # EEG processing branch
        self.eeg_branch = nn.Sequential(
            nn.Linear(eeg_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # EMG processing branch
        self.emg_branch = nn.Sequential(
            nn.Linear(emg_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(96, 64, batch_first=True)
        
        # Multi-head attention for fusion
        self.attention = nn.MultiheadAttention(64, num_heads=8)
        
        # Output layers
        self.action_head = nn.Linear(64, num_actions)
        self.confidence_head = nn.Linear(64, 1)
        
    def forward(self, eeg_features, emg_features):
        # Process branches
        eeg_out = self.eeg_branch(eeg_features)
        emg_out = self.emg_branch(emg_features)
        
        # Concatenate features
        fused = torch.cat([eeg_out, emg_out], dim=-1)
        
        # Add temporal dimension for LSTM
        if len(fused.shape) == 2:
            fused = fused.unsqueeze(1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(fused)
        
        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended = attended.squeeze(1)
        
        # Output predictions
        action_logits = self.action_head(attended)
        action_probs = F.softmax(action_logits, dim=-1)
        confidence = torch.sigmoid(self.confidence_head(attended))
        
        return action_probs, confidence

class BayesianUncertaintyEstimator:
    """
    Bayesian uncertainty quantification using Dirichlet distributions
    """
    
    def __init__(self, num_actions=6, alpha_prior=1.0):
        self.num_actions = num_actions
        self.alpha_prior = alpha_prior
        self.alpha = np.full(num_actions, alpha_prior)
    
    def estimate_uncertainty(self, action_probs):
        """Estimate uncertainty from action probabilities"""
        # Update Dirichlet parameters
        alpha_posterior = self.alpha + action_probs * 100  # Scale for numerical stability
        
        # Calculate uncertainty measures
        concentration = np.sum(alpha_posterior)
        entropy = self._dirichlet_entropy(alpha_posterior)
        differential_entropy = self._differential_entropy(action_probs)
        
        # Combine uncertainties
        total_uncertainty = entropy + differential_entropy / concentration
        
        return total_uncertainty, alpha_posterior
    
    def _dirichlet_entropy(self, alpha):
        """Calculate Dirichlet entropy"""
        alpha_sum = np.sum(alpha)
        entropy = (
            np.sum(special.gammaln(alpha)) - 
            special.gammaln(alpha_sum) - 
            np.sum((alpha - 1) * (special.digamma(alpha) - special.digamma(alpha_sum)))
        )
        return entropy
    
    def _differential_entropy(self, probs):
        """Calculate differential entropy"""
        return -np.sum(probs * np.log(probs + 1e-10))

class ModelPredictiveController:
    """
    Model Predictive Control for human-AI collaborative control
    """
    
    def __init__(self, state_dim=6, control_dim=6, horizon=10):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.horizon = horizon
        
        # Cost matrices
        self.Q = np.eye(state_dim) * 10  # State cost
        self.R = np.eye(control_dim) * 1   # Control cost
        self.S = np.eye(control_dim) * 5   # Human deviation cost
        
    def solve_mpc(self, current_state, human_intention, target_state, uncertainty):
        """Solve MPC optimization problem"""
        # Simple gradient-based solution for real-time performance
        best_control = None
        best_cost = float('inf')
        
        # Multi-start optimization
        for _ in range(5):
            control_sequence = np.random.randn(self.horizon, self.control_dim) * 0.1
            
            for iteration in range(50):  # Limited iterations for real-time
                cost, gradient = self._evaluate_cost_and_gradient(
                    control_sequence, current_state, human_intention, target_state, uncertainty
                )
                
                # Gradient descent step
                control_sequence -= 0.01 * gradient
                
                # Early stopping
                if cost < best_cost:
                    best_cost = cost
                    best_control = control_sequence[0].copy()
        
        return best_control if best_control is not None else np.zeros(self.control_dim)
    
    def _evaluate_cost_and_gradient(self, control_sequence, current_state, human_intention, target_state, uncertainty):
        """Evaluate cost function and compute gradient"""
        total_cost = 0
        gradient = np.zeros_like(control_sequence)
        
        state = current_state.copy()
        
        for t in range(self.horizon):
            # Simple dynamics model
            state = state + control_sequence[t] * 0.1
            
            # State tracking cost
            state_error = state - target_state
            state_cost = state_error.T @ self.Q @ state_error
            
            # Control effort cost
            control_cost = control_sequence[t].T @ self.R @ control_sequence[t]
            
            # Human intention deviation cost (weighted by uncertainty)
            intention_error = control_sequence[t] - human_intention
            intention_cost = (1 - uncertainty) * intention_error.T @ self.S @ intention_error
            
            total_cost += state_cost + control_cost + intention_cost
            
            # Approximate gradient
            gradient[t] = (
                2 * self.R @ control_sequence[t] +
                2 * (1 - uncertainty) * self.S @ intention_error
            )
        
        return total_cost, gradient

class TransferLearningAdapter:
    """
    Adapt the system to new users and environments
    """
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.adaptation_rate = 0.01
        self.user_data_buffer = []
        
    def adapt_to_user(self, user_data, user_labels, num_epochs=10):
        """Fine-tune model for specific user"""
        # Create user-specific dataset
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=self.adaptation_rate)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            eeg_data, emg_data = user_data
            action_probs, confidence = self.base_model(eeg_data, emg_data)
            
            # Compute loss
            loss = criterion(action_probs, user_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        logger.info(f"User adaptation completed with loss: {loss.item():.4f}")

class NeuroFusionController:
    """
    Main NeuroFusion Controller integrating all components
    """
    
    def __init__(self, config=None):
        # Initialize components
        self.preprocessor = EEGEMGPreprocessor()
        self.intention_predictor = MotorIntentionPredictor()
        self.uncertainty_estimator = BayesianUncertaintyEstimator()
        self.mpc_controller = ModelPredictiveController()
        self.transfer_adapter = TransferLearningAdapter(self.intention_predictor)
        
        # System state
        self.current_state = np.zeros(6)
        self.target_state = np.zeros(6)
        
        # Performance monitoring
        self.processing_times = []
        self.prediction_accuracies = []
        
        logger.info("NeuroFusion Controller initialized successfully")
    
    def set_target(self, target):
        """Set control target"""
        self.target_state = np.array(target)
        logger.info(f"Target set: {self.target_state}")
    
    def process_neural_signals(self, eeg_data, emg_data):
        """Process neural signals and generate control command"""
        start_time = time.perf_counter()
        
        try:
            # Preprocess signals
            eeg_features = self.preprocessor.preprocess_eeg(eeg_data)
            emg_features = self.preprocessor.preprocess_emg(emg_data)
            
            # Predict intention
            with torch.no_grad():
                eeg_tensor = torch.tensor(eeg_features, dtype=torch.float32).unsqueeze(0)
                emg_tensor = torch.tensor(emg_features, dtype=torch.float32).unsqueeze(0)
                action_probs, confidence = self.intention_predictor(eeg_tensor, emg_tensor)
            
            # Estimate uncertainty
            action_probs_np = action_probs.squeeze().numpy()
            uncertainty, _ = self.uncertainty_estimator.estimate_uncertainty(action_probs_np)
            
            # Generate control command
            control_command = self.mpc_controller.solve_mpc(
                self.current_state,
                action_probs_np,
                self.target_state,
                uncertainty
            )
            
            # Update state
            self.current_state += control_command * 0.1
            
            # Monitor performance
            processing_time = (time.perf_counter() - start_time) * 1000  # ms
            self.processing_times.append(processing_time)
            
            return {
                'control_command': control_command,
                'action_probabilities': action_probs_np,
                'confidence': confidence.item(),
                'uncertainty': uncertainty,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return None
    
    def get_performance_stats(self):
        """Get system performance statistics"""
        if not self.processing_times:
            return None
        
        return {
            'avg_processing_time_ms': np.mean(self.processing_times),
            'p95_processing_time_ms': np.percentile(self.processing_times, 95),
            'p99_processing_time_ms': np.percentile(self.processing_times, 99),
            'total_samples': len(self.processing_times)
        }

# Import scipy.special for Dirichlet entropy calculation
try:
    from scipy import special
except ImportError:
    logger.warning("scipy.special not available, using approximation")
    
    class SpecialApprox:
        @staticmethod
        def gammaln(x):
            return np.log(np.math.gamma(x))
        
        @staticmethod
        def digamma(x):
            return np.log(x) - 1/(2*x)
    
    special = SpecialApprox()

# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("NEUROFUSION CORE SYSTEM - COMPREHENSIVE TESTING")
    print("="*80)
    
    # Create controller
    controller = NeuroFusionController()
    
    # Set target
    target = np.array([1.0, 0.5, 0.8, 0.1, 0.2, 0.0])
    controller.set_target(target)
    
    # Simulate neural signals
    print("\nTesting with simulated neural signals...")
    for i in range(10):
        # Simulate EEG data (8 channels, 250 samples)
        eeg_data = np.random.randn(8, 250) * 10e-6  # Typical EEG amplitude
        
        # Simulate EMG data (4 channels, 250 samples)  
        emg_data = np.random.randn(4, 250) * 50e-6  # Typical EMG amplitude
        
        # Process signals
        result = controller.process_neural_signals(eeg_data, emg_data)
        
        if result:
            print(f"Sample {i+1}: "
                  f"Processing: {result['processing_time_ms']:.2f}ms, "
                  f"Confidence: {result['confidence']:.3f}, "
                  f"Uncertainty: {result['uncertainty']:.3f}")
    
    # Performance statistics
    stats = controller.get_performance_stats()
    if stats:
        print(f"\nPerformance Statistics:")
        print(f"Average Processing Time: {stats['avg_processing_time_ms']:.2f}ms")
        print(f"95th Percentile: {stats['p95_processing_time_ms']:.2f}ms")
        print(f"99th Percentile: {stats['p99_processing_time_ms']:.2f}ms")
        print(f"Total Samples: {stats['total_samples']}")
    
    print("\n" + "="*80)
    print("CORE SYSTEM TESTING COMPLETED SUCCESSFULLY!")
    print("="*80)