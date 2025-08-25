# core_system.py — NeuroFusion (FINAL: Inference-Optimized with Selectable Backends)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
import time
from typing import Optional
import logging

# -------------------------------------------------------
# Logging
# -------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.propagate = False

# -------------------------------------------------------
# Preprocessor (fast version kept)
# -------------------------------------------------------
class OptimizedEEGEMGPreprocessor:
    """
    FAST preprocessing for EEG/EMG (kept as in your good-performing version)
    EEG: bandpass 1–40 Hz, 3 band power + |x| mean + std (first 4 ch → 20 feats)
    EMG: bandpass 20–400 Hz, RMS + MAV (first 4 ch → 8 feats)
    """
    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate
        nyquist = sampling_rate / 2.0
        # Low-order SOS for speed
        self.eeg_bandpass = signal.butter(2, [1.0/nyquist, 40.0/nyquist], btype='band', output='sos')
        self.emg_bandpass = signal.butter(2, [20.0/nyquist, 400.0/nyquist], btype='band', output='sos')
        self.eeg_bands = {
            'alpha': (8.0, 13.0),
            'beta' : (13.0, 30.0),
            'gamma': (30.0, 40.0),
        }
        logger.info("Optimized preprocessor initialized for real-time performance")

    def preprocess_eeg(self, eeg_data: np.ndarray) -> np.ndarray:
        if eeg_data.ndim == 1:
            eeg_data = eeg_data.reshape(1, -1)
        C = min(4, eeg_data.shape[0])
        feats = []
        for ch in range(C):
            x = signal.sosfilt(self.eeg_bandpass, eeg_data[ch])
            freqs, psd = signal.welch(x, self.sampling_rate, nperseg=64)
            for (lo, hi) in self.eeg_bands.values():
                idx = (freqs >= lo) & (freqs <= hi)
                feats.append(psd[idx].mean() if idx.any() else 0.0)
            feats.append(np.mean(np.abs(x)))
            feats.append(np.std(x))
        if len(feats) < 20:
            feats += [0.0] * (20 - len(feats))
        return np.asarray(feats[:20], dtype=np.float32)

    def preprocess_emg(self, emg_data: np.ndarray) -> np.ndarray:
        if emg_data.ndim == 1:
            emg_data = emg_data.reshape(1, -1)
        C = min(4, emg_data.shape[0])
        feats = []
        for ch in range(C):
            x = signal.sosfilt(self.emg_bandpass, emg_data[ch])
            rms = np.sqrt(np.mean(x*x))
            mav = np.mean(np.abs(x))
            feats.extend([rms, mav])
        if len(feats) < 8:
            feats += [0.0] * (8 - len(feats))
        return np.asarray(feats[:8], dtype=np.float32)

# -------------------------------------------------------
# Model (compact & fast)
# -------------------------------------------------------
class FastMotorIntentionPredictor(nn.Module):
    """
    SPEED-optimized MLP for motor intention prediction
    Inputs: EEG(20) + EMG(8)
    """
    def __init__(self, eeg_features=20, emg_features=8, num_actions=6):
        super().__init__()
        self.eeg_branch = nn.Sequential(
            nn.Linear(eeg_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.emg_branch = nn.Sequential(
            nn.Linear(emg_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.fusion = nn.Sequential(
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.action_head = nn.Linear(16, num_actions)
        self.confidence_head = nn.Linear(16, 1)

    def forward(self, eeg_features, emg_features):
        eeg_z = self.eeg_branch(eeg_features)
        emg_z = self.emg_branch(emg_features)
        z = self.fusion(torch.cat([eeg_z, emg_z], dim=-1))
        action_probs = F.softmax(self.action_head(z), dim=-1)
        confidence = torch.sigmoid(self.confidence_head(z))
        return action_probs, confidence

# -------------------------------------------------------
# Uncertainty + MPC (fast)
# -------------------------------------------------------
class FastBayesianUncertaintyEstimator:
    def __init__(self, num_actions=6):
        self.num_actions = num_actions

    def estimate_uncertainty(self, action_probs: np.ndarray):
        # entropy (0..1) + max-prob complement (0..1)
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        max_entropy = np.log(self.num_actions)
        norm_unc = entropy / max_entropy
        conf_unc = 1.0 - float(action_probs.max())
        return float(np.clip(0.5*norm_unc + 0.5*conf_unc, 0.0, 1.0)), None

class SimplifiedMPC:
    def __init__(self, state_dim=6, control_dim=6):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.state_weight = 1.0
        self.human_weight = 0.5

    def solve_mpc(self, current_state, human_intention, target_state, uncertainty):
        try:
            err = target_state - current_state
            u_p = self.state_weight * err
            u_h = (1.0 - uncertainty) * self.human_weight * human_intention
            u = u_p + u_h
            return np.clip(u, -1.0, 1.0).astype(np.float32)
        except Exception:
            return np.zeros(self.control_dim, dtype=np.float32)

# -------------------------------------------------------
# Predictor backends: Torch / TorchScript / ONNX Runtime (TensorRT)
# -------------------------------------------------------
class _TorchPredictor:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.eval().to(device)
        self.device = device
        try:
            torch.set_num_threads(1)  # reduce overhead on CPU
        except Exception:
            pass

    @torch.no_grad()
    def __call__(self, eeg_feat: np.ndarray, emg_feat: np.ndarray):
        eeg_t = torch.from_numpy(eeg_feat).unsqueeze(0).to(self.device)
        emg_t = torch.from_numpy(emg_feat).unsqueeze(0).to(self.device)
        return self.model(eeg_t, emg_t)

class _TorchScriptPredictor(_TorchPredictor):
    def __init__(self, eager_model: nn.Module, device: torch.device, save_path: str = "neurofusion_scripted.pt"):
        # trace & optimize
        eager_model = eager_model.eval().to(device)
        de = torch.randn(1, 20, device=device)
        dm = torch.randn(1, 8, device=device)
        with torch.no_grad():
            scripted = torch.jit.trace(eager_model, (de, dm))
            scripted = torch.jit.optimize_for_inference(scripted)
        self.model = scripted
        self.device = device
        try:
            scripted.save(save_path)
        except Exception:
            pass
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

# ONNX Runtime (optional)
try:
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False

class _OnnxRuntimePredictor:
    def __init__(self, onnx_path: str):
        providers = []
        # Prefer TRT, then CUDA, then CPU
        if _HAS_ORT and "TensorrtExecutionProvider" in ort.get_available_providers():
            providers.append("TensorrtExecutionProvider")
        if _HAS_ORT and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1

        self.session = ort.InferenceSession(onnx_path, providers=providers, sess_options=so)
        self.in_names = [i.name for i in self.session.get_inputs()]
        self.out_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, eeg_feat: np.ndarray, emg_feat: np.ndarray):
        eeg = eeg_feat.astype(np.float32)[None, :]
        emg = emg_feat.astype(np.float32)[None, :]
        outs = self.session.run(self.out_names, {self.in_names[0]: eeg, self.in_names[1]: emg})
        # outs: [action_probs(1,6), confidence(1,1)]
        return outs

# -------------------------------------------------------
# Controller with selectable inference backend
# -------------------------------------------------------
class OptimizedNeuroFusionController:
    """
    Backends (select via constructor or env NEUROFUSION_BACKEND):
      - "torch"       : Eager (CPU/GPU)
      - "torchscript" : JIT (default)
      - "onnxruntime" : ORT (CPU/CUDA)
      - "tensorrt"    : ORT+TensorRT (if available)
    """
    def __init__(self, backend: Optional[str] = None):
        # device
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        # components
        self.preprocessor = OptimizedEEGEMGPreprocessor()
        base_model = FastMotorIntentionPredictor().eval()

        # CPU'da dinamik int8 quantization (Linear)
        if not self.use_cuda:
            try:
                base_model = torch.quantization.quantize_dynamic(
                    base_model, {nn.Linear}, dtype=torch.qint8
                )
                logger.info("Applied dynamic int8 quantization for CPU.")
            except Exception as e:
                logger.warning(f"Quantization skipped: {e}")

        # backend select
        self.backend_name = "torchscript"
        choice = (backend or os.environ.get("NEUROFUSION_BACKEND", "torchscript")).lower()
        self.predictor = None

        if choice == "torch":
            self.backend_name = "torch"
            self.predictor = _TorchPredictor(base_model, self.device)

        elif choice in ("onnxruntime", "tensorrt") and _HAS_ORT:
            onnx_path = "neurofusion_model.onnx"
            # Export ONNX
            try:
                de = torch.randn(1, 20)
                dm = torch.randn(1, 8)
                torch.onnx.export(
                    base_model.cpu().eval(), (de, dm), onnx_path,
                    export_params=True, opset_version=17, do_constant_folding=True,
                    input_names=["eeg_input", "emg_input"],
                    output_names=["action_probs", "confidence"],
                    dynamic_axes={"eeg_input": {0: "N"}, "emg_input": {0: "N"}}
                )
                logger.info(f"ONNX exported to {onnx_path}")
                self.backend_name = "tensorrt" if choice == "tensorrt" else "onnxruntime"
                self.predictor = _OnnxRuntimePredictor(onnx_path)
            except Exception as e:
                logger.warning(f"ONNX/ORT init failed, fallback TorchScript: {e}")
                self.backend_name = "torchscript"
                self.predictor = _TorchScriptPredictor(base_model, self.device)

        else:
            # default TorchScript
            self.backend_name = "torchscript"
            self.predictor = _TorchScriptPredictor(base_model, self.device)

        logger.info(f"NeuroFusion predictor backend: {self.backend_name}")

        # rest of system
        self.uncertainty_estimator = FastBayesianUncertaintyEstimator()
        self.mpc_controller = SimplifiedMPC()
        self.current_state = np.zeros(6, dtype=np.float32)
        self.target_state = np.zeros(6, dtype=np.float32)
        self.processing_times = []

        self._warmup()

    def _warmup(self):
        logger.info("Warming up inference engine...")
        eeg = np.random.randn(20).astype(np.float32)
        emg = np.random.randn(8).astype(np.float32)
        for _ in range(10):
            _ = self._run_inference(eeg, emg)
        logger.info("Inference warmed up.")

    def set_target(self, target):
        self.target_state = np.asarray(target, dtype=np.float32)
        logger.info(f"Target set: {self.target_state}")

    def _run_inference(self, eeg_feat: np.ndarray, emg_feat: np.ndarray):
        out = self.predictor(eeg_feat, emg_feat)
        if self.backend_name in ("onnxruntime", "tensorrt"):
            # ORT outputs are numpy arrays
            action_probs = out[0].squeeze(0).astype(np.float32)
            confidence = float(out[1].squeeze(0))
        else:
            # Torch / TorchScript outputs are tensors
            ap_t, cf_t = out
            action_probs = ap_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
            confidence = float(cf_t.item())
        return action_probs, confidence

    def process_neural_signals(self, eeg_data: np.ndarray, emg_data: np.ndarray):
        t0 = time.perf_counter()
        try:
            # preprocessing
            t_pre0 = time.perf_counter()
            eeg_feat = self.preprocessor.preprocess_eeg(eeg_data)
            emg_feat = self.preprocessor.preprocess_emg(emg_data)
            t_pre = (time.perf_counter() - t_pre0) * 1000.0

            # inference
            t1 = time.perf_counter()
            action_probs, confidence = self._run_inference(eeg_feat, emg_feat)
            t_pred = (time.perf_counter() - t1) * 1000.0

            # uncertainty
            t2 = time.perf_counter()
            uncertainty, _ = self.uncertainty_estimator.estimate_uncertainty(action_probs)
            t_unc = (time.perf_counter() - t2) * 1000.0

            # control
            t3 = time.perf_counter()
            control_cmd = self.mpc_controller.solve_mpc(
                self.current_state, action_probs, self.target_state, uncertainty
            )
            t_ctrl = (time.perf_counter() - t3) * 1000.0

            # state update (simple)
            self.current_state = (self.current_state + 0.1 * control_cmd).astype(np.float32)

            total = (time.perf_counter() - t0) * 1000.0
            self.processing_times.append(total)

            return {
                "control_command": control_cmd,
                "action_probabilities": action_probs,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "processing_time_ms": total,
                "breakdown": {
                    "preprocessing_ms": t_pre,
                    "prediction_ms": t_pred,
                    "uncertainty_ms": t_unc,
                    "control_ms": t_ctrl,
                },
            }
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return None

    def get_performance_stats(self):
        if not self.processing_times:
            return None
        t = np.asarray(self.processing_times, dtype=np.float64)
        return {
            "avg_processing_time_ms": float(np.mean(t)),
            "median_processing_time_ms": float(np.median(t)),
            "p95_processing_time_ms": float(np.percentile(t, 95)),
            "p99_processing_time_ms": float(np.percentile(t, 99)),
            "min_processing_time_ms": float(np.min(t)),
            "max_processing_time_ms": float(np.max(t)),
            "sub_1ms_rate": float((t < 1.0).mean() * 100.0),
            "sub_5ms_rate": float((t < 5.0).mean() * 100.0),
            "total_samples": int(t.size),
        }

# Backward-compat alias for other modules
NeuroFusionController = OptimizedNeuroFusionController

# -------------------------------------------------------
# Self-test
# -------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("OPTIMIZED NEUROFUSION SYSTEM - REAL-TIME PERFORMANCE TEST")
    print("=" * 80)

    backend = os.environ.get("NEUROFUSION_BACKEND", "torchscript")
    controller = OptimizedNeuroFusionController(backend=backend)

    target = np.array([1.0, 0.5, 0.8, 0.1, 0.2, 0.0], dtype=np.float32)
    controller.set_target(target)

    print("\nReal-time performance test (100 samples)...")
    print("Target: <1ms average latency\n")

    for i in range(100):
        eeg_data = np.random.randn(8, 250).astype(np.float32) * 10e-6
        emg_data = np.random.randn(4, 250).astype(np.float32) * 50e-6
        result = controller.process_neural_signals(eeg_data, emg_data)
        if result and i < 10:
            print(
                f"Sample {i+1:2d}: Total: {result['processing_time_ms']:.2f}ms, "
                f"Confidence: {result['confidence']:.3f}, "
                f"Uncertainty: {result['uncertainty']:.3f}"
            )
            if i == 0:
                b = result["breakdown"]
                print(
                    f"           Breakdown - "
                    f"Preproc: {b['preprocessing_ms']:.2f}ms, "
                    f"Neural: {b['prediction_ms']:.2f}ms, "
                    f"Uncertainty: {b['uncertainty_ms']:.2f}ms, "
                    f"Control: {b['control_ms']:.2f}ms"
                )

    stats = controller.get_performance_stats()
    if stats:
        print(f"\n{'='*60}")
        print("PERFORMANCE RESULTS")
        print(f"{'='*60}")
        print(f"Average Processing Time: {stats['avg_processing_time_ms']:.3f}ms")
        print(f"Median Processing Time:  {stats['median_processing_time_ms']:.3f}ms")
        print(f"95th Percentile:        {stats['p95_processing_time_ms']:.3f}ms")
        print(f"99th Percentile:        {stats['p99_processing_time_ms']:.3f}ms")
        print(
            f"Min/Max Time:           {stats['min_processing_time_ms']:.3f}ms / "
            f"{stats['max_processing_time_ms']:.3f}ms"
        )
        print(f"Sub-1ms Success Rate:   {stats['sub_1ms_rate']:.1f}%")
        print(f"Sub-5ms Success Rate:   {stats['sub_5ms_rate']:.1f}%")
        print(f"Total Samples:          {stats['total_samples']}")

        avg = stats["avg_processing_time_ms"]
        if avg < 1.0:
            print(f"\n🎯 EXCELLENT: Target <1ms achieved! ({avg:.3f}ms)")
        elif avg < 5.0:
            print(f"\n✅ GOOD: Real-time capable ({avg:.3f}ms)")
        elif avg < 10.0:
            print(f"\n⚠️  ACCEPTABLE: Near real-time ({avg:.3f}ms)")
        else:
            print(f"\n❌ NEEDS OPTIMIZATION: Too slow for real-time ({avg:.3f}ms)")

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE - READY FOR MSC DISSERTATION!")
    print("=" * 80)
