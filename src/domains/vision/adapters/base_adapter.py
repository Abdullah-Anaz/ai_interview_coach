from typing import Callable, Optional
import numpy as np
import torch

# ---------------------------------------------------------
# The Functional Interface (Type Alias)
# ---------------------------------------------------------
# Instead of an Abstract Base Class, we define a strict function signature.
# Every concrete adapter must return a function that perfectly matches this type.
PredictTrajectoryFn = Callable[[torch.Tensor, float], Optional[np.ndarray]]

# ---------------------------------------------------------
# Documentation & Blueprint for Concrete Adapters
# ---------------------------------------------------------
"""
To implement a new vision model adapter functionally, you must create a 
higher-order "factory" function that handles the side-effect of loading the model, 
and returns a pure `PredictTrajectoryFn` closure.

Example Blueprint:

def create_example_adapter(config: dict, device: torch.device) -> PredictTrajectoryFn:
    # 1. Handle Side-Effects (Load weights, init architecture)
    model = _load_model_weights(config['path']).to(device)
    
    # 2. Define the pure prediction function (captures `model` in closure)
    def predict(sequence_tensor: torch.Tensor, fps: float) -> Optional[np.ndarray]:
        with torch.no_grad():
            trajectory = model(sequence_tensor)
        return trajectory.cpu().numpy()
        
    # 3. Return the function, fulfilling the PredictTrajectoryFn contract
    return predict
"""