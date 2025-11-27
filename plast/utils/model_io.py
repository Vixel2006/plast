import pickle
import os
import numpy as np
from plast.nn.pipeline import Pipeline
from plast.nn.linear import Linear
from plast.nn.activations import Sigmoid, ReLU
from plast.functions import from_data

def save_model(model: Pipeline, filepath: str):
    """Saves a model's state to the specified filepath using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model_state = {
        "layers": []
    }
    for layer in model.layers:
        layer_state = {
            "type": layer.__class__.__name__
        }
        if isinstance(layer, Linear):
            layer_state["in_features"] = layer.W.shape[1]
            layer_state["out_features"] = layer.W.shape[0]
            layer_state["bias"] = layer.bias
            layer_state["weights"] = layer.W.data
            if layer.bias:
                layer_state["bias_data"] = layer.B.data
        # Add other layer types here as needed
        model_state["layers"].append(layer_state)

    with open(filepath, 'wb') as f:
        pickle.dump(model_state, f)
    print(f"Model state saved to {filepath}")

def load_model(filepath: str) -> Pipeline:
    """Loads a model's state from the specified filepath and reconstructs the model."""
    with open(filepath, 'rb') as f:
        model_state = pickle.load(f)
    
    layers = []
    for layer_state in model_state["layers"]:
        layer_type = layer_state["type"]
        if layer_type == "Linear":
            in_features = layer_state["in_features"]
            out_features = layer_state["out_features"]
            bias = layer_state["bias"]
            linear_layer = Linear(in_features, out_features, bias)
            linear_layer.W = from_data(layer_state["weights"].shape, layer_state["weights"])
            if bias:
                linear_layer.B = from_data(layer_state["bias_data"].shape, layer_state["bias_data"])
            layers.append(linear_layer)
        elif layer_type == "Sigmoid":
            layers.append(Sigmoid())
        elif layer_type == "ReLU":
            layers.append(ReLU())
        # Add other layer types here as needed
    
    model = Pipeline(*layers)
    print(f"Model loaded from {filepath}")
    return model
