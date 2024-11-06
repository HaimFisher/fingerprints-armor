"""
Model Layer Activation Extractor Module

This module provides functionality to extract the activations from the last N layers
of a neural network model. It's primarily used for model protection and analysis purposes
by monitoring the internal activation patterns of deep layers.
"""

import logging
from typing import Dict, List, Union, Optional
from pathlib import Path

import numpy as np
import torch
from torch import nn, Tensor
from torchvision.transforms import ToTensor
from PIL.Image import Image

from ..logger_config import get_logger


logger = get_logger(__name__)


class ModelLastLayers:
    """
    A class to extract and analyze activations from the last N layers of a neural network.
    
    This class wraps a PyTorch model and provides functionality to:
    1. Identify the last N FC significant layers based on parameter count
    2. Extract activations from these layers during forward passes
    3. Process and concatenate these activations for analysis
    
    Attributes:
        model (nn.Module): The PyTorch model to analyze
        n_layers (int): Number of layers to monitor
        activations (Dict[str, Tensor]): Storage for layer activations
        handles (List): Hooks for layer monitoring
    """


    def __init__(
        self, 
        model: nn.Module, 
        n_layers: int = 5,
        params_threshold: int = 1000
    ):
        """
        Initialize the ModelLastLayers instance.

        Args:
            model (nn.Module): PyTorch model to analyze
            n_layers (int, optional): Number of layers to monitor. Defaults to 5.
            params_threshold (int, optional): Minimum parameter count for a FC layer
                to be considered significant. Defaults to 1000.
        """
        self.model = model
        self.n_layers = n_layers
        self.activations: Dict[str, Tensor] = {}
        self.handles: List = []

        logger.info(
            f"Initializing ModelLastLayers with {n_layers} layers "
            f"and parameter threshold {params_threshold}"
        )

        self._register_hooks(params_threshold)

    def _get_activation_hook(self, name: str):
        """
        Create a hook function for collecting layer activations.

        Args:
            name (str): Identifier for the layer

        Returns:
            callable: Hook function that stores layer activations
        """
        def hook(model: nn.Module, input: Tensor, output: Tensor):
            self.activations[name] = output.detach()
        return hook

    def _register_hooks(self, param_threshold: int) -> None:
        """
        Register forward hooks on the last N significant layers of the model.

        A layer is considered significant if it has more parameters than
        the parameter threshold.

        Args:
            param_threshold (int): Minimum parameter count for a layer
                to be considered significant
        """
        significant_layers = []
        total_params = 0

        # Identify significant layers
        for name, module in reversed(list(self.model.named_children())):
            if len(significant_layers) == self.n_layers:
                break

            param_count = sum(p.numel() for p in module.parameters())
            total_params += param_count

            if param_count > param_threshold:
                significant_layers.append(module)
                logger.debug(f"Registered layer {name} with {param_count:,} parameters")

        # Register hooks
        for idx, layer in enumerate(significant_layers):
            handle = layer.register_forward_hook(
                self._get_activation_hook(f"layer_{idx}")
            )
            self.handles.append(handle)

        logger.info(
            f"Registered {len(significant_layers)} layers "
            f"with total parameters: {total_params:,}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the model.

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Model output
        """
        return self.model(x)

    def get_activations(self, image: Union[Image, Tensor]) -> np.ndarray:
        """
        Extract activations from the monitored layers for a given input.

        Args:
            image (Union[Image, Tensor]): Input image, either as a PIL Image
                or PyTorch Tensor

        Returns:
            np.ndarray: Concatenated activations from all monitored layers

        Raises:
            ValueError: If the input format is invalid
            RuntimeError: If the forward pass fails
        """
        # Prepare input tensor
        try:
            image_tensor = (
                image if isinstance(image, Tensor)
                else ToTensor()(image)
            )
        except Exception as e:
            logger.error(f"Failed to process input image: {e}")
            raise ValueError("Invalid input format") from e

        # Perform forward pass
        try:
            with torch.no_grad():
                try:
                    _ = self.model(image_tensor)
                except RuntimeError:
                    logger.debug("Adding batch dimension to input")
                    image_tensor = image_tensor.unsqueeze(0)
                    _ = self.model(image_tensor)
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            raise RuntimeError("Model forward pass failed") from e

        activations = np.concatenate(
            [
                self.activations[key][0].flatten().numpy()
                for key in self.activations.keys()
            ]
        )

        logger.debug(f"Extracted {len(activations)} activation values")
        return activations

    def __del__(self):
        """Clean up by removing all registered hooks."""
        for handle in self.handles:
            handle.remove()
        logger.debug("Removed all activation hooks")
