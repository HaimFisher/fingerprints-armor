"""
Adversarial Attacks Module

This module provides an implementation of the Attack class, which can be used to
perform various adversarial attacks on input tensors. The class currently
supports the Iterative Fast Gradient Sign Method (IFGSM) attack.

The Attack class takes a PyTorch model as input and provides a method to
generate adversarial examples that can fool the model.
"""

import os

import numpy as np
import torch
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from src.logger_config import get_logger


logger = get_logger(__name__)


class Attack:
    """
    A class that performs various types of attacks on input tensors.
    """

    def __init__(self, model):
        """
        Initialize the Attack class with a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model to be used for the attack.
        """
        self.model = model

    def ifgsm_attack(
        self,
        image_name: str,
        folder_name: str,
        folder_idx: int,
        tensor: torch.Tensor,
        target_class_id: int,
        target_class_tensor: torch.Tensor,
        target_folder_path: str,
        epsilon: float = 0.01,
        attack_confidence: float = 0.7,
        save_name: str = "eval_tmp",
    ) -> tuple[torch.Tensor, str]:
        """
        Perform a Fast Gradient Sign Method (FGSM) attack on the input tensor.

        Args:
            image_name (str): The name of the input image.
            folder_name (str): The name of the folder containing the input image.
            folder_idx (int): The index of the folder containing the input image.
            tensor (torch.Tensor): The input tensor.
            target_class_id (int): The target class ID.
            target_class_tensor (torch.Tensor): The target class tensor.
            target_folder_path (str): The path to the target folder.
            epsilon (float): The epsilon value for the FGSM attack. Defaults to 0.01.
            attack_confidence (float): The attack confidence threshold. Defaults to 0.7.
            save_name (str): The name to use for the saved file. Defaults to "eval_tmp".

        Returns:
            tuple[torch.Tensor, str]: The attacked tensor and the name of the saved file.
        """
        attacked = False
        attack_iter = 0
        adv_x = tensor

        if os.path.exists(target_folder_path):
            attacked_files = [
                file
                for file in os.listdir(target_folder_path)
                if file.startswith(image_name)
            ]
            if attacked_files != []:
                logger.info(f"Image: {image_name} attacked version already exist in DB")
                return None, None

        while not attacked and attack_iter < 150:
            adv_x = fast_gradient_method(self.model,
                                         adv_x,
                                         epsilon,
                                         np.inf,
                                         y=target_class_tensor,
                                         targeted=True)

            attack_iter += 1

            with torch.no_grad():
                out = self.model(adv_x)

            probabilities = torch.nn.functional.softmax(out[0], dim=0)
            top1_prob, top1_catid = torch.topk(probabilities, 1)

            if (
                top1_catid[0] == target_class_id
                and top1_prob[0].item() > attack_confidence
            ):
                logger.info(
                    f"Successfully attacked image to class {target_class_id} "
                    f"with confidence {top1_prob[0].item():.4f} "
                    f"(iteration {attack_iter})"
                )
                attacked = True
                break
        
        if not attacked:
            logger.error(
                f"Attack failed after {attack_iter} iterations. "
                f"Final confidence: {top1_prob[0].item():.4f}"
            )

        file_name = (
            f"{image_name}_{folder_idx}_to_{top1_catid[0]}_ifgsm_{attack_iter}_{epsilon}_"
            f"{top1_prob[0].item():.2f}.pkl"
        )

        logger.debug(f"Saving attacked image as: {file_name}")
        return adv_x, file_name
