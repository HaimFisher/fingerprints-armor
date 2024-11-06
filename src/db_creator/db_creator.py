"""
This module implements the database creation functionality for storing neural network
activations of both original and adversarial examples.

The module uses the ModelLastLayers class to extract activations from the final layers
of the neural network and the Attack class to generate adversarial examples.

Input Structure:
    images/
    ├── category_1/
    │   ├── image_1.JPEG
    │   ├── image_2.JPEG
    │   └── ...
    └── ...

Output Format:
    database/
    ├── category_1/
    │   ├── orig/
    │   │   ├── image_1.pkl
    │   │   └── ...
    │   └── attack/
    │       ├── image_1_0_to_1_ifgsm_3_01.pkl
    │       └── ...
    └── ...
"""

import pickle
import random
import os
from typing import Optional, Tuple, Any

import torch
import timm
from timm.data import create_transform, resolve_data_config
from PIL import Image
from numpy import ndarray as NDArray

from src.db_creator.get_last_layers_activations import ModelLastLayers
from src.db_creator.attack import Attack
import src.settings as settings
from src.logger_config import setup_logger, get_logger

setup_logger()
logger = get_logger(__name__)


class DatabaseCreator:
    """
    Creates a database of neural network activations for both original and adversarial images.
    
    This class handles the creation of a structured database containing activations from
    the last layers of a neural network model for both original images and their
    corresponding adversarial examples.

    Attributes:
        model: Pre-trained neural network model (inception_v3).
        transform: Image transformation pipeline for model input.
        last_layers_model: Wrapper for extracting last layer activations.
        attack: Instance for generating adversarial examples.
    """

    def __init__(self,
                 model: Optional[torch.nn.Module] = None,
                 model_name: str = "inception_v3",
                 pretrained: bool = True) -> None:
        """
        Initialize the database creator with a generic model and transformations.

        Args:
            model (Optional[torch.nn.Module]): A PyTorch model to use. If None, a model will be created based on `model_name`.
            model_name (str): Name of the model to create if `model` is not provided. Defaults to "inception_v3".
            pretrained (bool): If True, loads a model pre-trained on ImageNet. Ignored if `model` is provided. Defaults to True.

        Attributes:
            model (torch.nn.Module): The model used for database creation, set to evaluation mode.
            transform (Callable): Transformations to prepare input images for the model.
            last_layers_model (ModelLastLayers): An instance to extract activations from the model's last layers.
            attack (Attack): An attack instance for IFGSM attack.
        """
        logger.info("Initializing DatabaseCreator")

        # If no model is provided, create one based on the specified name
        self.model = model or timm.create_model(model_name, pretrained=pretrained)
        self.model.eval()
        
        # Resolve configuration for the model's input requirements
        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)
        
        # Initialize last layers model and attack
        self.last_layers_model = ModelLastLayers(self.model)
        self.attack = Attack(self.model)
    
    def extract_activations(self, pil_img: Image.Image) -> NDArray:
        """
        Extract activations from the last layers for a given image.

        Args:
            pil_img (PIL.Image): Input image to process.

        Returns:
            NDArray: Activation values from the last layers.
        """
        return self.last_layers_model.get_activations(pil_img)
    
    @staticmethod
    def save_activations(data: Any, file_path: str) -> None:
        """
        Save activation data as a pickle file.

        Args:
            data (Any): Activation data to save.
            file_path (str): Target path for the pickle file.
        """
        with open(file_path, "wb") as file:
            pickle.dump(data, file)

    def process_original_image(
        self, 
        image_name: str, 
        target_path: str, 
        pil_img: torch.Tensor
    ) -> None:
        """
        Process and save original image activations.

        Args:
            image_name (str): Name of the source image.
            target_path (str): Directory to save the activation file.
            pil_img (torch.Tensor): Preprocessed image tensor.
        """
        pickle_name = image_name.replace(".JPEG", ".pkl")
        save_path = os.path.join(target_path, pickle_name)
        
        if os.path.exists(save_path):
            logger.info(f"Image: {image_name} normal version already exist in DB")
            return

        activations = self.extract_activations(pil_img)
        self.save_activations(activations, save_path)


    def create_adversarial_example(
        self,
        image_name: str,
        folder: str,
        target_path: str,
        folder_idx: int,
        pil_img: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, str]]:
        """
        Generate an adversarial example using IFGSM attack.

        The output filename format is:
        {image_name}_{orig_idx}_to_{attack_idx}_{attack_name}_{attack_iter}_{epsilon}.pkl

        Args:
            image_name (str): Name of the source image.
            folder (str): Category folder name.
            target_path (str): Directory to save results.
            folder_idx (int): Index of current category.
            pil_img (torch.Tensor): Preprocessed image tensor.

        Returns:
            Optional[Tuple[torch.Tensor, str]]: Adversarial image and target filename,
                or None if attack fails.
        """
        available_targets = [i for i in range(settings.INPUT_DB_CLASSES_COUNT) if i != folder_idx]
        target_idx = random.choice(available_targets)
        target_tensor = torch.tensor([target_idx])
        
        return self.attack.ifgsm_attack(
            image_name,
            folder,
            folder_idx,
            pil_img,
            target_idx,
            target_tensor,
            target_path,
        )

    def process_adversarial_image(
        self,
        image_name: str,
        folder: str,
        folder_idx: int,
        target_path: str,
        pil_img: torch.Tensor,
    ) -> None:
        """
        Process and save adversarial image activations.

        Args:
            image_name (str): Name of the source image.
            folder (str): Category folder name.
            folder_idx (int): Index of current category.
            target_path (str): Directory to save results.
            pil_img (torch.Tensor): Preprocessed image tensor.
        """
        adv_x, name = self.create_adversarial_example(
            image_name, folder, target_path, folder_idx, pil_img
        )
        if adv_x is None:
            return

        activations = self.extract_activations(pil_img)
        self.save_activations(activations, os.path.join(target_path, name))


    def create_database(
        self, 
        source_dir: str, 
        target_dir: str, 
        max_images_per_folder: int = 500
    ) -> None:
        """
        Create the activation database from an ImageNet-style dataset.

        This method processes the source directory's images, generates adversarial
        examples, and saves both original and adversarial activation data in a
        structured database format.

        Args:
            source_dir (str): Path to source ImageNet-style directory.
            target_dir (str): Path for the output database.
            max_images_per_folder (int, optional): Maximum images to process per 
                category. Defaults to 500.
        """
        os.makedirs(target_dir, exist_ok=True)
        
        folders = sorted([
            d for d in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, d))
        ])

        logger.info(f"Processing {len(folders)} folders, up to {max_images_per_folder} images each")

        for folder_idx, folder in enumerate(folders):
            logger.info(f"Processing folder {folder_idx + 1}/{len(folders)}: {folder}")

            source_folder = os.path.join(source_dir, folder)
            target_orig = os.path.join(target_dir, folder, "orig")
            target_attack = os.path.join(target_dir, folder, "attack")
            
            os.makedirs(target_orig, exist_ok=True)
            os.makedirs(target_attack, exist_ok=True)

            images = sorted([
                f for f in os.listdir(source_folder)
                if os.path.isfile(os.path.join(source_folder, f))
            ])[:max_images_per_folder]

            logger.info(f"Found {len(images)} images in folder {folder}")

            for idx, image in enumerate(images, 1):
                source_path = os.path.join(source_folder, image)
                pil_img = Image.open(source_path).convert("RGB")
                pil_img = self.transform(pil_img).unsqueeze(0)

                self.process_original_image(image, target_orig, pil_img)
                self.process_adversarial_image(
                    image, folder, folder_idx, target_attack, pil_img
                )
                logger.debug(f"Processed image {idx}/{len(images)}: {image}")


def main():
    """
    Main execution function for database creation.
    
    Uses settings.py for source and target directory configurations.
    """
    logger.info("Starting database creation process")
    try:
        creator = DatabaseCreator()
        creator.create_database(
            source_dir=settings.SOURCE_DIRECTORY,
            target_dir=settings.TARGET_DIRECTORY
        )
        logger.info("Database creation completed successfully")
    except Exception as e:
        logger.error(f"Database creation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()