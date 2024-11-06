"""
Neural Network Fingerprint Detection Module.

This module implements activation fingerprint detection algorithm to identify
potential adversarial attacks on neural networks. It uses statistical methods
for fingerprint analysis.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, norm

from src.logger_config import get_logger

logger = get_logger(__name__)


@dataclass
class Fingerprint:
    """
    Represents a fingerprint generated from activation data of an image.

    Attributes:
        selected_activations (List[float]): The activations selected to create the fingerprint,
            representing the activations from the original data.
        original_means (np.ndarray): The mean activation values of the selected features
            from the original data.
        adversarial_means (np.ndarray): The mean activation values of the selected features
            from the adversarial data.
        original_stats (Tuple[float, float]): A tuple containing the mean and standard
            deviation of the original means.
        adversarial_stats (Tuple[float, float]): A tuple containing the mean and standard
            deviation of the adversarial means.
    """
    selected_activations: List[float]
    original_means: np.ndarray
    adversarial_means: np.ndarray
    original_stats: Tuple[float, float]  # (mean, std)
    adversarial_stats: Tuple[float, float]  # (mean, std)

    
class SingleClassFingerprintsArmor:
    """
    Detects and analyzes neural network activation fingerprints.
    
    Implements methods for generating, filtering, and using activation patterns
    as fingerprints to detect potential adversarial attacks on neural networks.

    Attributes:
        _activations_per_fingerprint (int): Number of activation features to use in each fingerprint
        _fingerprints (List): List of tuples containing fingerprint data
    """


    def __init__(self, activations_per_fingerprint: int = 50):
        """
        Initialize the activation fingerprint detector.

        Args:
            activations_per_fingerprint (int, optional): Number of activation features 
                to use in each fingerprint. Defaults to 50.
        """
        self._activations_per_fingerprint = activations_per_fingerprint
        self._fingerprints: List[Fingerprint] = []
        logger.info(f"Initialized detector using {activations_per_fingerprint} activations per fingerprint")

    def generate_fingerprints(self,
                              original_data: pd.DataFrame,
                              adversarial_data: pd.DataFrame, 
                              num_samples: int) -> None:
        """
        Generate fingerprints from activation data.

        Args:
            original_data (pd.DataFrame): Original images activation
            adversarial_data (pd.DataFrame): Adversarial images activation
            num_samples (int): Number of samples to generate
        """
        logger.info(f"Generating {num_samples} fingerprint samples")
        self._fingerprints = []

        for _ in range(num_samples):
            # Randomly select fingerprint activations
            selected_activations = np.random.choice(
                original_data.columns.values, 
                self._activations_per_fingerprint, 
                replace=False
            )
            
            # Calculate mean activations across selected features
            original_means = original_data.loc[:, selected_activations].mean(axis=1)
            adversarial_means = adversarial_data.loc[:, selected_activations].mean(axis=1)
            
            # Store fingerprint statistics
            self._fingerprints.append(Fingerprint(
                selected_activations=selected_activations,
                original_means=original_means,
                adversarial_means=adversarial_means,
                original_stats=(original_means.mean(), original_means.std()),
                adversarial_stats=(adversarial_means.mean(), adversarial_means.std())
            ))
        
        logger.debug(f"Generated {len(self._fingerprints)} fingerprint samples")

    @staticmethod
    def _compute_abs_d_effect_size(fingerprint: Fingerprint) -> float:
        """
        Calculate absolute Cohen's d effect size for a fingerprint.

        Args:
            fingerprint (Tuple): Fingerprint tuple containing activation statistics

        Returns:
            float: Absolute Cohen's d effect size value
        """
        orig_sample_size  = fingerprint.original_means.shape[0]
        adv_sample_size  = fingerprint.adversarial_means.shape[0]
        orig_mean, orig_std  = fingerprint.original_stats
        adv_mean, adv_std  = fingerprint.adversarial_stats

        # Calculate pooled standard deviation
        pooled_std = np.sqrt(
            ((orig_sample_size - 1) * (orig_std ** 2) + 
             (adv_sample_size - 1) * (adv_std ** 2)) / 
            (orig_sample_size + adv_sample_size - 2)
        )

        absolute_cohens_d = abs((orig_mean - adv_mean) / pooled_std)
        return absolute_cohens_d
    
    def filter_by_significance(self,
                               p_value_threshold: float = 0.01, 
                               min_effect_size: float = 0.01) -> None:
        """
        Filter fingerprints using statistical significance and effect size.

        Args:
            p_value_threshold (float, optional): Maximum p-value threshold. Defaults to 0.01.
            min_effect_size (float, optional): The minimum absolute Cohen's d effect size. Defaults to 0.01.
        """
        initial_count = len(self._fingerprints)

        # Filter fingerprints based on t-test results
        self._fingerprints = [
            fingerprint for fingerprint in self._fingerprints
            if ttest_ind(fingerprint.original_means, fingerprint.adversarial_means)[1] < p_value_threshold
        ]
        
        # Further filter fingerprints based on absolute Cohen's d effect size
        self._fingerprints = [
            fingerprint for fingerprint in self._fingerprints
            if self._compute_abs_d_effect_size(fingerprint) >= min_effect_size
        ]

        logger.info(f"Filtered fingerprints: {initial_count} -> {len(self._fingerprints)}")

    def filter_top_k(self, k: int) -> None:
        """
        Keep only the top performing fingerprints based on effect size.

        Args:
            k (int): Number of fingerprints to save
        """
        if len(self._fingerprints) <= k:
            logger.debug(f"Skipping top filtering: {len(self._fingerprints)} <= {k}")
            return
            
        effect_sizes = [self._compute_abs_d_effect_size(f) for f in self._fingerprints]
        top_indices = np.argsort(effect_sizes)[::-1][:k]
        self._fingerprints = [self._fingerprints[i] for i in top_indices]
        logger.info(f"Saved top {k} fingerprints")

    def reduce_top_k(self, new_k: int) -> None:
        """
        Reduce top k fingerprints to a new k

        Args:
            new_k (int): New number of fingerprints to save
        
        NOTE: This makes sense only if filter_top_k was used in fit
        """
        if len(self._fingerprints) >= new_k:
            logger.debug(f"Skipping reduce top k filtering: {len(self._fingerprints)} <= {new_k}")
            return

        self._fingerprints = self._fingerprints[:new_k]
    
    def fit(self,
            original_data: pd.DataFrame,
            adversarial_data: pd.DataFrame, 
            num_samples: int,
            apply_significance_filter: bool = True, 
            p_value_threshold: float = 0.01,
            min_effect_size: float = 0.1,
            apply_top_filter: bool = False,
            top_count: int = 300) -> None:
        """
        Generate and filter fingerprints from training data.

        Args:
            original_data (pd.DataFrame): Original activation data
            adversarial_data (pd.DataFrame): Adversarial activation data
            num_samples (int): Number of fingerprints samples to generate
            apply_significance_filter (bool, optional): Whether to apply statistical filtering. 
                Defaults to True.
            p_value_threshold (float, optional): Maximum p-value for significance. 
                Defaults to 0.01.
            min_effect_size (float, optional): Minimum effect size threshold. 
                Defaults to 0.1.
            apply_top_filter (bool, optional): Whether to keep only top fingerprints. 
                Defaults to False.
            top_count (int, optional): Number of top performing fingerprints to keep. 
                Defaults to 300.
        """
        if apply_top_filter and num_samples < top_count:
            raise ValueError("num_samples must be greater than or equal to top_count when apply_top_filter is True.")
        
        self.generate_fingerprints(original_data, adversarial_data, num_samples)

        if apply_significance_filter:
            self.filter_by_significance(p_value_threshold, min_effect_size)
        
        if apply_top_filter:
            self.filter_top_k(top_count)

    def vote(self, activations: pd.DataFrame) -> np.ndarray:
        """
        Calculate voting scores for each activation pattern.
        
        Computes the ratio of fingerprints that are closer to the original
        pattern than to the attack pattern.
        
        Args:
            activations (pd.DataFrame): DataFrame containing activation patterns
                to analyze. Columns should match fingerprint indices.
        
        Returns:
            np.ndarray: Array of voting scores between 0 and 1, where higher values
                indicate greater similarity to original patterns.
        """
        votes = []

        for fingerprint in self._fingerprints:
            # Extract relevant features for this fingerprint
            this_fingerprint = activations.loc[:, fingerprint.selected_activations].mean(axis=1)
            orig_mean, orig_std = fingerprint.original_stats
            adv_mean, adv_std = fingerprint.adversarial_stats

            # Compare normalized distances to original vs adversarial patterns
            this_vote = np.abs((this_fingerprint - orig_mean) / orig_std) <= \
                       np.abs((this_fingerprint - adv_mean) / adv_std)
            votes.append(this_vote)

        return np.array(votes).mean(axis=0)
    
    def hard_vote(self, activations: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Perform hard voting classification based on voting scores.
        
        Args:
            activations (pd.DataFrame): DataFrame containing activation patterns
                to analyze. Columns should match fingerprint indices.
            threshold (float, optional): Classification threshold. 
                Defaults to 0.5.
        
        Returns:
            np.ndarray: Binary array where 1 indicates pattern classified as
                original and 0 as attacked.
        """
        return (self.vote(activations) > threshold).astype(int)
    
    @staticmethod
    def _log_likelihood(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """
        Calculate log likelihood of samples under a normal distribution.
        
        Args:
            x (np.ndarray): Input samples to evaluate
            mu (float): Mean of the normal distribution
            sigma (float): Standard deviation of the normal distribution
        
        Returns:
            np.ndarray: Log likelihood values for each input sample
        """
        return norm.logpdf(x, mu, sigma)

    def likelihood(self, activations: pd.DataFrame) -> np.ndarray:
        """
        Calculate log likelihood of patterns under original distribution.
        
        Args:
            activations (pd.DataFrame): DataFrame containing activation patterns
                to analyze. Columns should match fingerprint indices.
        
        Returns:
            np.ndarray: Log likelihood scores for each pattern under original
                distribution.
        """
        log_likelihood_orig = 0
        
        for fingerprint in self._fingerprints:
            this_fingerprint = activations.loc[:, fingerprint.selected_activations].mean(axis=1)
            orig_mean, orig_std = fingerprint.original_stats
            log_likelihood_orig += self._log_likelihood(this_fingerprint, orig_mean, orig_std)
            
        return log_likelihood_orig

    def likelihood_ratio(self, activations: pd.DataFrame) -> np.ndarray:
        """
        Calculate log likelihood ratio between original and adversarial distributions.
        
        A positive ratio indicates the pattern is more likely from the original
        distribution, while a negative ratio suggests it's more likely from the
        adversarial distribution.
        
        Args:
            activations (pd.DataFrame): DataFrame containing activation patterns
                to analyze. Columns should match fingerprint indices.
        
        Returns:
            np.ndarray: Log likelihood ratios for each pattern. Positive values
                indicate greater similarity to original patterns.
        """
        log_likelihood_orig = 0
        log_likelihood_adv = 0
        
        for fingerprint in self._fingerprints:
            this_fingerprint = activations.loc[:, fingerprint.selected_activations].mean(axis=1)
            
            orig_mean, orig_std = fingerprint.original_stats
            adv_mean, adv_std = fingerprint.adversarial_stats
            
            log_likelihood_orig += self._log_likelihood(this_fingerprint, orig_mean, orig_std)
            log_likelihood_adv += self._log_likelihood(this_fingerprint, adv_mean, adv_std)
            
        return log_likelihood_orig - log_likelihood_adv
