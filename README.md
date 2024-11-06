# Neural Fingerprints for Adversarial Attack Detection

Neural Fingerprints is a method for detecting adversarial attacks on deep neural networks. This project implements the key components of the Neural Fingerprints approach as described in the paper [Neural Fingerprints for Adversarial Attack Detection](https://arxiv.org/abs/2411.04533).

The module consists of two main components:
1. [Data base creator](#data-base-creator) - Generates a database of activations from the last layers of a neural network model, for both original images and adversarial examples.
2. [Fingerprints armor](#fingerprints-armor) - Implements the Neural Fingerprints technique to detect potential adversarial attacks by analyzing the activations in the database.

## Data base creator

The [Data Base Creator module](src/db_creator/) contains scripts and modules for creating adversarial examples activations database from the last layers of a neural network model. The goal is to build a database to test our adversarial fingerprint protection mechanism.

### Input Structure

The input should be structured as an ImageNet dataset:

```sh
images/
├── category_1/
│   ├── image_1.JPEG
│   ├── image_2.JPEG
│   └── ...
├── category_2/
│   ├── image_1.JPEG
│   ├── image_2.JPEG
│   └── ...
└── ...
```

### Output Format

The output will be pickle files containing the activations of the last X layers of the model for the original images and the adversarial examples generated from them.

Example directory structure:

```sh
database/
├── category_1/
│   ├── orig
│   │   ├── image_1.pkl
│   │   ├── image_2.pkl
│   │   └── ...
│   ├── attack
│   │   ├── image_1_0_to_1_ifgsm_3_01.pkl
│   │   ├── image_2_0_to_31_ifgsm_1_01.pkl
│   │   └── ...
├── category_2/
│   ├── orig
│   │   ├── image_1.pkl
│   │   ├── image_2.pkl
│   │   └── ...
│   ├── attack
│   │   ├── image_1_0_to_1_ifgsm_96_01.pkl
│   │   ├── image_2_0_to_78_ifgsm_9_01.pkl
│   │   └── ...
└── ...
```

Supported Attacks: The module supports generating adversarial examples using Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks.

### Files

#### [attack.py](src/db_creator/attack.py)
This module contains the `Attack` class, which provides methods for performing adversarial attacks on input tensors. It includes methods for Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks.

#### [get_last_layers_activations.py](src/db_creator/get_last_layers_activations.py)
This module contains the `ModelLastLayers` class, which is used to extract activations from the last X FC layers of a model. It registers hooks to capture the activations during the forward pass.

#### [db_creator.py](src/db_creator/db_creator.py)
The main script to create the DB. It uses the `ModelLastLayers` class to extract activations and the `Attack` class to generate adversarial examples.

### Exsample

see [db_creator_test](/tests/db_creator_test.py) for an exsample how to run the db_creator file.

## Fingerprints Armor

The folder [fingerprints_armor](src/fingerprints_armor/) contains scripts and modules that implement the Neural Fingerprints method to detect adversarial attacks. The goal is to create fingerprints such that they can be used by the following methods: vote, anomaly, and ll_ratio to detect patterns that match an adversarial attack.

The fingerprint method is implemented via [SingleClassFingerprintsArmor](src/fingerprints_armor/fingerprints.py#L44) to find the fingerprint for every class to detect the adversarial attack on it.

The input should be the output from [Data Base Creator](#data-base-creator) and the output will supply functions:
- **vote**: Predicted class labels based on the voting mechanism.
- **anomaly (likelihood)**: Anomaly score for each data point based on likelihood anomaly detection.
- **likelihood_ratio**: Likelihood ratio value used to determine if a data point is an adversarial attack or not.

These functions will take a DataFrame containing the activations from the last x layers of the model after a suspicious image has been run through it as input and will return detection results indicating whether the image is attacked or not.

### Files

#### [fingerprints_io.py](src/fingerprints_armor/fingerprints_io.py)
This module handles the input and output operations for processing the activations database.

#### [fingerprints.py](src/fingerprints_armor/fingerprints.py)
The main script that implements the activation fingerprint detection algorithm to identify potential adversarial attacks on neural networks. It uses statistical methods for fingerprint analysis.

### Example

See [fingerprints_armor_test](/tests/fingerprints_armor_test.py) for an example of how to run the fingerprints file and get the results.

In this test, each class tested will result in a CSV file with the following structure:

- **File Naming Convention**:
  Each CSV file is named after the class being analyzed (e.g., class_name.csv).

- **File Structure**:
  The CSV file contains the following columns:

  | y   | vote | anomaly | ll_ratio | cls     |
  | --- | ---- | ------- | -------- | ------- |
  | 0   | 0    | 0.2     | 1.5      | class_1 |
  | 1   | 1    | 0.3     | 2.1      | class_1 |
  | 0   | 0    | 0.1     | 1.2      | class_1 |
  | 1   | 1    | 0.4     | 2.5      | class_1 |
  | ... | ...  | ...     | ...      | ...     |

  Each row represents a data point, with the corresponding values for the actual label, predicted class label (vote), anomaly score, and likelihood ratio.

#### Aggregated Results:
After running the tests for all classes, the individual CSV files are concatenated into a single DataFrame (data). The aggregated results provide a comparison across multiple classes, showing how the defense mechanism performed in terms of voting, anomaly detection, and likelihood ratio.

#### Metric Columns:

- **vote**: Based on the voting mechanism (e.g., majority vote from multiple fingerprint models).
- **anomaly**: Anomaly scores that help identify unusual data points.
- **ll_ratio**: Likelihood ratio for detecting adversarial examples.
- **Ground Truth (y)**: The ground truth label (0 for original data and 1 for attacked data) to compare against the predicted values.

#### Confusion Matrix:
For each metric (vote, anomaly, and ll_ratio), the confusion matrix is calculated. The confusion matrix compares the predicted values (from vote, anomaly, or ll_ratio) against the actual ground truth (y). This helps evaluate the performance of the adversarial defense mechanism.

Confusion Matrix Structure:

|         | Predicted: 0        | Predicted: 1        |
| ------- | ------------------- | ------------------- |
| True: 0 | True Negative (TN)  | False Positive (FP) |
| True: 1 | False Negative (FN) | True Positive (TP)  |

Where:
- **True Negative (TN)**: The number of original data points classified correctly as original.
- **False Positive (FP)**: The number of attacked data points classified incorrectly as original.
- **False Negative (FN)**: The number of original data points classified incorrectly as attacked.
- **True Positive (TP)**: The number of attacked data points classified correctly as attacked.

### Quantile-Based Decision:
For each of the metrics (vote, anomaly, ll_ratio), the output is analyzed using a threshold based on the 1% quantile of the attack data. If a prediction for a particular metric exceeds this threshold, it is classified as an adversarial attack. This helps in assessing the robustness of the defense mechanism at different thresholds. 

## Configuration and Settings
The project uses a settings approach to manage configurations, allowing you to set paths, thresholds, and parameters directly in the code. This means you don’t have to specify configurations through command-line arguments each time you run the script, which simplifies setup and execution.

Open the [settings](src/settings.py) script and adjust the values as needed.

## Setup

### pip

Run:
```bash
pip install fingerprints-armor
```

Then use it as follow:
```python
import db_creator.db_creator
import fingerprints_armor.fingerprints
```

### Docker

To ensure a consistent and reproducible environment, a Dockerfile is provided. The Docker setup installs all necessary dependencies and sets up the environment for running the scripts.

#### Building the Docker Image

To build the Docker image, run the following command in the root directory of the repository:

```sh
docker build -t neural-fingerprints .
```

#### Running the Docker Container

To run the Docker container, use the following command:

```sh
docker run -v /path/to/imagenet:/image_net_data/ -v /path/to/output:/neural_fingerprints_db/ -it neural-fingerprints
```

Replace `/path/to/imagenet` with the path to your ImageNet dataset and `/path/to/output` with the path where you want to save the output DB.

## Running Without Docker

If you prefer not to use Docker, you can run the project directly on your local machine by setting up a Python environment and installing the required dependencies. Follow the steps below to get started.

1. Clone the Repository
Start by cloning the repository to your local machine:

```bash
git clone https://github.com/yourusername/neural-fingerprints.git
cd neural-fingerprints
```

2. Set Up a Virtual Environment
It's recommended to create a virtual environment to isolate the project dependencies. You can do this with venv (or virtualenv if preferred).

For Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install Dependencies
Once the virtual environment is activated, install the required dependencies using the requirements.txt file:
```bash
pip install -r requirements.txt
```
This will install all the necessary Python libraries, including any dependencies needed for adversarial attack detection.

4. Running the Project
After the dependencies are installed, you can run the project scripts directly.

To create the activations database (original and adversarial examples), use:
```bash
python src/db_creator/db_creator.py
```

To run the Neural Fingerprints defense on the database and detect adversarial attacks:
```bash
python src/fingerprints_armor/fingerprints.py
```

This will process the activations from the database and output detection results to the specified directory.
