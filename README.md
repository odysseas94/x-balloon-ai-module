# X-Balloon: AI-Powered Image Annotation and Segmentation

**X-Balloon** is an advanced AI framework designed to streamline image annotation, segmentation, and analysis, leveraging the power of Mask R-CNN for state-of-the-art performance. This system integrates a robust backend, a dynamic annotation module, and an AI processing application for seamless workflows across diverse use cases, including digital pathology and beyond.

## Features
- **Automated Segmentation:** Uses Mask R-CNN for accurate object detection and segmentation.
- **Interactive Annotation:** Intuitive tools for manual and semi-automated image annotation.
- **Scalable Backend:** Manages datasets, model weights, and configurations for training and testing.
- **Training and Inference:** Supports data augmentation, model evaluation, and performance metrics.

## Installation and Setup




### Prerequisites
- Python 3.x
- TensorFlow (GPU preferred for training)
- Keras
- Additional libraries as listed in the Mask R-CNN [GitHub repository](https://github.com/matterport/Mask_RCNN).


## installation
To install all the libraries, follow these steps:
```python
python -m pip install -r requirements.txt

```

also you need to copy
https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5
in the `app/datasets/mask_rcnn_coco.h5`

### Conifguration

set the `app\connection\UrlConnection.py` file with your backend URL:

```python
# URL for the backend server
domainUrl = "https://{your_domain}/index.php?r=api/v1/"
```


   
## Running the Application
### Training
To initiate training with X-Balloon's custom configurations: at the root directory of the project, run:

 ```bash
python -m app.src.Application train
  ```
This will:

Fetch datasets from the backend.
Apply configurations and data augmentations.
Train the model with Mask R-CNN using your dataset.
### Inference
To perform inference on test datasets: at the root directory of the project, run:

 ```bash
python -m app.src.Application detect
  ```

The AI application will:

Load the trained model weights.
Fetch test datasets from the backend.
Generate and upload segmentation results for validation.
### Workflow
Dataset Preparation: Manage and annotate datasets using the backend and annotation module.
Model Training: Train Mask R-CNN on the prepared datasets with custom configurations.
Inference and Evaluation: Use trained models to generate segmentation masks and analyze results.
## License
Copyright (C) 2025 Odysseas Tsakai

This project is licensed under the GNU General Public License v3 (GPLv3).
See <https://www.gnu.org/licenses/>.

This project includes code from the Mask R-CNN project developed by Matterport, Inc.,
which is licensed under the MIT License.

Third-party components retain their original licenses.

