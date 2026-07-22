"""Package setup for cough-vision TB detection system."""
from setuptools import find_packages, setup

setup(
    name="cough-vision",
    version="0.1.0",
    description="Clinically usable deep-learning system for pulmonary TB detection from chest X-rays",
    author="cough-vision team",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
        "pydicom>=2.4.0",
        "scikit-learn>=1.3.0",
        "onnx>=1.14.0",
        "captum>=0.6.0",
        "grad-cam>=1.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        "train": [
            "wandb>=0.15.0",
            "tqdm>=4.65.0",
        ],
    },
)
