"""Package setup for cough-vision TB detection system."""
from setuptools import find_packages, setup

# Runtime deps — the minimum required to import and run the pipeline.
# Torch/torchvision are intentionally kept in install_requires; see README
# for install instructions.
INSTALL_REQUIRES = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "timm>=0.9.0",
    "numpy>=1.24.0",
    "opencv-python>=4.8.0",
    "Pillow>=10.0.0",
    "pydicom>=2.4.0",
    "scikit-learn>=1.3.0",
    "onnx>=1.14.0",
    "onnxruntime>=1.15.0",
    "tqdm>=4.65.0",
    "matplotlib>=3.7.0",
]

setup(
    name="cough-vision",
    version="0.1.0",
    description=(
        "Clinically usable deep-learning system for pulmonary TB detection "
        "from chest X-rays. WHO TPP target: >=90% sensitivity / >=70% specificity."
    ),
    author="cough-vision team",
    python_requires=">=3.10",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=INSTALL_REQUIRES,
    extras_require={
        # pip install -e ".[dev]"
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
        # pip install -e ".[train]"
        "train": [
            "wandb>=0.15.0",
        ],
        # pip install -e ".[onnx]"  (optional graph optimisation)
        "onnx": [
            "onnxoptimizer>=0.3.0",
        ],
    },
)
