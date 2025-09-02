from setuptools import setup, find_packages

setup(
    name="sports_cv_project",         # Package name
    version="0.1.0",                  # Version number
    author="Your Name",
    author_email="youremail@example.com",
    description="Computer vision project for sports image classification/detection",
    packages=find_packages(),         # Automatically include all Python packages in the repo
    install_requires=[                # Dependencies
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy",
        "Pillow",
        "PyYAML",
        "tqdm",
        "scikit-learn",
        "matplotlib"
    ],
    python_requires=">=3.8",          # Minimum Python version
)
