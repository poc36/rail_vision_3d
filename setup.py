from setuptools import setup, find_packages

setup(
    name="rail_vision_3d",
    version="0.1.0",
    description="3D Rail Detection System — Computer Vision for Robot",
    author="Rail Vision Team",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "open3d>=0.17.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0"],
    },
)
