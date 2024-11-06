from setuptools import setup, find_packages

setup(
    name="fingerprints-armor",
    version="1.0.0",
    description="A package implementing the NEURAL FINGERPRINTS FOR ADVERSARIAL ATTACK DETECTION",
    author="Haim D. Fisher, Yehezkel S. Resheff, Moni Shahar",
    author_email="haimdfisher@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.1.2",
        "torch>=2.5.0",
        "cleverhans>=4.0.0",
        "Pillow>=11.0.0",
        "timm>=1.0.11",
        "pandas>=2.2.3",
        "scikit-learn>=1.5.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
        ],
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HaimFisher/fingerprints-armor",
    project_urls={
        "Homepage": "https://github.com/HaimFisher/fingerprints-armor",
        "Documentation": "https://github.com/HaimFisher/fingerprints-armor/blob/main/README.md",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)