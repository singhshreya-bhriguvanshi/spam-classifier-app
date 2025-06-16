from setuptools import setup

setup(
    name="spam-classifier",
    version="0.1.0",
    python_requires=">=3.10,<3.11",
    install_requires=[
        "Flask==2.3.2",
        "python-dotenv==1.0.1",
        "requests==2.32.3",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scikit-learn==1.3.2",
        "nltk==3.8.1"
    ],
) 