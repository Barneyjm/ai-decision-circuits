from setuptools import setup, find_packages

setup(
    name="langchain-robust-evaluator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-anthropic>=0.3.10",
        "python-dotenv>=1.0.0"
    ],
    author="James Barney",
    author_email="your.email@example.com",
    description="A robust evaluation system for LLM classification tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Barneyjm/langchain-robust-evaluator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
