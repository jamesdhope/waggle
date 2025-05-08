from setuptools import setup, find_packages

setup(
    name="waggle",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "gym>=0.21.0",
        "acp-sdk>=0.8.1",
    ],
    author="James Hope",
    author_email="jamesdhope@gmail.com",
    description="A reinforcement learning extension for IBM Bee AI Agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jamesdhope/waggle",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 