from setuptools import setup, find_packages

setup(
    name="neurologics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A brief description of your project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neurologics",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
