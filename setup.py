from setuptools import setup, find_packages

# It's good practice to read dependencies from requirements.txt
# For simplicity here, we'll list a few common ones.
# You should populate this list with all dependencies from your requirements.txt
# or read them dynamically from the file.
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="RAG_BOT",
    version="0.1.0",
    author="Anupam Yadav", 
    author_email="your.email@example.com", 
    description="A Telegram RAG agent for spiritual questions.", 
    long_description=open('README.md').read(), # Optional: Uses your README
    long_description_content_type="text/markdown", # Optional
    url="https://github.com/bk-anupam/LLM_agents", # Optional: URL to your project (e.g., GitHub repo)
    packages=find_packages(include=['RAG_BOT', 'RAG_BOT.*']), # Finds RAG_BOT and its submodules
    install_requires=required, # List your project's dependencies here
    classifiers=[ # Optional: Trove classifiers
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # If you have a license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10', # Specify your Python version requirement
)