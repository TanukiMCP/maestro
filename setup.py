# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

from setuptools import setup, find_packages
import json

# Read package.json for metadata
with open('package.json', 'r') as f:
    package_data = json.load(f)

# Read requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name=package_data['name'],
    version=package_data['version'],
    description=package_data['description'],
    author=package_data['author'],
    license=package_data['license'],
    url=package_data['repository'],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.9',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'maestro=main:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords=package_data['keywords'],
    project_urls={
        'Bug Reports': package_data['bugs'],
        'Source': package_data['repository'],
    },
) 
