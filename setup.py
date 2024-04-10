#!/usr/bin/env python3
import setuptools

with open('requirements.txt', 'r') as f:
    requires = [x.strip() for x in f if x.strip()]

setuptools.setup(
    install_requires=requires,
    include_package_data=True
)



