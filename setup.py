import os
from setuptools import setup, find_packages

path = os.path.abspath(os.path.dirname(__file__))

readme = open(path + "/docs/README.md")

setup(
  name="modified-som",
  version="1.0.0",
  description="A modified version of SOM and LVQ for classification problems",
  url="https://github.com/KienMN/Modified-SOM",
  author="Kien MN",
  author_email="kienmn97@gmail.com",
  license="MIT",
  packages=find_packages(exclude=["docs","tests", ".gitignore"]),
  install_requires=["numpy", "pandas", "scikit-learn"],
  dependency_links=[""],
  include_package_data=True
)