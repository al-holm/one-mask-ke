from setuptools import find_packages, setup

setup(
    name="revit",
    version="0.1.0",
    description="Pipiline: edit, prune, evaluate",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
