from setuptools import setup, find_packages

setup(
    name="ultralytics",
    version="0.0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "yolo=ultralytics.__main__:main",
        ],
    },
)
