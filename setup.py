from setuptools import setup, find_packages

#files = ["embedded_model/*", "ngconverter/*"]
#files = ["things/*"]

setup(
    name='ngconvert',
    version='0.0.1',
    packages=find_packages(exclude=["test", "data", "dependencies", "tianyuan"]),
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'ngconvert=ngconverter.commands.ngconvert:main'
        ]
    }
)
