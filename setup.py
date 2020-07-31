from setuptools import setup, find_packages
import shutil, os
from ngconverter.core.constants import EMBEDDED_SSD_PIPELINE_CONFIG_PATH
from ngconverter.util.filesystem import try_makedirs

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

try_makedirs(os.path.dirname(EMBEDDED_SSD_PIPELINE_CONFIG_PATH))
shutil.copy("embedded_model/empty_ssd_pipeline.config",EMBEDDED_SSD_PIPELINE_CONFIG_PATH)
