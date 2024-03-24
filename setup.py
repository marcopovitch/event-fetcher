from setuptools import setup

setup(
    author="Marc Grunberg",
    license="GNU Lesser General Public License, Version 3 (LGPLv3)",
    platforms="OS Independent",
    name="eventfetcher",
    version="1.2",
    install_requires=[
        "obspy >= 1.4.0",
        "pyyaml",
    ],
)
