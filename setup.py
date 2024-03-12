import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    # 'cilog>=1.2.3',
]

setuptools.setup(
    name="atta",
    version="0.0.2",
    author="",
    author_email="",
    description=".",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='GPLv3',
    url=".",
    project_urls={
        "Bug Tracker": ".",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    package_dir={"ATTA": "ATTA"},
    install_requires=install_requires,
    entry_points = {
        'console_scripts': [
            # 'gttatg = ATTA.kernel.main:gttatg',
            'attatg = ATTA.kernel.alg_main:main',
            'attatl = ATTA.kernel.launch:launch'
        ]
    },
    python_requires=">=3.8",
)