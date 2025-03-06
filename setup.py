# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess

import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension


class CustomBuildExt(build_ext):
    def run(self):
        # Use bash explicitly instead of relying on SHELL environment variable
        shell = "/bin/bash"

        # Define script paths
        scripts = [
            os.path.join(
                os.path.dirname(__file__),
                "mithril",
                "cores",
                "c",
                "raw_c",
                "compile.sh",
            ),
            os.path.join(
                os.path.dirname(__file__),
                "mithril",
                "cores",
                "c",
                "ggml",
                "build_ggml.sh",
            ),
            os.path.join(
                os.path.dirname(__file__),
                "mithril",
                "cores",
                "c",
                "ggml",
                "compile.sh",
            ),
        ]

        print("Running compilation scripts...")

        # Save current working directory
        original_dir = os.getcwd()

        try:
            # Run each script from its own directory
            for script_path in scripts:
                script_dir = os.path.dirname(script_path)
                script_name = os.path.basename(script_path)
                print(f"Running {script_name} in {script_dir}")
                os.chdir(script_dir)
                subprocess.check_call([shell, f"./{script_name}"])
                os.chdir(original_dir)  # Return to original directory
        finally:
            # Make sure we return to the original directory even if an error occurs
            os.chdir(original_dir)

        # Continue with the normal build
        super().run()


with open("README.md") as f:
    long_description = f.read()

setuptools.setup(
    name="mithril",
    version="0.1.1",
    author="Synnada, Inc.",
    author_email="opensource@synnada.ai",
    description="A Modular Machine Learning Library for Model Composability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/synnada-ai/mithril",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=[],
    cmdclass={"build_ext": CustomBuildExt},
    ext_modules=[Extension("mithril.c_sources", sources=[])],
    include_package_data=True,
)
