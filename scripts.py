"""This module provides scripts for the repo-map tool."""
import subprocess


def run_format():
    """
    Run black to format the source code.
    """
    subprocess.run(["black", "src"], check=True)
