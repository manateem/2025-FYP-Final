"""
Make sure everything works on everyone's machine

Add global constants here
"""
import os, sys

PROJECT_NAME = "2025-FYP-Final"
PROJECT_DIR = os.getcwd()
if PROJECT_NAME not in PROJECT_DIR:
    print(f"Wrong working directory! File must be ran from inside {PROJECT_NAME}")


while not PROJECT_DIR.endswith(PROJECT_NAME):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)


def p(path: str) -> str:
    """ Create absolute path based on a relative path to the project root """
    return os.path.join(PROJECT_DIR, path)
