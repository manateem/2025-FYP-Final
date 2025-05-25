"""
Move files listed in result/testing_data.txt from data/ to data/test, and
the files listed in result/training_data.txt from data/ to data/training
"""
import os, shutil

PROJECT_DIR = os.getcwd()
# find project directory
while not PROJECT_DIR.endswith("2025-FYP-Final"):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)

TESTING_DATA_FOLDER = os.path.join(
    PROJECT_DIR,
    "data/test/"
)
TRAINING_DATA_FOLDER = os.path.join(
    PROJECT_DIR,
    "data/training/"
)

if not os.path.exists(TESTING_DATA_FOLDER):
    os.makedirs(TESTING_DATA_FOLDER)

if not os.path.exists(TRAINING_DATA_FOLDER):
    os.makedirs(TRAINING_DATA_FOLDER)

def move_training_files():
    with open("result/training_data.txt", 'r', encoding="utf-8") as training_data:
        for training_image in training_data.readlines():
            training_image = training_image.rstrip('\n').rstrip() + ".png"
            training_image_abspath = os.path.join(
                PROJECT_DIR,
                "data",
                training_image
            )

            shutil.move(
                training_image_abspath,
                TRAINING_DATA_FOLDER
            )

            print(f"moved {training_image} to {TRAINING_DATA_FOLDER}")


def move_test_files():
    with open("result/testing_data.txt", 'r', encoding="utf-8") as testing_data:
        for testing_image in testing_data.readlines():
            testing_image = testing_image.rstrip('\n').rstrip() + ".png"
            testing_image_abspath = os.path.join(
                PROJECT_DIR,
                "data",
                testing_image
            )

            shutil.move(
                testing_image_abspath,
                TESTING_DATA_FOLDER
            )

            print(f"moved {testing_image} to {TESTING_DATA_FOLDER}")


if __name__ == "__main__":
    print("moving training files....")
    move_training_files()
    print("moving test files....")
    move_test_files()
