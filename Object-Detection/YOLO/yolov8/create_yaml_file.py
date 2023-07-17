import os
import glob
import yaml
import re
import argparse


def write_to_yaml_file(
        images_dir: str,
        class_names: list[str],
        yaml_path: str) -> None:
    """Creates a Yaml file with the info of the images_dir, 
        class_names and yaml_path

    Args:
        images_dir (str): dir path containing subdirectories with images
        class_names (list[str]): list of class names.
        yaml_path (str): Path of the Yaml file to be created
    """

    train_dir = os.path.join(images_dir, "train")
    test_dir = os.path.join(images_dir, "test")
    val_dir = os.path.join(images_dir, "validation")

    class_number = len(class_names)

    with open(yaml_path, "w") as f:
        yaml.dump({
            "train": train_dir,
            "test": test_dir,
            "validation": val_dir,
            "no_of_class": class_number,
            "names": class_names,  }, f)
        
    
def read_class_names_from_file(class_name_file:str) -> list:
    """Open existing file and extract class names

    Args:
        class_name_file (str): The name of the file with the names

    Returns:
        _type_: list of class names 
    """

    assert os.path.isfile(
        class_name_file), f" class name file {class_name_file} does not exist"
    
    with open(class_name_file, "r") as f:
        file_content = f.read()

    class_names = file_content.split("\n")

    return class_names


def get_max_class_number(labels_dir: str) -> int:
    """Get the labels directory mathch a regex and return 
    the maximum class number

    Args:
        labels_dir (str): dir name of the labels file

    Returns:
        int: the maximum class number
    """
    max_class_number = 0
    label_glob = glob.glob(os.path.join(labels_dir, "**", "*.txt"))
    pattern =r"^(\d)+([0-9]+\.?[0-9]*){4}$"

    for label_file in label_glob:
        with open(label_file, "r") as f:
            for line in f:
                matches = re.match(pattern, line)
                if matches:
                    max_class_number = max(max_class_number, matches.group(1))


def create_yaml(
        model_dir: str,
        images_dir: str,
        labels_dir: str):
    """Creates the yaml file with the labels and the dir paths

    Args:
        model_dir (str): dir to store model files
        images_dir (str): dir to store images
        labels_dir (str): dir to store labes
    """
    model_dir = os.path.abspath(model_dir)
    images_dir = os.path.abspath(images_dir)
    labels_dir = os.path.abspath(labels_dir)
    class_names_file = os.path.join(model_dir, "classes.txt")

    if os.path.isfile(class_names_file):
        class_names = list(read_class_names_from_file(class_names_file))
    else:
        max_class_number = get_max_class_number(labels_dir=labels_dir)
        class_names = list(map(lambda x: "Default" + str(x), range(max_class_number)))

    yaml_path = os.path.join(model_dir, "data.yaml")
    write_to_yaml_file(images_dir = images_dir,
              class_names = class_names,
              yaml_path = yaml_path)
    print("Yaml file is created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "create YAML for Yolov8")
    parser.add_argument('--model_dir', required=True, help='dir to store model files')
    parser.add_argument('--images_dir', required=True, help='dir to store images')
    parser.add_argument('--labels_dir', required=True, help='dir to store labels')

