import yaml
from typing import Union, Dict


def parse_yaml_file(file_path: str) -> Union[Dict, None]:
    """Parse a yaml file and return the content
    as a dictionary

    Args:
        file_path (str): Path to the yaml file

    Returns:
        Union[Dict, None]: Dict if the file was parsed successfully,
        None otherwise
    """
    with open(file_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(f"Error while parsing yaml file: {e}")
            return None
