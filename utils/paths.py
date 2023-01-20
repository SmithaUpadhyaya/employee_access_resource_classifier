from pathlib import Path

def get_project_root() -> Path:
    parent_path = Path()
    return parent_path.resolve().parent
    #return Path(__file__).parent.parent

def get_configuration_path() -> Path:
    return get_project_root() / "params.yaml"