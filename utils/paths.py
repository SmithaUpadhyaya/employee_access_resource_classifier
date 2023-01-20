from pathlib import Path

def get_project_root() -> Path:

    i = 3 #Since in my project structure are max at 3 level
    parent_path = Path()
    while i > 0:  

        if Path.exists(parent_path / 'params.yaml'):
            return parent_path
        else:
            parent_path = parent_path.resolve().parent
        
        i = i - 1

    return parent_path
    #return Path(__file__).parent.parent

def get_configuration_path() -> Path:
    return get_project_root() / "params.yaml"