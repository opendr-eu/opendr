from pathlib import Path


def get_or_create_dir(dir_str: str, folder) -> str:
    dir_path = Path(dir_str, folder)
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
    return str(dir_path)
