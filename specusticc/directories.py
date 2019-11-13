def get_timestamp_dir() -> str:
    from datetime import datetime
    now = datetime.now()
    save_dir = now.strftime('%Y-%m-%d_%H-%M-%S')
    return save_dir


def create_save_dir(save_dir: str) -> None:
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
