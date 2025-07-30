import shutil
import logging
from pathlib import Path
import socket

import warnings
warnings.filterwarnings("ignore", message=r"^urllib3 v2 only supports OpenSSL")


def check_internet():
    print("Checking internet connection...")
    try:
        socket.setdefaulttimeout(1)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('1.1.1.1', 53))
        print("Internet connection detected.")
    except socket.error:
        raise Exception("STOP: No internet. Please connect before running this script.")
def delete_folder(folder_path):
    """Delete entire folder and all its contents if it exists."""
    folder = Path(folder_path)
    if folder.exists() and folder.is_dir():
        shutil.rmtree(folder)
        logging.info(f"Deleted folder and its contents: {folder}")
    else:
        logging.info(f"Folder '{folder}' does not exist or is not a directory. Nothing deleted.")