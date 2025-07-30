import warnings
warnings.filterwarnings("ignore", message=r"^urllib3 v2 only supports OpenSSL")

import socket

def check_internet():
    print("Checking internet connection...")
    try:
        socket.setdefaulttimeout(1)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(('1.1.1.1', 53))
        print("Internet connection detected.")
    except socket.error:
        raise Exception("STOP: No internet. Please connect before running this script.")
