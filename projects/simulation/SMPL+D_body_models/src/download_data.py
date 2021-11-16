from opendr.engine.constants import OPENDR_SERVER_URL
from urllib.request import urlretrieve
import tarfile
import os
import time
import sys

OPENDR_HOME = os.environ["OPENDR_HOME"]


def download_data(raw_data_only):
    def reporthook(count, block_size, total_size):
        nonlocal start_time
        nonlocal last_print

        if count == 0:
            start_time = time.time()
            last_print = start_time
            return

        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        if time.time() - last_print >= 1:
            last_print = time.time()
            print(
                "\r%d MB, %d KB/s, %d seconds passed" %
                (progress_size / (1024 * 1024), speed, duration),
                end=''
            )

    human_data_url = OPENDR_SERVER_URL + "simulation/SMPLD_body_models/human_data.tar.gz"
    downloaded_human_data_path = os.path.join(OPENDR_HOME, 'projects/simulation/SMPL+D_body_models/human_data.tar.gz')
    print("Downloading data from", human_data_url, "to", downloaded_human_data_path)
    start_time = 0
    last_print = 0
    urlretrieve(human_data_url, downloaded_human_data_path, reporthook=reporthook)
    with tarfile.open(downloaded_human_data_path) as tar:
        tar.extractall(path=os.path.join(OPENDR_HOME, 'projects/simulation/SMPL+D_body_models'))
    tar.close()
    os.remove(downloaded_human_data_path)

    if raw_data_only:
        return

    model_url = OPENDR_SERVER_URL + "simulation/SMPLD_body_models/model.tar.gz"
    downloaded_model_path = os.path.join(OPENDR_HOME, 'projects/simulation/SMPL+D_body_models/model.tar.gz')
    print("Downloading data from", model_url, "to", downloaded_model_path)
    start_time = 0
    last_print = 0
    urlretrieve(model_url, downloaded_model_path, reporthook=reporthook)
    with tarfile.open(downloaded_model_path) as tar:
        tar.extractall(path=os.path.join(OPENDR_HOME, 'projects/simulation/SMPL+D_body_models'))
    tar.close()
    os.remove(downloaded_model_path)

if __name__ == "__main__":
    raw_data = False
    if len(sys.argv) > 1 and sys.argv[1] == 'raw':
        raw_data = True
    download_data(raw_data)
