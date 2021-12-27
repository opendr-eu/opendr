from urllib.request import urlretrieve
from opendr.engine.constants import OPENDR_SERVER_URL

url = OPENDR_SERVER_URL + "planning/end_to_end_planning/ardupilot.zip"
file_destination = "./ardupilot.zip"
urlretrieve(url=url, filename=file_destination)
