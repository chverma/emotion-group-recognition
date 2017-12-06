# !/usr/bin/python
import sys
import os
import requests


def submitToServer(name, filePath):
    url = 'http://localhost:5001/checkIdentity'
    files = {'file': open(filePath, 'rb')}
    r = requests.post(url, files=files, params={"name": name})
    print r.text


for root, dirs, files in os.walk(sys.argv[1], topdown=False):
    cDir = ''
    for dirname in dirs:
        cDir = dirname
    for name in files:
        print(os.path.join(root, name))
        submitToServer(root.split('/')[-1], os.path.join(root, name))
        break
