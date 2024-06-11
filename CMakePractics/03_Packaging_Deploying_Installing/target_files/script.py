#!/usr/bin/env python3

import os

file_path = os.path.dirname(__file__)

try:
    with open(str(file_path + "/script")) as file:
        print(file.readlines())
except Exception as e:
    print(e)
