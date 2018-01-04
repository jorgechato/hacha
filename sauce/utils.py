# -*- coding: utf-8 -*-
"""
Utils file

Author:  Jorge Chato
Repo:    github.com/jorgechato/sauce
Web:     jorgechato.com
"""
import sys


def print_info():
    print("\nsauce v0.1 (http://github.com/jorgechato/sauce)")

def print_argument_error():
    print("No arguments provided. Execute '" + sys.argv[0] + " -h' for help")
    exit()
