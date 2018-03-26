__author__ = 'jnejati'

import apache_conf
from urllib.parse import urlparse
import os
import shutil
from bs4 import BeautifulSoup
import urllib.request
import urllib.response
import io
import subprocess


def main():
    input_file = 'mixed200.txt'
    arch_dir = '/home/jnejati/PLTSpeed/arch_dir2'
    #exp_type = 'compression'
    apache_conf.rm_dir(arch_dir)
    apache_conf.fetch_all_sites(input_file, arch_dir)


if __name__ == '__main__':
    main()
