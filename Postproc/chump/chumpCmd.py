# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 04:10:33 2020

@author: mou@sspa.com
"""

import argparse
import chump
import chump.operations as operations
from chump.config import config
import openpyxl              # explicit import for pyinstaller
import openpyxl.cell._writer # explicit import for pyinstaller
openpyxl.cell._writer.__package__

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=chump._cmd_head_)
    parser.add_argument("command",  help='CHUMP command: prepare, extract stat or pdf(plot)')
    parser.add_argument("config",   help='CHUMP configuration file', )
    parser.add_argument("-o", "--only",   help='the elements to gernerate', default=[], action='append', nargs='*')
    parser.add_argument("-v", "--verbose",help='verbose level, 0 is silent. Default is 1', default=1, )

    args = parser.parse_args()
    # step 1. read model setting/config
    this_config = config(args)

    # step 2. execute command
    command = args.command.strip().lower()
    onlys = []
    for o in args.only:
        onlys += o

    opts = {c:getattr(operations, c) for c in dir(operations) if not c.startswith('_')}
    if command not in opts:
        raise ValueError(f'Unknown commands: {command}.')

    for p in (onlys or this_config.dict[command]):
        this_config.runCmd(command, p, )
