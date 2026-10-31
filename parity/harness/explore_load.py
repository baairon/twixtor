#!/usr/bin/env python3
"""Loads one plugin under Qiling and prints what the loader reports (missing DLLs). usage: explore_load.py ROOTFS PLUGIN"""
import sys
from qiling import Qiling
from qiling.const import QL_VERBOSE

rootfs, plugin = sys.argv[1], sys.argv[2]
try:
    ql = Qiling([plugin], rootfs, verbose=QL_VERBOSE.DEFAULT, libcache=False)
    print("LOADED", hex(ql.loader.pe_image_address), "images", len(ql.loader.images))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("FAILED", type(e).__name__, e)
