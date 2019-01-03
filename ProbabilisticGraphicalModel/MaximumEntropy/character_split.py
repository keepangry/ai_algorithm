#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-1-3 下午10:41
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : character_split.py.py
# @Software: PyCharm

"""

共同创造美好的新世纪——二○○一年新年贺词
==>
共 同 创 造 美 好 的 新 世 纪 — — 二 ○ ○ 一 年 新 年 贺 词

"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 52nlpcn@gmail.com
# Copyright 2014 @ YuZhen Technology
#
# split chinese characters and add space between them

import codecs
import sys


def character_split(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        for word in line.strip():
            output_data.write(word + " ")
        output_data.write("\n")
    input_data.close()
    output_data.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Please use: python character_split.py input output")
        sys.exit()
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    character_split(input_file, output_file)
