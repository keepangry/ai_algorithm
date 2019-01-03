#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-1-3 下午10:43
# @Author  : yangsen
# @Mail    : 0@keepangry.com
# @File    : character_2_word.py.py
# @Software: PyCharm

"""

共/B 同/E 创/B 造/E 美/B 好/E 的/S 新/S 世/B 纪/E —/B —/M 二/M ○/M ○/M 一/M 年/E 新/S 年/S 贺/B 词/E
==>
 共同  创造  美好  的  新  世纪  ——二○○一年  新  年  贺词


"""

import codecs
import sys


def character_2_word(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    # 4-tags for character tagging: B(Begin), M(Middle), E(End), S(Single)
    for line in input_data.readlines():
        if len(line) > 1:
            char_tag_list = line.strip().split()
            for char_tag in char_tag_list:
                char_tag_pair = char_tag.split('/')
                char = char_tag_pair[0]
                tag = char_tag_pair[1]
                if tag == 'B':
                    output_data.write(' ' + char)
                elif tag == 'M':
                    output_data.write(char)
                elif tag == 'E':
                    output_data.write(char + ' ')
                else:  # tag == 'S':
                    output_data.write(' ' + char + ' ')
            output_data.write("\n")
    input_data.close()
    output_data.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python character_2_word.py input output")
        sys.exit()
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    character_2_word(input_file, output_file)

