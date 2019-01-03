# encoding: utf-8
'''
@author: yangsen
@license:
@contact: 0@keepangry.com
@software:
@file: character_tagging.py
@time: 18-10-2 下午1:03
@desc:

迈向  充满  希望  的  新  世纪  ——  一九九八年  新年  讲话  （  附  图片  １  张  ）
==>
迈/B 向/E 充/B 满/E 希/B 望/E 的/S 新/S 世/B 纪/E —/B —/E 一/B 九/M 九/M 八/M 年/E 新/B 年/E 讲/B 话/E （/S 附/S 图/B 片/E １/S 张/S ）/S


'''

import codecs
import sys

def character_tagging(input_file, output_file):
    input_data = codecs.open(input_file, 'r', 'utf-8')
    output_data = codecs.open(output_file, 'w', 'utf-8')
    for line in input_data.readlines():
        word_list = line.strip().split()
        for word in word_list:
            if len(word) == 1:
                output_data.write(word + "/S ")
            else:
                output_data.write(word[0] + "/B ")
                for w in word[1:len(word) - 1]:
                    output_data.write(w + "/M ")
                output_data.write(word[len(word) - 1] + "/E ")
        output_data.write("\n")
    input_data.close()
    output_data.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Please use: python character_tagging.py input output")
        sys.exit()
    input_file = sys.argv[1]
    output_file = sys.argv[2]
