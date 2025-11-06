import time
import random 

def 主体函数():
    开始时间 = time.time()
    打印数量 = 0
    上一次数量 = 0
    while True:
        print(上一次数量)
        打印数量 += 1
        结束时间 = time.time()
        if 结束时间 - 开始时间 >= 1:
            上一次数量 = 打印数量
            开始时间 = 结束时间
主体函数()