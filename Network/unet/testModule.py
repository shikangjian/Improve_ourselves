'''
from skimage import io
import os

root = '/data/caozheng/caries/dataset/train_set/masks'
fpath = os.path.join(root, os.listdir(root)[0])

img = io.imread(fpath)

img[:,:][img[:,:] > 0] = 1

'''

# ts = 100

# 消息推送
'''
import requests
api = 'https://sc.ftqq.com/SCU90678Ted3c83ea40302ad405225b95fc2d22bb5e775c15857eb.send'
title = '训练完成！'
content = '训练完成，使用%d秒' % ts

data = {
    "text":title,
    "desp":content
}
req = requests.post(api, data=data)
print('End, Message Send, %s' % req)
'''

from pynvml import *
import requests
import time
import sys

class GPUStat:
    def __init__(self):
        nvmlInit()
        self.deviceCount = nvmlDeviceGetCount()
        self.var = 1
        super()

    def test(self):
        print(self.var)

    def lookMem(self, id, out=False):
        handle = nvmlDeviceGetHandleByIndex(id)
        info = nvmlDeviceGetMemoryInfo(handle)
        gpuname =  nvmlDeviceGetName(handle)
        tmp = nvmlDeviceGetTemperature(handle, 0)
        mem = info.free / 1024**2
        freemem = '{:.2f}'.format(info.free/info.total*100)
        if out:
            # print('GPU Index:', id, 'GPU Name:', gpuname,
            #       'Free Memory:', mem, 'Mem left(%):', freemem)
            print('GPU Index:{}, GPU:{}, Free Memory:{}M,\tMem left: {}%'.format(id, gpuname, mem, freemem))
            return id, gpuname, mem, freemem
        else:
            return id, gpuname, mem, freemem

    def monitor(self):
        print('Attempt to get GPUstat')
        for i in range(self.deviceCount):
            id, gpuname, mem, freemem = self.lookMem(i, out=True)
            if float(freemem) > 50:
                self.server_post(id, gpuname, mem, freemem)

    def server_post(self, id, gpuname, mem, freemem):
        print('Find Usable GPU! Sending Message...')
        api = 'https://sc.ftqq.com/SCU90678Ted3c83ea40302ad405225b95fc2d22bb5e775c15857eb.send'
        title = '发现空余GPU'
        content = 'GPU Index:{}, GPU:{}, Free Memory:{}M,\tMem left: {}%'.format(id, gpuname, mem, freemem)
        data = {
            "text": title,
            "desp": content
        }
        req = requests.post(api, data=data)
        print('Send Message Done, %s' % req.text)
        print('Program Reactivate after 1 hour!')
        time.sleep(60*60)


if __name__ == '__main__':
    gpu = GPUStat()
    while (1):
        gpu.monitor()
        time.sleep(10)

