#!/usr/bin/env python
# coding: utf-8

# 更大胆的想法
# 抠图 将视频中人物抠出来放到背景中去
# 

# 一、将原视频中的每一帧提取出来并保存为图片在transv_result文件夹中（文件夹提前建好）

# In[ ]:


# 这里将视频的每一帧都提取出来了
import cv2
from tqdm import tqdm
video = cv2.VideoCapture("video.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("总共的帧数为：",frameCount)
success, frame = video.read() 
index = 0
for i in tqdm(range(int(frameCount)),desc='处理进度'):
    if success:
        cv2.imwrite('transv_result/'+str(index)+'.jpg', frame)
    success, frame = video.read()
    index += 1


# 二、显示待预测图片，检测是否第一步已成功

# In[ ]:


# 待预测图片
img_name = '0.jpg'
#test_img_path = ["./"+img_name]
test_img_path = ["transv_result/0.jpg"]

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

img = mpimg.imread(test_img_path[0]) 

# 展示待预测图片
plt.figure(figsize=(10,10))
plt.imshow(img) 
plt.axis('off') 
plt.show()


# 三、扣出每一帧人物图像，并保存在humanseg_output的文件夹里

# In[ ]:


get_ipython().system('pip install paddlehub==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple')
get_ipython().system('hub install deeplabv3p_xception65_humanseg==1.0.0')


# In[ ]:


#导入包
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import paddlehub as hub
# import paddle
# paddle.enable_static() ##防止报错




# In[ ]:


# 抠出每一帧的人像图片，进行一个循环

for i in range(63):
    img_name = str(i)+'.jpg'
    test_img_path = ["transv_result/"+img_name]
#test_img_path = ["./transvideo_result/0.jpg"]
    img = mpimg.imread(test_img_path[0]) 
    module = hub.Module(name="deeplabv3p_xception65_humanseg")
    input_dict = {"image": test_img_path}

# execute predict and print the result
    results = module.segmentation(data=input_dict)


# In[ ]:


# 抠图结果展示
test_img_path = "humanseg_output/"+img_name.split('.')[0]+'.png'
img = mpimg.imread(test_img_path)
plt.figure(figsize=(10,10))
plt.imshow(img) 
plt.axis('off') 
plt.show()


# 四、视频图像提取 利用cv2库提取视频里的图像

# In[ ]:


# 这里将背景的每一帧都提取出来了
import cv2
from tqdm import tqdm
video = cv2.VideoCapture("bg.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("总共的帧数为：",frameCount)
success, frame = video.read() 
index = 0
for i in tqdm(range(int(frameCount)),desc='处理进度'):
    if success:
        cv2.imwrite('back_video_img/'+str(index)+'.jpg', frame)
    success, frame = video.read()
    index += 1


# 以下两步代码为第二种方法保存背景视频为图片，可忽略

# In[ ]:


# import cv2
# import matplotlib.pyplot as plt 
# from PIL import Image
# import numpy as np

# #保存原背景视频为图片
# cap = cv2.VideoCapture("bg.mp4")
# i=1
# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         break
#     else:
#         img = Image.fromarray(np.uint8(frame))
#         img.save('./back_video_img/'+str(i)+".jpg")
#         i+=1


# In[ ]:


# # 查看背景图片数量
# import os

# back_input_list = []
# back_img_list = os.listdir('./back_video_img')
# for i in back_img_list:
#     back_input_list.append('./back_video_img/'+i)

# print(len(back_input_list))
# print(back_input_list[:3])


# 五、将视频的每一帧与图片进行融合，最后保存为图片

# In[ ]:


from PIL import Image
import numpy as np


# In[ ]:


def blend_images(fore_image, base_image, img_num):
#def blend_images(fore_image, base_image):
    """
    将抠出的人物图像换背景
    fore_image: 前景图片，抠出的人物图片
    base_image: 背景图片
    """
    # 读入图片
    base_image = Image.open(base_image).convert('RGB')
    fore_image = Image.open(fore_image).resize(base_image.size)

    # 图片加权合成
    scope_map = np.array(fore_image)[:,:,-1] / 255
    scope_map = scope_map[:,:,np.newaxis]
    scope_map = np.repeat(scope_map, repeats=3, axis=2)
    res_image = np.multiply(scope_map, np.array(fore_image)[:,:,:3]) + np.multiply((1-scope_map), np.array(base_image))
    
    #保存图片
    res_image = Image.fromarray(np.uint8(res_image))
    res_image.save('./blend_img/'+str(img_num)+".jpg")


# In[ ]:


#抠出的人物图片和背景图片进行融合，进行循环
import os
for i in range(62):
    blend_images('./humanseg_output/'+str(i+1)+'.png', './back_video_img/'+str(i+1)+".jpg", i+1)


# 六、将视频的每一帧与图片进行融合，最后保存为图片

# In[ ]:


# 合成视频
import cv2
import os

# 查看原始视频的参数
cap = cv2.VideoCapture("bg.mp4")
ret, frame = cap.read()
height=frame.shape[0]
width=frame.shape[1]
fps = cap.get(cv2.CAP_PROP_FPS)  #返回视频的fps--帧率
size=cap.get(cv2.CAP_PROP_FRAME_WIDTH)  #返回视频的宽，等同于frame.shape[1]
size1=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  #返回视频的高，等同于frame.shape[0]

#把参数用到我们要创建的视频上
video = cv2.VideoWriter('Happy.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width,height)) #创建视频流对象
"""
参数1 即将保存的文件路径
参数2 VideoWriter_fourcc为视频编解码器
    fourcc意为四字符代码（Four-Character Codes），顾名思义，该编码由四个字符组成,下面是VideoWriter_fourcc对象一些常用的参数,注意：字符顺序不能弄混
    cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi 
    cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'),该参数是MPEG-1编码类型，文件名后缀为.avi 
    cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),该参数是MPEG-4编码类型，文件名后缀为.avi 
    cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'),该参数是Ogg Vorbis,文件名后缀为.ogv 
    cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),该参数是Flash视频，文件名后缀为.flv
    cv2.VideoWriter_fourcc('m', 'p', '4', 'v')    文件名后缀为.mp4
参数3 为帧播放速率
参数4 (width,height)为视频帧大小
"""
path = './blend_img/'
filelist = os.listdir(path)
img_num = len(filelist)

for i in range(img_num):
    #if item.endswith('.jpg'):   #判断图片后缀是否是.png
    item = path + str(i+1) + '.jpg' 
    img = cv2.imread(item)  #使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
    video.write(img)        #把图片写进视频
video.release() #释放


# 七、提取音频，合到新的视频上

# In[ ]:


# 提取原音频，合成到新的视频上
from moviepy.editor import *
video_o = VideoFileClip("video.mp4")
videoclip = VideoFileClip("Happy.mp4")
audio_o = video_o.audio

videoclip2 = videoclip.set_audio(audio_o)

videoclip2.write_videofile("result.mp4")


# 八、风格化视频

# 梵高
# 文森特·威廉·梵高（Vincent Willem van Gogh，1853年3月30日—1890年7月29日），荷兰后印象派画家。代表作有《星月夜》、自画像系列、向日葵系列等。 梵高出生于1853年3月30日荷兰乡村津德尔特的一个新教牧师家庭，早年的他做过职员和商行经纪人，还当过矿区的传教士最后他投身于绘画。他早期画风写实，受到荷兰传统绘画及法国写实主义画派的影响。1886年，他来到巴黎，结识印象派和新印象派画家，并接触到日本浮世绘的作品，视野的扩展使其画风巨变。1888年，来到法国南部小镇阿尔，创作《阿尔的吊桥》；同年与画家保罗·高更交往，但由于二人性格的冲突和观念的分歧，合作很快便告失败。此后，梵高的疯病（有人记载是“癫痫病”）时常发作，但神志清醒时他仍然坚持作画。1889年创作《星月夜》。1890年7月，梵高在精神错乱中开枪自杀，年仅37岁。

# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, you need to use the persistence path as the following:
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')
get_ipython().system('pip install lxml -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可:
# Also add the following code, so that every time the environment (kernel) starts, just run the following code:
import sys
sys.path.append('/home/aistudio/external-libraries')


# 爬取必要风格图片
# 这里我们采用百度百科中关于梵高的介绍图集中的所有图片，随便找一张图片作为styles的特征提取图片。

# In[ ]:


get_ipython().system('pip install lxml')


# In[ ]:


# 爬取梵高百度百科中的图集
import os
import re
import time
import requests
from bs4 import BeautifulSoup
def down_pics(link, fold):

    headers = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36' 
     }
    
    pic_urls=[]
    response = requests.get(link,headers=headers)
    soup = BeautifulSoup(response.text,'lxml')
    pic_list_url = soup.select('.summary-pic a')[0].get('href')
    pic_list_url = 'https://baike.baidu.com' + pic_list_url
    pic_response = requests.get(pic_list_url,headers=headers)
    soup = BeautifulSoup(pic_response.text,'lxml')
    pic_list_html = soup.select('.pic-list img ')
    path = 'work/'+fold+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    for i, pic_html in enumerate(pic_list_html):
        pic_url = pic_html.get('src')
        try:
            pic = requests.get(pic_url, timeout=15)
            string = str(i + 1) + '.jpg'
            with open(path+string, 'wb') as f:
                    f.write(pic.content)
                    print('成功下载第%s张图片: %s' % (str(i + 1), str(pic_url)))
        except Exception as e:
            print('下载第%s张图片时失败: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue
    time.sleep(0.1) # 休眠0.1s看服务器承载压力，
if __name__ == '__main__':
    # 梵高百度百科
    link = 'https://baike.baidu.com/item/%E5%A4%9C%E9%97%B4%E7%9A%84%E9%9C%B2%E5%A4%A9%E5%92%96%E5%95%A1%E5%BA%A7/667274?fr=aladdin'
    # 罗纳河上的星夜百度百科
    link1 = 'https://baike.baidu.com/item/%E5%B7%B4%E5%8B%83%E7%BD%97%C2%B7%E6%AF%95%E5%8A%A0%E7%B4%A2/22027443?fromtitle=%E6%AF%95%E5%8A%A0%E7%B4%A2&fromid=159928'
    down_pics(link1, 'Mountain')
    down_pics(link,'pic')


# In[ ]:


get_ipython().system('pip install --upgrade pillow')


# In[ ]:


get_ipython().system('pip install   --upgrade paddlepaddle')


# In[ ]:


# 安装stylepro_artistic
get_ipython().system('pip install --upgrade paddlehub  # -i https://pypi.tuna.tsinghua.edu.cn/simple')
get_ipython().system('hub install stylepro_artistic')


# 单个图像的融合
# 当戴珍珠耳环的少女遇上梵高的画风时

# In[ ]:


# 图像融合
import paddlehub as hub
import cv2

stylepro_artistic = hub.Module(name="stylepro_artistic")
result = stylepro_artistic.style_transfer(
    images=[{
        'content': cv2.imread('work/pic/12.png'),
        'styles': [cv2.imread('work/pic/2.jpg'),cv2.imread('work/pic/1.jpg')]
    }],
    visualization=True,
    output_dir='mountain_result')
print(result[0]['save_path'])


# In[ ]:


#显示已完成融合的图片
from PIL import Image
import matplotlib.pyplot as plt
plt.figure(figsize=[20,6])
plt.subplot(1,3,1)
plt.imshow(Image.open('work/Mountain/1.jpg'))
plt.axis('off')
plt.xticks([])
plt.subplot(1,3,2)
plt.imshow(Image.open('work/pic/1.jpg'))
plt.axis('off')
plt.xticks([])
plt.subplot(1,3,3)
plt.imshow(Image.open('mountain_result/ndarray_1651728751.8931372.jpg'))
plt.axis('off')
plt.xticks([])
plt.show()


# 更大胆的想法当极乐净土与上梵高画风时

# In[8]:


# 这里将极乐净土视频中的每一帧都提取出来
import cv2
from tqdm import tqdm
video = cv2.VideoCapture("re.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("总共的帧数为：",frameCount)
success, frame = video.read() 
index = 0
for i in tqdm(range(int(frameCount)),desc='处理进度'):
    if success:
        cv2.imwrite('work/target/'+str(index)+'.jpg', frame)
    success, frame = video.read()
    index += 1


# 将视频的每一帧与图片进行融合，最后保存为图片
# 极乐净土视频，跑了运行时长: 4小时09分钟40秒

# In[7]:


import cv2
import paddlehub as hub
from tqdm import tqdm
stylepro_artistic = hub.Module(name="stylepro_artistic")
video = cv2.VideoCapture("re.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("总共的帧数为：",frameCount)
success, frame = video.read() 
file_paths = []
index = 0
for i in tqdm(range(int(frameCount))):
    if success and index >= 0:
            result = stylepro_artistic.style_transfer(
                images=[{
                    'content': frame,
                    'styles': [cv2.imread('work/pic/2.jpg'),cv2.imread('work/pic/1.jpg')]
    }],
                #use_gpu=True,
                visualization=True,
                output_dir='transvideo_result')
            file_paths.append(result[0]['save_path'])
    elif success:
        filep = 'transvideo_result/'+str(index)+'.jpg'
        cv2.imwrite(filep, frame)
        file_paths.append(filep)
    success, frame = video.read()
    index += 1


# 将图片合成为视频

# In[ ]:


import os
import cv2
import datetime
file_dict = {}
video = cv2.VideoCapture("re.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
for i in os.listdir('transvideo_result/'):
    file_dict['transvideo_result/'+i] = float(i.replace('ndarray_','').replace('.jpg',''))
file_dict = sorted(file_dict.items(),key = lambda x:x[1])
videoWriter = cv2.VideoWriter('Victory.avi', cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
flag = True
for i in file_dict:
    if flag:
        for j in range(34):
            videoWriter.write(cv2.imread('work/target/0.jpg'))
        flag = False
    videoWriter.write(cv2.imread(i[0]))
videoWriter.release()
cv2.destroyAllWindows()


# In[ ]:


# 提取原音频，合成到新的视频上
from moviepy.editor import *
video_o = VideoFileClip("re.mp4")
videoclip = VideoFileClip("Victory.avi")
audio_o = video_o.audio

videoclip2 = videoclip.set_audio(audio_o)

videoclip2.write_videofile("Vic.mp4")

