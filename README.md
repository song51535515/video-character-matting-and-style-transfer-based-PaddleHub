# video character matting and style transfer based PaddleHub

这个项目实现了将视频中的人物抠出，替换背景，并融合梵高的《夜间的露天咖啡座》与莫奈的《日出印象》的艺术风格对视频每一帧画面实现风格迁移，最终生成新的视频并保留原视频音频


## 主要功能

1.  **视频帧提取**：将原视频和背景视频的每一帧提取为图片
2.  **人物抠图**：使用深度学习模型抠出视频中的人物
3.  **背景替换**：将抠出的人物与新背景融合
4.  **视频合成**：将融合后的图片重新合成为视频
5.  **音频处理**：提取原视频音频并合并到新视频中
6.  **风格迁移**：将梵高风格等艺术效果应用到视频上

## 效果预览
**原图**![原图片](/img/55.jpg)**抠图并与背景融合**![抠图并与背景融合](/img/551.jpg)**风格迁移**![实现风格迁移后的图片](/img/552.jpg)


## 环境要求

-   Python 3.x
-   所需依赖库：
    -   opencv-python
    -   paddlehub==1.6.0
    -   deeplabv3p_xception65_humanseg==1.0.0
    -   stylepro_artistic
    -   matplotlib
    -   tqdm
    -   moviepy
    -   requests
    -   beautifulsoup4
    -   lxml
    -   pillow

## 安装依赖
```python
pip install opencv-python matplotlib tqdm moviepy requests beautifulsoup4 lxml pillow 
pip install paddlehub==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple 
hub install deeplabv3p_xception65_humanseg==1.0.0 
hub install stylepro_artistic
```

## 使用步骤

1.  准备素材：
    
    -   包含人物的原视频（命名为 video.mp4）
    -   作为新背景的视频（命名为 bkg.mp4）
    
1.  检查必要的文件夹（若没有需创建）：
```python
mkdir transv_result humanseg_output back_video_img blend_img work/Cafe work/Sunrise transvideo_result
``` 

3.  运行程序：按照代码中的步骤依次执行，或直接运行完整脚本


## 项目结构


```plaintext
.
├── main.ipynb               # 主程序脚本
├── video.mp4                # 输入的原视频（需用户提供）
├── bkg.mp4                  # 输入的背景视频（需用户提供）
├── transv_result/           # 原视频提取的帧
├── humanseg_output/         # 抠图后的人物图像
├── back_video_img/          # 背景视频提取的帧
├── blend_img/               # 人物与背景融合后的图像
├── work/                    # 风格图片相关文件夹
│   ├── Cafe/
│   └── Sunrise/
├── transvideo_result/       # 风格迁移后的帧
├── Happy.mp4                # 中间生成的视频
├── result.mp4               # 带音频的中间视频
├── Victory.avi              # 风格迁移后的视频
└── Vic.mp4                  # 最终生成的带音频视频
```

## 代码步骤
**step 1  视频帧提取＋人物抠像**

将原视频的每一帧都提取出来, 分别放入transv_result文件夹
```python
import cv2
from tqdm import tqdm
video = cv2.VideoCapture("video.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("总共的帧数为：",frameCount)
success, frame = video.read() 
index = 0
img_num=int(frameCount)
for i in tqdm(range(int(frameCount)),desc='处理进度'):
    if success:
        cv2.imwrite('transv_result/'+str(index)+'.jpg', frame)
    success, frame = video.read()
    index += 1

```
抠出每一帧人像，进行循环，保存在humanseg_output文件夹中
```python
for i in range(img_num):
    img_name = str(i)+'.jpg'
    test_img_path = ["transv_result/"+img_name]
#test_img_path = ["./transvideo_result/0.jpg"]
    img = mpimg.imread(test_img_path[0]) 
    module = hub.Module(name="deeplabv3p_xception65_humanseg")
    input_dict = {"image": test_img_path}

# execute predict and print the result
    results = module.segmentation(data=input_dict)

```
同样的方法提取背景视频的每一帧存入back_video_img文件夹
```python
import cv2
from tqdm import tqdm
video = cv2.VideoCapture("bkg.mp4")
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
```
**step 2 背景图像提取，并与抠出的人像融合**

抠出的人物图片和背景图片进行融合，进行循环
```python
from PIL import Image
import numpy as np
import os


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
for i in range(img_num):
    blend_images('./humanseg_output/'+str(i)+'.png', './back_video_img/'+str(i)+".jpg", i)

```
将视频的每一帧与图片进行融合，最后保存为图片
```python
import cv2
import os

# 查看原始视频的参数
cap = cv2.VideoCapture("bkg.mp4")
ret, frame = cap.read()
height=frame.shape[0]
width=frame.shape[1]
fps = cap.get(cv2.CAP_PROP_FPS)  #返回视频的fps--帧率
size=cap.get(cv2.CAP_PROP_FRAME_WIDTH)  #返回视频的宽，等同于frame.shape[1]
size1=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  #返回视频的高，等同于frame.shape[0]

#把参数用到我们要创建的视频上
video = cv2.VideoWriter('Happy.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width,height)) #创建视频流对象
path = './blend_img/'
filelist = os.listdir(path)
img_num = len(filelist)

for i in range(img_num):
    #if item.endswith('.jpg'):   #判断图片后缀是否是.png
    item = path + str(i) + '.jpg' 
    img = cv2.imread(item)  #使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
    video.write(img)        #把图片写进视频
video.release() #释放
```
提取原视频音频，合到新的视频上
```python
from moviepy.editor import *
video_o = VideoFileClip("video.mp4")
videoclip = VideoFileClip("Happy.mp4")
audio_o = video_o.audio

videoclip2 = videoclip.set_audio(audio_o)

videoclip2.write_videofile("result.mp4")

```
**step 3 风格化视频**

爬取百度百科中《夜间的露天咖啡座》和《日出印象》对应画作
```python
import os
import time
import requests
from bs4 import BeautifulSoup

def down_pics(link, fold):
    # 升级请求头，模拟现代浏览器
    headers = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Referer': 'https://baike.baidu.com/'  # 增加来源页，降低反爬概率
    }
    
    # 确保保存目录存在（无论是否已存在）
    path = f'work/{fold}/'
    os.makedirs(path, exist_ok=True)  # exist_ok=True：目录存在时不报错
    
    try:
        # 获取主页面内容
        response = requests.get(link, headers=headers, timeout=10)
        response.raise_for_status()  # 检查请求是否成功（如404、503等）
        soup = BeautifulSoup(response.text, 'lxml')
        
        # 提取图片标签（支持单张或多张，这里以匹配所有符合条件的img标签为例）
        # 选择器：匹配百度百科词条中常见的图片标签
        img_tags = soup.select('div.lemmaPicture_SCqia img.picture_TczB1')
        
        if not img_tags:
            print("未找到任何图片标签，可能页面结构已变化")
            return
        
        # 遍历所有图片标签，下载图片
        for idx, img_tag in enumerate(img_tags, 1):  # idx从1开始计数
            pic_url = img_tag.get('src')
            if not pic_url:
                print(f"第{idx}张图片URL不存在，跳过")
                continue
            
            if not pic_url.startswith(('http://', 'https://')):
                pic_url = 'https:' + pic_url
            
            try:
                # 下载图片
                pic_response = requests.get(pic_url, headers=headers, timeout=15)
                pic_response.raise_for_status()
                
                # 保存图片
                filename = f"{idx}.jpg"
                with open(os.path.join(path, filename), 'wb') as f:
                    f.write(pic_response.content)
                print(f'成功下载第{idx}张图片: {pic_url}')
                time.sleep(0.5)  # 适当延长休眠，避免请求过快
                
            except Exception as e:
                print(f'下载第{idx}张图片失败: {pic_url}')
                print(f'错误原因: {str(e)}')
        
    except Exception as e:
        print(f"访问页面或解析时出错: {str(e)}")

if __name__ == '__main__':
    # 测试链接（确保链接正确）
    link1 = 'https://baike.baidu.com/item/%E5%A4%9C%E9%97%B4%E7%9A%84%E9%9C%B2%E5%A4%A9%E5%92%96%E5%95%A1%E5%BA%A7/667274'  # 夜间咖啡馆（正确链接）
    link2 = 'https://baike.baidu.com/item/%E6%97%A5%E5%87%BA%C2%B7%E5%8D%B0%E8%B1%A1/53697'  # 日出·印象（正确链接）
    
    down_pics(link1, 'Cafe')
    down_pics(link2, 'Sunrise')

```
将视频中的每一帧进行风格迁移（为了同时展示抠图和风格迁移成果前44帧保留抠图视频，44帧后为风格迁移视频）
```python
import cv2
import paddlehub as hub
from tqdm import tqdm
stylepro_artistic = hub.Module(name="stylepro_artistic")
video = cv2.VideoCapture("result.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("总共的帧数为：",frameCount)
success, frame = video.read() 
file_paths = []
index = 0
for i in tqdm(range(int(frameCount))):
    if success and index >= 44:
            result = stylepro_artistic.style_transfer(
                images=[{
                    'content': frame,
                    'styles': [cv2.imread('work/Sunrise/1.jpg'),cv2.imread('work/Cafe/1.jpg')]
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

```
将图片合成为视频
```python
import os
import cv2

# 从原视频获取帧率和尺寸参数（用于保持视频属性一致）
video = cv2.VideoCapture("result.mp4")
fps = video.get(cv2.CAP_PROP_FPS)  # 帧率
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 视频尺寸
video.release()  # 获取参数后释放视频对象

# 收集transvideo_result文件夹中的图片并按序号排序
file_dict = {}
for filename in os.listdir('transvideo_result/'):
    # 只处理图片文件
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            # 提取文件名中的数字（假设格式为"ndarray_数字.jpg"）
            num = float(filename.replace('ndarray_', '').split('.')[0])
            file_dict[os.path.join('transvideo_result', filename)] = num
        except (ValueError, IndexError):
            # 跳过命名不符合规则的文件
            continue

# 按数字序号排序图片路径
sorted_files = sorted(file_dict.items(), key=lambda x: x[1])

# 如果没有有效图片，提示并退出
if not sorted_files:
    print("transvideo_result文件夹中未找到有效图片")
    exit()

# 创建视频写入对象
videoWriter = cv2.VideoWriter(
    'Victory.avi',
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    size
)

# 按顺序写入所有图片
for file_path, _ in sorted_files:
    img = cv2.imread(file_path)
    if img is not None:
        # 确保图片尺寸与视频尺寸一致（如不一致则调整）
        if img.shape[:2] != (size[1], size[0]):
            img = cv2.resize(img, size)
        videoWriter.write(img)
    else:
        print(f"警告：无法读取图片 {file_path}，已跳过")

# 释放资源
videoWriter.release()
cv2.destroyAllWindows()
print(f"视频合成完成，共写入 {len(sorted_files)} 张图片")
```
提取原音频，合成到新的视频上
```python
from moviepy.editor import *
video_o = VideoFileClip("video.mp4")
videoclip = VideoFileClip("Victory.avi")
audio_o = video_o.audio

videoclip2 = videoclip.set_audio(audio_o)

videoclip2.write_videofile("Vic.mp4")
```

