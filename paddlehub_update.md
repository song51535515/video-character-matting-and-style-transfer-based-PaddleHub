# Video Character Matting and Style Transfer Based on PaddleHub

This project implements the removal of the characters in the video, replacing the background, and integrating the artistic style of Van Gogh's "The Night Cafe in the Place Lamartine in Arles" and Monet's "Impression Sunrise" to realize the style transfer of each frame of the video, and finally generates a new video and retains the original video audio


## Major Function 

- **Video frame extraction**: Extract each frame of the original video and background video as a picture
- **Character matt**: Using deep learning model to extract the characters in the video
- **Background replacement**: Blend the extracted characters with the new background
- **Video compositing**: Recombine the fused image into video
- **Audio processing**: Extract the original video audio and merge it into the new video
- **Style transferring**: Apply artistic effects such as Van Gogh style to the video

## Output preview
**Original picture**![原图片](/img/55.jpg)**Matting and blending with background**![抠图并与背景融合](/img/551.jpg)**Style transferring**![实现风格迁移后的图片](/img/552.jpg)


## Environmental requirements

-   Python 3.x
-   Required dependent Libraries:
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

## Installation dependencies
```python
pip install opencv-python matplotlib tqdm moviepy requests beautifulsoup4 lxml pillow 
pip install paddlehub==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple 
hub install deeplabv3p_xception65_humanseg==1.0.0 
hub install stylepro_artistic
```

## Processing steps

1.  Prepare materials:
    
    -  Original video containing characters (video.mp4)
    -   Video as new background (bkg.mp4)
    
1.  Check the necessary folders (If there is no folder, you need to create it yourself):
```python
mkdir transv_result humanseg_output back_video_img blend_img work/Cafe work/Sunrise transvideo_result
``` 

3.  Run the program: follow the steps in the code, or directly run the complete script


## Project Structure


```plaintext
.
├── main.ipynb              # Main program script
├── video.mp4               # Original video (Provided by the user)

├── bkg.mp4                 # Background video (Provided by the user)
├── transv_result/          # Frames extracted from original video
├── humanseg_output/        # The character images after matting
├── back_video_img/         # Frames extracted from background video
├── blend_img/              # Images blended with character and background
├── work/                   # Style images folder
│   ├── Cafe/
│   └── Sunrise/
├── transvideo_result/      # Frames after style transfer
├── Happy.mp4               # Character matting video without audio
├── result.mp4              # Character matting video
├── Victory.avi             # Style transferring video without audio
└── Vic.mp4                 # Style transferring video
```

## Code steps
**step 1  Video frame extraction + character segmentation**

Extract every frame of the original video and place them separately in the transv-result folder
```python
import cv2
from tqdm import tqdm
video = cv2.VideoCapture("video.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("Number of frames: ",frameCount)
success, frame = video.read() 
index = 0
img_num=int(frameCount)
for i in tqdm(range(int(frameCount)),desc='Processing progress'):
    if success:
        cv2.imwrite('transv_result/'+str(index)+'.jpg', frame)
    success, frame = video.read()
    index += 1

```
Extract each frame of the character, loop through it, and save it in the humanseg_output folder
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
Extract each frame of the background video and store it in the back-video_img folder
```python
import cv2
from tqdm import tqdm
video = cv2.VideoCapture("bkg.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("Number of frames: ",frameCount)
success, frame = video.read() 
index = 0
for i in tqdm(range(int(frameCount)),desc='Processing progress'):
    if success:
        cv2.imwrite('back_video_img/'+str(index)+'.jpg', frame)
    success, frame = video.read()
    index += 1
```
**step 2 Background image extraction and blend with the extracted character**

Blend the extracted character image with the background image and loop through them
```python
from PIL import Image
import numpy as np
import os


def blend_images(fore_image, base_image, img_num):
#def blend_images(fore_image, base_image):
    """
    Replace the extracted character image with a background
    fore_image: character
    base_image: background
    """
    # read images
    base_image = Image.open(base_image).convert('RGB')
    fore_image = Image.open(fore_image).resize(base_image.size)

    # Weighted synthesis of images
    scope_map = np.array(fore_image)[:,:,-1] / 255
    scope_map = scope_map[:,:,np.newaxis]
    scope_map = np.repeat(scope_map, repeats=3, axis=2)
    res_image = np.multiply(scope_map, np.array(fore_image)[:,:,:3]) + np.multiply((1-scope_map), np.array(base_image))
    
    #Save images
    res_image = Image.fromarray(np.uint8(res_image))
    res_image.save('./blend_img/'+str(img_num)+".jpg")
for i in range(img_num):
    blend_images('./humanseg_output/'+str(i)+'.png', './back_video_img/'+str(i)+".jpg", i)

```
Merge each frame of the video with the image and save it as an image
```python
import cv2
import os

# View the parameters of the original video
cap = cv2.VideoCapture("bkg.mp4")
ret, frame = cap.read()
height=frame.shape[0]
width=frame.shape[1]
fps = cap.get(cv2.CAP_PROP_FPS)  #Return the fps frame rate of the video
size=cap.get(cv2.CAP_PROP_FRAME_WIDTH)  #Return the width of the video, equivalent to frame. shape [1]
size1=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  #Return the height of the video, equivalent to frame.scape [0]

#Apply the parameters to the video we want to create
video = cv2.VideoWriter('Happy.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width,height)) #Create a video stream object
path = './blend_img/'
filelist = os.listdir(path)
img_num = len(filelist)

for i in range(img_num):
    item = path + str(i) + '.jpg' 
    img = cv2.imread(item)  #Read the image using OpenCV and directly return the numpy.ndarray object. The channel order is BGR Note that it is BGR. The default range for channel values is 0-255.
    video.write(img)        
video.release() 
```
Extract the original video audio and merge it into a new video
```python
from moviepy.editor import *
video_o = VideoFileClip("video.mp4")
videoclip = VideoFileClip("Happy.mp4")
audio_o = video_o.audio

videoclip2 = videoclip.set_audio(audio_o)

videoclip2.write_videofile("result.mp4")

```
**step 3 Stylish video**

Crawl the corresponding paintings of "The Night Cafe in the Place Lamartine in Arles" and "Impressions Sunrise" from Baidu Baike
```python
import os
import time
import requests
from bs4 import BeautifulSoup

def down_pics(link, fold):
    # Simulate modern browsers
    headers = { 
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Referer': 'https://baike.baidu.com/'  # Add source pages to reduce the probability of anti-crawling
    }
    
    # Ensure that the save directory exists (regardless of whether it already exists)
    path = f'work/{fold}/'
    os.makedirs(path, exist_ok=True)  # exist_ok=True: No error reported when directory exists
    
    try:
        # Get the content
        response = requests.get(link, headers=headers, timeout=10)
        response.raise_for_status()  # Check if the request is successful 
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Selectors: Match common image tags in Baidu Baike
        img_tags = soup.select('div.lemmaPicture_SCqia img.picture_TczB1')
        
        if not img_tags:
            print("No image tags were found, Maybe the page structure has changed")
            return
        
        # Traverse all image tags and download images
        for idx, img_tag in enumerate(img_tags, 1):  # Idx starts counting from 1
            pic_url = img_tag.get('src')
            if not pic_url:
                print(f"The URL for the {idx} th image does not exist, skip it")
                continue
            
            if not pic_url.startswith(('http://', 'https://')):
                pic_url = 'https:' + pic_url
            
            try:
                # download pictures
                pic_response = requests.get(pic_url, headers=headers, timeout=15)
                pic_response.raise_for_status()
                
                # save pictures
                filename = f"{idx}.jpg"
                with open(os.path.join(path, filename), 'wb') as f:
                    f.write(pic_response.content)
                print(f'Successfully downloaded the {idx} th image: {pic_url}')
                time.sleep(0.5)  # Extend sleep to avoid requests being too fast
                
            except Exception as e:
                print(f'Failed to download the {idx} th image: {pic_url}')
                print(f'Reason for error: {str (e)}')
        
    except Exception as e:
        print(f"Error accessing page or parsing: {str (e)}")

if __name__ == '__main__':

    link1 = 'https://baike.baidu.com/item/%E5%A4%9C%E9%97%B4%E7%9A%84%E9%9C%B2%E5%A4%A9%E5%92%96%E5%95%A1%E5%BA%A7/667274'  # The Night Cafe in the Place Lamartine in Arles
    link2 = 'https://baike.baidu.com/item/%E6%97%A5%E5%87%BA%C2%B7%E5%8D%B0%E8%B1%A1/53697'  # Impression sunrise
    
    down_pics(link1, 'Cafe')
    down_pics(link2, 'Sunrise')

```
Transfer the style of each frame in the video. To simultaneously display both the cutout results and style transfer results, the first 44 frames will retain the cutout video content. In contrast, the style transfer video content will be displayed from frame 45 onwards.
```python
import cv2
import paddlehub as hub
from tqdm import tqdm
stylepro_artistic = hub.Module(name="stylepro_artistic")
video = cv2.VideoCapture("result.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("Number of frames: ",frameCount)
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
Synthesize images into videos
```python
import os
import cv2

# Get frame rate and size parameters from the original video (to maintain consistency in video attributes)
video = cv2.VideoCapture("result.mp4")
fps = video.get(cv2.CAP_PROP_FPS)  # frame rate
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # video size
video.release() 

# Collect images from the transvideo_desult folder and sort them by sequence number
file_dict = {}
for filename in os.listdir('transvideo_result/'):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            # Extract the numbers from the file name (assuming the format is "ndarray_number.jpg"))
            num = float(filename.replace('ndarray_', '').split('.')[0])
            file_dict[os.path.join('transvideo_result', filename)] = num
        except (ValueError, IndexError):
            # Skip files with irregular naming conventions
            continue

# Sort image paths by numerical sequence number
sorted_files = sorted(file_dict.items(), key=lambda x: x[1])

# If there are no valid images, prompt and exit
if not sorted_files:
    print("No valid image was found in the transvideo_desult folder")
    exit()

# Create video writing object
videoWriter = cv2.VideoWriter(
    'Victory.avi',
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    size
)

# Write all images in order
for file_path, _ in sorted_files:
    img = cv2.imread(file_path)
    if img is not None:
        # Ensure that the image size is consistent with the video size (adjust if not)
        if img.shape[:2] != (size[1], size[0]):
            img = cv2.resize(img, size)
        videoWriter.write(img)
    else:
        print(f"Warning: Unable to read image {file_cath}, skipped")

videoWriter.release()
cv2.destroyAllWindows()
print(f"Video synthesis completed, with a total of {len (sorted files)} images written")
```
Extract the original audio and synthesize it into a new video
```python
from moviepy.editor import *
video_o = VideoFileClip("video.mp4")
videoclip = VideoFileClip("Victory.avi")
audio_o = video_o.audio

videoclip2 = videoclip.set_audio(audio_o)

videoclip2.write_videofile("Vic.mp4")
```