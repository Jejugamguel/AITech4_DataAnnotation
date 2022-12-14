## annotation.json을 불러오면, 이미지와 annotation box를 표시하는 기능
## bbox 조건 : (polygon의 네 점)
## 사용법 : img_path, ann_path, pre_path를 알맞게 변경하고 show_image_box.py를 실행, line5의 #%% 위에 뜨는 '셀 샐행'

#%%
import json
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

sample_num = 16
img_path = '/opt/ml/input/data/dataset/images/'
ann_path = '/opt/ml/input/data/dataset/ufo/'
pre_path = '/opt/ml/code/predictions/output.csv'

with open(ann_path+'train.json', 'r') as f:
    ann_info = json.load(f)
with open(pre_path, 'r') as f:
    pred_info = json.load(f)
    

def show_img_with_ann(file_name, img_path):
    file_path = img_path + file_name
    img = mpimg.imread(file_path)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    
    for ann in ann_info['images'][file_name]['words'].values():
        if not ann['illegibility']:
            poly = patches.Polygon(ann['points'], closed=True, edgecolor='red', fill=False, lw=3)
            ax.add_patch(poly)

    plt.show(block=True)
    
def show_img_with_predNgt(file_name, img_path):
    file_path = img_path + file_name
    img = mpimg.imread(file_path)
    
    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    ax1.imshow(img)
    ax2.imshow(img)
    
    for ann in ann_info['images'][file_name]['words'].values():
        if not ann['illegibility']:
            poly = patches.Polygon(ann['points'], closed=True, edgecolor='red', fill=False, lw=3)
            ax1.add_patch(poly)
    
    for ann in pred_info['images'][file_name]['words'].values():
        poly = patches.Polygon(ann['points'], closed=True, edgecolor='blue', fill=False, lw=3)
        ax2.add_patch(poly)

    plt.show(block=True)
    


if __name__ == '__main__':
    #show_file_name = '20210913_210859.jpeg'
    #show_img_with_ann(file_name=show_file_name, img_path=img_path)
    for _ in range(sample_num):
        show_file_name = random.choice(list(pred_info['images'].keys()))
        show_img_with_predNgt(file_name=show_file_name, img_path=img_path)
    
#%%
