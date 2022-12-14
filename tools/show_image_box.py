## annotation.json을 불러오면, 이미지와 annotation box를 표시하는 기능
## bbox 조건 : (polygon의 네 점)
#%%
import json
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

#sample_num = 16
img_path = '/opt/ml/input/data/dataset/images/'
ann_path = '/opt/ml/input/data/dataset/ufo/'

with open(ann_path+'train.json', 'r') as f:
    ann_info = json.load(f)
    

def show_img_with_ann(file_name, img_path):
    file_path = img_path + file_name
    img = mpimg.imread(file_path)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    
    for ann in ann_info['images'][file_name]['words'].values():
        if not ann['illegibility']:
            poly = patches.Polygon(ann['points'], closed=True, edgecolor='red', fill=False)
            ax.add_patch(poly)

    plt.show(block=True)
    

if __name__ == '__main__':
    show_file_name = '20210913_210859.jpeg'
    show_img_with_ann(file_name=show_file_name, img_path=img_path)
# %%
