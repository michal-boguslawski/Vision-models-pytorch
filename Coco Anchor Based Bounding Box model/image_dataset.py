from PIL import Image
import os
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class ImageIdList:
    def __init__(self, data_type: str = 'train', 
                 selected_categories: list = ['person', ], 
                 selected_output: list = ['bbox', ]):
        '''
        Stores links to images and selected output features to predict
        '''
        self.data_type = data_type
        self.image_path = f'data/{data_type}2017/'
        self.annotation_path = f'data/annotations/instances_{data_type}2017.json'
        
        with open(self.annotation_path, 'r') as file:
            data = json.load(file)
            
        images_data = data['images']
        annotations_data = data['annotations']
        categories_data = data['categories']
    
        #Get ids of selected categories and store it into dictionary
        selected_categories_dict = {}
        for cat in categories_data:
            if cat['name'] in selected_categories:
                selected_categories_dict[cat['id']] = cat['name']
        self.selected_categories_dict = selected_categories_dict

        #Create dictionary with image ids as keys and features to predict as values
        ann_dict_by_image = {}
        selected_categories_ids = selected_categories_dict.keys()
        for ann in annotations_data:
            cat_id = ann['category_id']
            if cat_id in selected_categories_ids:
                image_id = ann['image_id']
                temp_dict = {}
                temp_dict['category_id'] = cat_id
                for feat in selected_output:
                    temp_dict[feat] = ann[feat]
                if image_id in ann_dict_by_image.keys():
                    ann_dict_by_image[image_id].append(temp_dict)
                else:
                    ann_dict_by_image[image_id] = [temp_dict, ]
        self.ann_dict_by_image = ann_dict_by_image

        #Get list of links to images for selected categories
        images_links = {}
        image_ids = ann_dict_by_image.keys()
        for img in images_data:
            img_id = img['id']
            if img_id in image_ids:
                images_links[img_id] = img['file_name']
        self.images_links = images_links

class ImageDataset(Dataset):
    def __init__(self, 
                 data_type: str = 'train',
                 selected_categories: list = ['person'],
                 selected_output: list = ['bbox'],
                 transform = None
                ):
        super(ImageDataset, self).__init__()
        image_links = ImageIdList(data_type = 'train',
                          selected_categories = ['skis'],
                          selected_output = ['bbox'])
        self.image_links = image_links
        self.transform = transform

    def __len__(self):
        return len(self.image_links.images_links.keys())

    def __getitem__(self, idx):
        output_dict = {}
        im_key = list(self.image_links.images_links.keys())[idx]
        im_link = self.image_links.image_path + self.image_links.images_links[im_key]
        im = Image.open(im_link)
        im_trf = self.transform(im)
        output_dict['image'] = im_trf
        
        im_anns = self.image_links.ann_dict_by_image[im_key]
        output_dict['annotations'] = im_anns

        return output_dict
        
dataset = ImageDataset(data_type = 'train',
                       selected_categories = ['skis', ],
                       selected_output = ['bbox', ],
                       transform = transforms.Compose([
                           transforms.ToTensor(),
                       ]
                                                     )
                      )

print(dataset[0])
