from PIL import Image
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


def coco_collate_fn(batch):
    try:
        images = []
        bboxs = []
        category_ids = []
        center_widths = []
        center_heights = []
        img_ids = []
        i = 0
        for sample in batch:
            images.append(sample['image'])
            for ann in sample['annotations']:
                bboxs.append(ann['bbox'])
                img_ids.append(i)
                category_ids.append(ann['category_id'])
                center_widths.append(ann['center_width'])
                center_heights.append(ann['center_height'])
            i += 1

        images = torch.stack(images, dim=0)
        bboxs = torch.stack(bboxs, dim=0)
        category_ids = torch.tensor(category_ids)
        center_widths = torch.tensor(center_widths)
        center_heights = torch.tensor(center_heights)
        img_ids = torch.tensor(img_ids)
        out_dict = {
            'images': images,
            'bboxs': bboxs,
            'category_ids': category_ids,
            'center_widths': center_widths,
            'center_heights': center_heights,
            'img_ids': img_ids
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        print(category_ids)
    return out_dict


class ImageIdList:
    def __init__(self,
                 data_type: str = 'train',
                 selected_categories: list = None,
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

        # Get ids of selected categories and store it into dictionary
        selected_categories_dict = {}
        for cat in categories_data:
            if selected_categories is None:
                selected_categories_dict[cat['id']] = cat['name']
            elif cat['name'] in selected_categories:
                selected_categories_dict[cat['id']] = cat['name']
        self.selected_categories_dict = selected_categories_dict

        # Create dictionary with image ids as keys and features to predict as values
        ann_dict_by_image = {}
        selected_categories_ids = selected_categories_dict.keys()
        for ann in annotations_data:
            cat_id = ann['category_id']
            if cat_id in selected_categories_ids:
                image_id = ann['image_id']
                temp_dict = {}
                temp_dict['category_id'] = cat_id
                for feat in selected_output:
                    temp_dict[feat] = torch.tensor(ann[feat], dtype=torch.float32)
                if image_id in ann_dict_by_image.keys():
                    ann_dict_by_image[image_id]['annotations'].append(temp_dict)
                else:
                    ann_dict_by_image[image_id] = {'annotations': [temp_dict, ]}

        # Get list of links to images for selected categories
        images_links = {}
        image_ids = ann_dict_by_image.keys()
        for img in images_data:
            img_id = img['id']
            if img_id in image_ids:
                images_links[img_id] = img['file_name']
                ann_dict_by_image[img_id]['width'] = img['width']
                ann_dict_by_image[img_id]['height'] = img['height']

        self.ann_dict_by_image = ann_dict_by_image
        self.images_links = images_links


class ImageDataset(Dataset):
    def __init__(self,
                 data_type: str = 'train',
                 selected_categories: list = None,
                 selected_output: list = ['bbox', ],
                 new_size: tuple = (512, 512),
                 patch_size: int = 16,
                 transform=None,
                 dataset_size=None
                 ) -> dict:
        super(ImageDataset, self).__init__()
        self.data_type = data_type
        self.selected_categories = selected_categories
        self.selected_output = selected_output
        self.new_size = new_size
        self.patch_size = patch_size
        self.transform = transform
        self.dataset_size = dataset_size

        image_links = ImageIdList(data_type,
                                  selected_categories,
                                  selected_output)
        self.image_links = image_links

        cat_dict = {}
        for i, key in enumerate(image_links.selected_categories_dict):
            cat_dict[key] = i
        self.cat_dict = cat_dict

    def __len__(self):
        if self.dataset_size:
            return min(self.dataset_size, len(self.image_links.images_links.keys()))
        else:
            return len(self.image_links.images_links.keys())

    def __getitem__(self, idx):
        output_dict = {}
        im_key = list(self.image_links.images_links.keys())[idx]
        im_link = self.image_links.image_path + self.image_links.images_links[im_key]
        im = Image.open(im_link)
        im_trf = self.transform(im)
        output_dict['image'] = im_trf

        im_anns = self.image_links.ann_dict_by_image[im_key]

        height_scale = self.new_size[0] / im_anns['height']
        width_scale = self.new_size[1] / im_anns['width']

        annotations = []
        for ann in im_anns['annotations']:
            temp_ann = {}
            bbox = ann['bbox'].clone()
            bbox[::2] *= width_scale
            bbox[1::2] *= height_scale

            center_width = (bbox[0] + bbox[2] / 2) // self.patch_size
            center_width = center_width.int()
            center_height = (bbox[1] + bbox[3] / 2) // self.patch_size
            center_height = center_height.int()
            cat_id = ann['category_id']

            temp_ann['bbox'] = bbox
            temp_ann['category_id'] = cat_id
            temp_ann['center_width'] = center_width
            temp_ann['center_height'] = center_height

            annotations.append(temp_ann)

        output_dict['width'] = im_anns['width']
        output_dict['height'] = im_anns['height']
        output_dict['annotations'] = annotations

        return output_dict


def create_dataloader(
    data_type: str = 'train',
    selected_categories: list = None,
    selected_output: list = ['bbox', ],
    new_size: tuple = (512, 512),
    patch_size: int = 16,
    transform=None,
    batch_size=8,
    num_workers=12,
    shuffle=True,
    dataset_size=None
):
    dataset = ImageDataset(
        data_type=data_type,
        selected_categories=selected_categories,
        selected_output=selected_output,
        new_size=new_size,
        patch_size=patch_size,
        transform=transform,
        dataset_size=dataset_size
    )

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=coco_collate_fn,
                            num_workers=num_workers
                            )
    return dataloader

