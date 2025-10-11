import torch
from faster_coco_eval.utils.pytorch import FasterCocoDetection
from .coco_dataset import CocoDetection, convert_to_tv_tensor
from ...core import register

@register()
class IntelliSeeDataset(CocoDetection): # Inherits from the original CocoDetection
    def __init__(self, img_folder, ann_file, transforms, return_masks=False, classes=None, **kwargs):
        super().__init__(img_folder, ann_file, transforms, return_masks=return_masks, **kwargs)

        if classes is None:
            # No classes list provided. Running with original dataset categories
            return

        print(f"INFO: Initializing with custom remapping for {len(classes)} classes.")

        self.classes = classes
        # MappingL class name to the contiguous ID (1, 2, ...)
        name_to_new_id = {name: i for i, name in enumerate(self.classes)}   # {"class_name": new_contiguous_id }
        # Mapping original class ID to contiguous ID
        self.original_to_contiguous = {}    # {original_cls_id: new_contiguous_id}

        # Iterate through the original categories loaded by the parent class
        for original_id, cat_info in self.coco.cats.items():
            if cat_info['name'] in name_to_new_id:
                self.original_to_contiguous[original_id] = name_to_new_id[cat_info['name']]

        # Filter and Remap the annotations.
        # Parent class (CocoDetection) has loaded all annotations into self.coco.anns
        new_anns_dict = {}
        kept_image_ids = set()

        for ann_id, ann in self.coco.anns.items():
            original_cat_id = ann['category_id']
            # Check if the annotation's class is in the target cls list
            if original_cat_id in self.original_to_contiguous:
                # Update its category ID to the remapped ID
                ann['category_id'] = self.original_to_contiguous[original_cat_id]
                new_anns_dict[ann_id] = ann
                # Keep track of which images have these valid annotations
                kept_image_ids.add(ann['image_id'])

        # Directly overwrite the attributes in the underlying coco object.
        self.coco.dataset['images'] = [img for img in self.coco.dataset['images'] if img['id'] in kept_image_ids]
        self.coco.dataset['annotations'] = list(new_anns_dict.values())
        self.coco.dataset['categories'] = [{'id': i, 'name': class_name} for i, class_name in enumerate(self.classes)]

        self.coco.createIndex()

        self.ids = sorted(self.coco.getImgIds())

        print(f"INFO: Remapping complete. Kept {len(self.ids)} images containing the target classes.")

    def load_item(self, idx):
        image, target = super(FasterCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        image, target = self.prepare(image, target, category2label=self.category2label)

        target['idx'] = torch.tensor([idx])

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')

        return image, target

    # Override properties to use our new remapped categories
    @property
    def categories(self):
        return self.coco.dataset['categories']