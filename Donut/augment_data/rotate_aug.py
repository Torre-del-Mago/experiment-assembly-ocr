from PIL import Image
import random
from augment_data.augment import DataAugmentation


class RotateDataAugmentation(DataAugmentation):
    
    def __init__(self, json_path):
        super().__init__(json_path)
        self.name = "rotate"
    
    def resize_to_original(self, image: Image, original_size):
        return image.resize(original_size, resample=Image.BICUBIC)
    
    def aug_image(self, image):
        if image.mode != 'RGB': 
            image = image.convert('RGB') 
        original_size = image.size
        
        if 0.5 < random.random():
            rand_num = random.randrange(-30, -15)
        else:
            rand_num = random.uniform(15, 30)
        
        augmented_image = image.rotate(rand_num, resample=Image.BICUBIC, expand=True)
        augmented_image = self.resize_to_original(augmented_image, original_size)
        
        return augmented_image