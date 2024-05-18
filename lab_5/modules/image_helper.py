import cv2
import os
import numpy as np

class ImageHelper:
    @staticmethod
    def get_links(num_test: int = 1, all: bool = False, dataset: str = "dataset") -> tuple[list, list]:

        #dataset = 'dataset/Educate'

        links_x = list()
        links_y = list()
        if all:
            links_xy = list()
            for i in range(5):
                links_xy = ImageHelper.get_links(i + 1)
                links_x += links_xy[0]
                links_y += links_xy[1]
        else:
            k = 0

            for directory in os.listdir(f"{dataset}/person"):
                if os.path.isfile(f"{dataset}/person/{directory}/{directory}.jpg"):
                    links_x.append(f"{dataset}/person/{directory}/{directory}.jpg")
                    links_y.append(f"{dataset}/pet/{directory}/{directory}.jpg")

        print(links_x)
        print('-----p-----')
        print(links_y)
        return links_x, links_y

    @staticmethod
    def resize_and_crop(image: np.ndarray, target_width: int = 10, target_height: int = 10, isheight: bool = True) -> np.ndarray:

        original_height, original_width = image.shape
        aspect_ratio = original_width / original_height

        if isheight:
            new_height = target_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(new_width / aspect_ratio)

        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    @staticmethod
    def sharpen_image(image_path: str, new_image_path: str) -> None:

        image = cv2.imread(image_path)

        # Define the sharpening kernel
        sharpen_filter = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])

        sharpened_image = cv2.filter2D(image, -1, sharpen_filter)
        cv2.imwrite(new_image_path, sharpened_image)

    @staticmethod
    def get_pictures(links, target_size=(256, 256)) -> np.ndarray:
        result = list()
        for filename in links:

            image = cv2.imread(filename, 0)
            height, width = image.shape
            min_size = min(height, width)
            cropped_image = image[:min_size, :min_size]

            resized_image = cv2.resize(cropped_image, target_size)

            img = np.float64(resized_image) / 255
            result.append(img)

        return np.array(result)