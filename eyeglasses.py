import PIL.Image
import PIL
import os
import torch

from torchvision import transforms


class Predictor:
    def __init__(self, data_path, model, iterations):
        self.data_path = data_path
        self.model = model
        self.images = os.listdir(data_path)
        self.iterations = iterations

    def predict(self, img):

        try:
            image = PIL.Image.open(f'{self.data_path}/{img}')
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            image = transform(image)
            image = image[None]

            prediction = self.model(image)

            class_label = torch.argmax(prediction).item()
            if class_label == 0:
                print('!!! Eyeglasses found at:', self.data_path + '\\' + img)
            return class_label

        except Exception:
            print(f'{img} cannot be read')
#nig
    def get_labels(self):
        i, glasses = 0, 0
        for img in self.images:
            if self.predict(img) == 0:
                glasses += 1

            i += 1
            if i >= self.iterations:
                break
        if glasses == 0:
            print('Zero people with eyeglasses found')
        else:
            print(f'There are {glasses} people with eyeglasses on')