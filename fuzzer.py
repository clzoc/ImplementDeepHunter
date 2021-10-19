import albumentations as album
import cv2
import torchvision.transforms as transforms


class ColorSpace:
    def bgr2rgb(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def bgr2hsv(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv_image

    def bgr2lab(self, image):
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return lab_image

    def bgr2xyz(self, image):
        xyz_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
        return xyz_image

    def bgr2gray(self, image):
        gray_image = album.to_gray(image)
        return gray_image


class Augment:
    def __init__(self) -> None:
        self.tran1 = album.Compose(
            [
                album.RandomResizedCrop(224, 224, (0, 1)),
            ]
        )

        self.tran2 = album.Compose(
            [
                album.VerticalFlip(p=0.5),
                album.HorizontalFlip(p=0.5),
                album.RandomRotate90(p=0.5),
                album.Transpose(p=0.5),
                album.ShiftScaleRotate(p=0.5),
            ]
        )

        self.tran3 = album.Compose(
            [
                album.ImageCompression(p=0.5, quality_lower=98, quality_upper=100),
                album.CLAHE(p=0.5),
                album.Blur(p=0.5),
                album.GaussianBlur(p=0.5),
                album.GlassBlur(p=0.5),
                album.FancyPCA(p=0.5, alpha=10),
                album.RandomGamma(p=0.5),
            ]
        )

        self.tran4 = album.Compose(
            [
                album.RandomBrightnessContrast(p=0.5),
                album.RandomSunFlare(p=0.5),
                album.RandomRain(p=0.5),
                album.RandomSnow(p=0.5),
                album.RandomFog(p=0.5),
            ]
        )

        self.tran5 = album.Compose(
            [
                album.Affine(p=0.1),
                album.ElasticTransform(p=0.1),
            ]
        )

        self.tran6 = album.Compose(
            [
                album.RandomShadow(p=0.5),
            ]
        )


class Normalize:
    def __init__(self) -> None:
        self.para1 = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.para2 = transforms.Normalize(
            mean=[0.471, 0.448, 0.408], std=[0.234, 0.239, 0.242]
        )

        self.norm1 = transforms.Compose(
            [
                transforms.ToTensor(),
                self.para1,
            ]
        )

        self.norm2 = transforms.Compose(
            [
                transforms.ToTensor(),
                self.para2,
            ]
        )
