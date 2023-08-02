import albumentations as A
from albumentations.pytorch import ToTensorV2


# Image Transforms
means = [0.4914, 0.4822, 0.4465]
stds = [0.2470, 0.2435, 0.2616]

def get_train_transforms():
    return A.Compose(
              [
                  A.Normalize(mean=means, std=stds, always_apply=True),
                  A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                  A.RandomCrop(height=32, width=32, always_apply=True),
                  A.HorizontalFlip(),
                  A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
                  ToTensorV2(),
              ]
            )
def get_test_transforms():
    return A.Compose(
      [
          A.Normalize(mean=means, std=stds, always_apply=True),
          ToTensorV2(),
      ]
    )

class Cifar10SearchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]

        if self.transform is not None:

            transformed = self.transform(image=image)

            image = transformed["image"]

        return image, label
