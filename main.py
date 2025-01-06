import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from variables import trainloader, torchvision, classes
from predict import predict, predict_by_class
from train import train

if __name__ == "__main__":
    # For more backend options, check https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.use
    matplotlib.use("module://matplotlib-backend-kitty")

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # print images
    imshow(torchvision.utils.make_grid(images))
    print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))

    train()
    predict()
    predict_by_class()

    del dataiter
