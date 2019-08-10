import matplotlib.pyplot as plt


class Gui:

    def __init__(self):
        self._index = 1

    def add(self, image, title=""):
        plt.subplot(3, 3, self._index), plt.imshow(image, 'gray')
        plt.title(title)
        self._index += 1

    def show(self):
        plt.show()
