from model import Cnn

class App:

    def __init__(self):

        self.model = Cnn()

    def run(self):

        self.model.train_model()


if __name__ == '__main__':
    app = App()
    app.run()
