

class TrainBase:
    def __init__(
        self,
        config,
        loader_train,
        loader_validation,
        model,
        scheduler,
        criterion,
    ):
        pass

    def train_epoch(self):
        pass

    def train(self):
        pass

    def save_checkpoint(self):
        pass
