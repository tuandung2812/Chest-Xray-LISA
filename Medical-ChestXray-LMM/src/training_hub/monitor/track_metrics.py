import wandb
import pandas as pd


class WandbTracker:
    def __init__(self, cfg) -> None:
        wandb.login()
        wandb.init(
            project="peft-sam-ml",
            name=cfg.version_name.split("/")[-1],)

    def push(self, data: dict) -> None:
        wandb.log(data)


class MetricTracker:
    def __init__(self, *keys) -> None:
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        """
            Update metric for every iteration
        """
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        """
            Update metric for every epoch
        """
        return dict(self._data.average)
