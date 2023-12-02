import pandas as pd


class Move:
    """Process data.

    Yes a long one.
    """

    def __init__(self, data: str) -> None:
        """Do something nice."""
        self.data = pd.read_csv(data)

        self.columns = list(data.columns)
        self.index = list(data.index)

    def __getitem__(self, item: str) -> str:
        """Do something nice squared."""
        return self.data[item]
