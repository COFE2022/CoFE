import datasets

from .. import BaseWrapperDataset


class Seq2SeqDataset(BaseWrapperDataset):
    def __init__(self, dataset: datasets.Dataset,
                 source_column: str = None,
                 target_column: str = None):
        super().__init__(dataset)
        self._source_column = source_column
        self._target_column = target_column

    @property
    def source_column(self):
        return self.source_column

    @property
    def target_column(self):
        return self.target_column

    @source_column.setter
    def source_column(self, value):
        self._source_column = value

    @target_column.setter
    def target_column(self, value):
        self._target_column = value
