from datetime import timedelta

import lance
import pandas as pd

from src.utils.logger_setup import get_logger

logger = get_logger(__name__)


class EmbeddingsStore:
    def __init__(self, path):
        self.path = path
        self.created = False

    def cleanup(self):
        try:
            ds = lance.dataset(self.path)
        except ValueError:
            return
        time_delta = timedelta(microseconds=1)
        ds.cleanup_old_versions(older_than=time_delta)

    def write(self, data: pd.DataFrame):
        if not self.created:
            logger.debug(f"Creating lance dataset at '{self.path}'.")
            lance.write_dataset(
                data,
                self.path,
                mode="overwrite",
                data_storage_version="stable",
            )
            self.created = True
        else:
            lance.write_dataset(
                data,
                self.path,
                mode="append",
                data_storage_version="stable",
            )

    def sample(self, num_rows: int):
        ds = lance.dataset(self.path)
        df = ds.sample(num_rows).to_pandas()
        df.drop(columns="id", inplace=True)
        samples = df.to_numpy()
        return samples

    def nr_embeddings(self):
        ds = lance.dataset(self.path)
        return ds.count_rows()

    def get_embeddings_size(self):
        ds = lance.dataset(self.path)
        return len(ds.schema) - 1

    def get_batches(self, batch_size: int):
        ds = lance.dataset(self.path)
        for batch in ds.to_batches(batch_size=batch_size):
            df = batch.to_pandas()
            ids = df["id"].to_numpy()
            df.drop(columns="id", inplace=True)
            embeddings = df.to_numpy()
            yield ids, embeddings
