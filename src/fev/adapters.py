from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import datasets
import numpy as np
import pandas as pd

from .task import Task

if TYPE_CHECKING:
    import autogluon.timeseries
    import gluonts.dataset.pandas


class DatasetAdapter:
    """Convert a time series dataset into format suitable for other frameworks."""

    def convert_input_data(
        self,
        past: datasets.Dataset,
        future: datasets.Dataset,
        task: Task,
    ) -> Any:
        raise NotImplementedError


class PandasAdapter(DatasetAdapter):
    @staticmethod
    def _to_long_df(dataset: datasets.Dataset, id_column: str) -> pd.DataFrame:
        """Convert time series dataset into long DataFrame format.

        Parameters
        ----------
        dataset
            Dataset that must be converted.
        """
        df = dataset.to_pandas()
        # Equivalent to df.explode() but much faster
        length_per_item = df[df.columns.drop(id_column)[0]].apply(len).values
        df_dict = {id_column: np.repeat(df[id_column].values, length_per_item)}
        for col in df.columns:
            if col != id_column:
                df_dict[col] = np.concatenate(df[col])
        return pd.DataFrame(df_dict).astype({id_column: str})

    def convert_input_data(
        self,
        past: datasets.Dataset,
        future: datasets.Dataset,
        task: Task,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        past_df = self._to_long_df(past.remove_columns(task.static_columns), id_column=task.id_column)
        future_df = self._to_long_df(future.remove_columns(task.static_columns), id_column=task.id_column)
        if len(task.static_columns) > 0:
            static_df = past.select_columns([task.id_column] + task.static_columns).to_pandas()
            # Infer numeric dtypes if possible (e.g., object -> float), but make sure that id_column has str dtype
            static_df = static_df.infer_objects().astype({task.id_column: str})
        else:
            static_df = None
        return past_df, future_df, static_df


class GluonTSAdapter(PandasAdapter):
    """Converts dataset to format required by GluonTS.

    Optionally, this adapter can fill in missing values in the dynamic & static feature columns.
    """

    @staticmethod
    def _convert_dtypes(df: pd.DataFrame, float_dtype: str = "float32") -> pd.DataFrame:
        """Convert numeric dtypes to float32 and object dtypes to category"""
        astype_dict = {}
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                astype_dict[col] = "category"
            elif pd.api.types.is_numeric_dtype(df[col]):
                astype_dict[col] = float_dtype
        return df.astype(astype_dict)

    def convert_input_data(
        self,
        past: datasets.Dataset,
        future: datasets.Dataset,
        task: Task,
    ) -> tuple["gluonts.dataset.pandas.PandasDataset", "gluonts.dataset.pandas.PandasDataset"]:
        try:
            from gluonts.dataset.pandas import PandasDataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Please install GluonTS before using {self.__class__.__name__}")
        past_df, future_df, static_df = super().convert_input_data(past=past, future=future, task=task)

        past_df = self._convert_dtypes(past_df)
        future_df = self._convert_dtypes(future_df)
        if static_df is not None:
            static_df = self._convert_dtypes(static_df.set_index(task.id_column))

        # GluonTS uses pd.Period, which requires frequencies like 'M' instead of 'ME'
        gluonts_freq = pd.tseries.frequencies.get_period_alias(task.freq)
        # We compute names of feature columns after non-numeric columns have been removed
        feat_dynamic_real = list(future_df.columns.drop([task.id_column, task.timestamp_column]))
        past_feat_dynamic_real = list(past_df.columns.drop(list(future_df.columns) + [task.target_column]))
        past_dataset = PandasDataset.from_long_dataframe(
            past_df,
            item_id=task.id_column,
            timestamp=task.timestamp_column,
            target=task.target_column,
            static_features=static_df,
            freq=gluonts_freq,
            feat_dynamic_real=feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
        )
        prediction_dataset = PandasDataset.from_long_dataframe(
            pd.concat([past_df, future_df]),
            item_id=task.id_column,
            timestamp=task.timestamp_column,
            target=task.target_column,
            static_features=static_df,
            freq=gluonts_freq,
            future_length=task.horizon,
            feat_dynamic_real=feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
        )
        return past_dataset, prediction_dataset


class NixtlaAdapter(PandasAdapter):
    """Converts dataset to format required by StatsForecast, NeuralForecast or MLForecast.

    Returns
    -------
    past_df : pd.DataFrame
        Dataframe with columns [unique_id, ds, y] as well as all dynamic features.
    future_df : pd.DataFrame
        Dataframe with columns [unique_id, ds] as well as dynamic features that are known in the future.
    static_df : pd.DataFrame
        Dataframe containing the static (time-independent) features.
    """

    id_column: str = "unique_id"
    timestamp_column: str = "ds"
    target_column: str = "y"

    def convert_input_data(
        self,
        past: datasets.Dataset,
        future: datasets.Dataset,
        task: Task,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        past_df, future_df, static_df = super().convert_input_data(past=past, future=future, task=task)
        past_df = past_df.rename(
            columns={
                task.id_column: self.id_column,
                task.timestamp_column: self.timestamp_column,
                task.target_column: self.target_column,
            }
        )
        future_df = future_df.rename(
            columns={
                task.id_column: self.id_column,
                task.timestamp_column: self.timestamp_column,
            }
        )
        if static_df is not None:
            static_df = static_df.rename(columns={task.id_column: self.id_column})

        return past_df, future_df, static_df


class AutoGluonAdapter(PandasAdapter):
    """Converts dataset to format required by AutoGluon.

    Returns
    -------
    past_df : autogluon.timeseries.TimeSeriesDataFrame
        Dataframe containing the past values of the time series as well as all dynamic features.

        If static features are present in the dataset, they are stored as `past_df.static_features`.
    known_covariates : autogluon.timeseries.TimeSeriesDataFrame
        Dataframe containing the future values of the dynamic features that are known in the future.
    """

    def convert_input_data(
        self,
        past: datasets.Dataset,
        future: datasets.Dataset,
        task: Task,
    ) -> tuple["autogluon.timeseries.TimeSeriesDataFrame", "autogluon.timeseries.TimeSeriesDataFrame"]:
        try:
            from autogluon.timeseries import TimeSeriesDataFrame
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Please install AutoGluon before using {self.__class__.__name__}")

        past_df, future_df, static_df = super().convert_input_data(past=past, future=future, task=task)
        past_data = TimeSeriesDataFrame.from_data_frame(
            past_df,
            id_column=task.id_column,
            timestamp_column=task.timestamp_column,
            static_features_df=static_df,
        )
        known_covariates = TimeSeriesDataFrame.from_data_frame(
            future_df,
            id_column=task.id_column,
            timestamp_column=task.timestamp_column,
        )
        return past_data, known_covariates


class DartsAdapter(DatasetAdapter):
    pass


DATASET_ADAPTERS: dict[str, DatasetAdapter] = {
    "pandas": PandasAdapter,
    "gluonts": GluonTSAdapter,
    "nixtla": NixtlaAdapter,
    "darts": DartsAdapter,
    "autogluon": AutoGluonAdapter,
}


def convert_input_data(
    task: Task,
    adapter: Literal["pandas", "gluonts", "nixtla", "darts", "autogluon"] = "pandas",
    **kwargs,
) -> Any:
    """Convert the output of `task.get_input_data()` to a format compatible with popular forecasting frameworks.

    Parameters
    ----------
    task : fev.Task
        Task object for which input data must be converted.
    adapter : {"pandas", "gluonts", "nixtla", "darts", "autogluon"}
        Format to which the dataset must be converted.
    **kwargs
        Keyword arguments passed to :meth:`fev.Task.get_input_data`.
    """
    past, future = task.get_input_data(**kwargs)
    if adapter not in DATASET_ADAPTERS:
        raise KeyError(f"`adapter` must be one of {list(DATASET_ADAPTERS)}")
    return DATASET_ADAPTERS[adapter]().convert_input_data(past=past, future=future, task=task)
