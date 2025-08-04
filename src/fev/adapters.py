from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, Type

import datasets
import numpy as np
import pandas as pd

from . import utils
from .task import Task

if TYPE_CHECKING:
    import autogluon.timeseries
    import gluonts.dataset.pandas


class DatasetAdapter(ABC):
    """Convert a time series dataset into format suitable for other frameworks."""

    @classmethod
    @abstractmethod
    def convert_input_data(
        cls,
        past: datasets.Dataset,
        future: datasets.Dataset,
        *,
        target_column: str | list[str],
        id_column: str,
        timestamp_column: str,
        static_columns: list[str],
    ) -> Any:
        """Convert the input data of the task into a format compatible with the framework."""
        pass


class DatasetsAdapter(DatasetAdapter):
    """Keeps data formatted as datasets.Dataset objects."""

    @classmethod
    def convert_input_data(
        cls,
        past: datasets.Dataset,
        future: datasets.Dataset,
        *,
        target_column: str | list[str],
        id_column: str,
        timestamp_column: str,
        static_columns: list[str],
    ) -> tuple[datasets.Dataset, datasets.Dataset]:
        return past, future


class PandasAdapter(DatasetAdapter):
    """Converts data to pandas.DataFrame objects."""

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

    @classmethod
    def convert_input_data(
        cls,
        past: datasets.Dataset,
        future: datasets.Dataset,
        *,
        target_column: str | list[str],
        id_column: str,
        timestamp_column: str,
        static_columns: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        past_df = cls._to_long_df(past.remove_columns(static_columns), id_column=id_column)
        future_df = cls._to_long_df(future.remove_columns(static_columns), id_column=id_column)
        if len(static_columns) > 0:
            static_df = past.select_columns([id_column] + static_columns).to_pandas()
            # Infer numeric dtypes if possible (e.g., object -> float), but make sure that id_column has str dtype
            static_df = static_df.infer_objects().astype({id_column: str})
        else:
            static_df = None
        return past_df, future_df, static_df


class GluonTSAdapter(PandasAdapter):
    """Converts dataset to format required by GluonTS."""

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

    @classmethod
    def convert_input_data(
        cls,
        past: datasets.Dataset,
        future: datasets.Dataset,
        *,
        target_column: str | list[str],
        id_column: str,
        timestamp_column: str,
        static_columns: list[str],
    ) -> tuple["gluonts.dataset.pandas.PandasDataset", "gluonts.dataset.pandas.PandasDataset"]:
        try:
            from gluonts.dataset.pandas import PandasDataset
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Please install GluonTS before using {cls.__name__}")
        assert isinstance(target_column, str), f"{cls.__name__} does not support multivariate tasks."

        past_df, future_df, static_df = super().convert_input_data(
            past=past,
            future=future,
            target_column=target_column,
            id_column=id_column,
            timestamp_column=timestamp_column,
            static_columns=static_columns,
        )

        past_df = cls._convert_dtypes(past_df)
        future_df = cls._convert_dtypes(future_df)
        if static_df is not None:
            static_df = cls._convert_dtypes(static_df.set_index(id_column))
        else:
            static_df = pd.DataFrame()

        # GluonTS needs to know the data frequency, we infer it from the timestamps
        freq = pd.infer_freq(np.concatenate([past[0][timestamp_column], future[0][timestamp_column]]))
        # GluonTS uses pd.Period, which requires frequencies like 'M' instead of 'ME'
        gluonts_freq = pd.tseries.frequencies.get_period_alias(freq)
        # We compute names of feature columns after non-numeric columns have been removed
        feat_dynamic_real = list(future_df.columns.drop([id_column, timestamp_column]))
        past_feat_dynamic_real = list(past_df.columns.drop(list(future_df.columns) + [target_column]))
        past_dataset = PandasDataset.from_long_dataframe(
            past_df,
            item_id=id_column,
            timestamp=timestamp_column,
            target=target_column,
            static_features=static_df,
            freq=gluonts_freq,
            feat_dynamic_real=feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
        )
        prediction_dataset = PandasDataset.from_long_dataframe(
            pd.concat([past_df, future_df]),
            item_id=id_column,
            timestamp=timestamp_column,
            target=target_column,
            static_features=static_df,
            freq=gluonts_freq,
            future_length=len(future[0][timestamp_column]),
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

    @classmethod
    def convert_input_data(
        cls,
        past: datasets.Dataset,
        future: datasets.Dataset,
        *,
        target_column: str | list[str],
        id_column: str,
        timestamp_column: str,
        static_columns: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        assert isinstance(target_column, str), f"{cls.__name__} does not support multivariate tasks."

        past_df, future_df, static_df = super().convert_input_data(
            past=past,
            future=future,
            target_column=target_column,
            id_column=id_column,
            timestamp_column=timestamp_column,
            static_columns=static_columns,
        )
        past_df = past_df.rename(
            columns={
                id_column: cls.id_column,
                timestamp_column: cls.timestamp_column,
                target_column: cls.target_column,
            }
        )
        future_df = future_df.rename(
            columns={
                id_column: cls.id_column,
                timestamp_column: cls.timestamp_column,
            }
        )
        if static_df is not None:
            static_df = static_df.rename(columns={id_column: cls.id_column})

        return past_df, future_df, static_df


class AutoGluonAdapter(PandasAdapter):
    """Converts dataset to format required by AutoGluon.

    Returns
    -------
    past_data : autogluon.timeseries.TimeSeriesDataFrame
        Dataframe containing the past values of the time series as well as all dynamic features.

        Target column is always renamed to "target".

        If static features are present in the dataset, they are stored as `past_data.static_features`.
    known_covariates : autogluon.timeseries.TimeSeriesDataFrame
        Dataframe containing the future values of the dynamic features that are known in the future.
    """

    target_column: str = "target"

    @classmethod
    def convert_input_data(
        cls,
        past: datasets.Dataset,
        future: datasets.Dataset,
        *,
        target_column: str | list[str],
        id_column: str,
        timestamp_column: str,
        static_columns: list[str],
    ) -> tuple["autogluon.timeseries.TimeSeriesDataFrame", "autogluon.timeseries.TimeSeriesDataFrame"]:
        try:
            from autogluon.timeseries import TimeSeriesDataFrame
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Please install AutoGluon before using {cls.__name__}")
        assert isinstance(target_column, str), f"{cls.__name__} does not support multivariate tasks."

        past_df, future_df, static_df = super().convert_input_data(
            past=past,
            future=future,
            target_column=target_column,
            id_column=id_column,
            timestamp_column=timestamp_column,
            static_columns=static_columns,
        )
        past_data = TimeSeriesDataFrame.from_data_frame(
            past_df.rename(columns={target_column: cls.target_column}),
            id_column=id_column,
            timestamp_column=timestamp_column,
            static_features_df=static_df,
        )
        known_covariates = TimeSeriesDataFrame.from_data_frame(
            future_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
        )
        return past_data, known_covariates


class DartsAdapter(DatasetAdapter):
    pass


DATASET_ADAPTERS: dict[str, Type[DatasetAdapter]] = {
    "pandas": PandasAdapter,
    "datasets": DatasetsAdapter,
    "gluonts": GluonTSAdapter,
    "nixtla": NixtlaAdapter,
    "darts": DartsAdapter,
    "autogluon": AutoGluonAdapter,
}


def convert_input_data(
    task: Task,
    adapter: Literal["pandas", "datasets", "gluonts", "nixtla", "darts", "autogluon"] = "pandas",
    *,
    as_univariate: bool = False,
    univariate_target_column: str = "target",
    **kwargs,
) -> Any:
    """Convert the output of `task.get_input_data()` to a format compatible with popular forecasting frameworks.

    Parameters
    ----------
    task
        Task object for which input data must be converted.
    adapter : {"pandas", "datasets", "gluonts", "nixtla", "darts", "autogluon"}
        Format to which the dataset must be converted.
    as_univariate
        If True, separate instances will be created from each target column before passing the data to the adapter.

        Equivalent to setting `generate_univariate_targets_from = "__ALL__"` in `Task` constructor.
    univariate_target_column
        Target column name used when as_univariate=True. Only used by the "datasets" adapter.
    **kwargs
        Keyword arguments passed to :meth:`fev.Task.get_input_data`.
    """
    past, future = task.get_input_data(**kwargs)

    if as_univariate:
        if univariate_target_column in past.column_names and univariate_target_column != task.target_column:
            raise ValueError(
                f"Column '{univariate_target_column}' already exists. Choose a different univariate_target_column."
            )
        target_column = univariate_target_column
        if task.is_multivariate:
            past = utils.generate_univariate_targets_from_multivariate(
                past,
                id_column=task.id_column,
                new_target_column=target_column,
                generate_univariate_targets_from=task.target_columns_list,
            )
            # We cannot apply generate_univariate_targets_from_multivariate to future since it does not contain target cols,
            # so we just repeat each entry and insert the IDs from past, repeating entries as [0, 0, ..., 1, 1, ..., N -1, N - 1, ...]
            original_column_order = future.column_names
            future = future.select([i for i in range(len(future)) for _ in range(len(task.target_columns_list))])
            future = future.remove_columns(task.id_column).add_column(name=task.id_column, column=past[task.id_column])
            future = future.select_columns(original_column_order)
        else:
            if target_column not in past.column_names:
                past = past.rename_column(task.target_column, target_column)
    else:
        target_column = task.target_column

    if adapter not in DATASET_ADAPTERS:
        raise KeyError(f"`adapter` must be one of {list(DATASET_ADAPTERS)}")
    adapter_cls = DATASET_ADAPTERS[adapter]

    return adapter_cls().convert_input_data(
        past=past,
        future=future,
        target_column=target_column,
        id_column=task.id_column,
        timestamp_column=task.timestamp_column,
        static_columns=task.static_columns,
    )
