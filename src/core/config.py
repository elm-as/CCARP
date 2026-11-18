"""Immutable configuration dataclasses for the CCARP credit risk pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for Kaggle dataset acquisition and local data paths."""

    kaggle_slug: str
    kaggle_config_dir: Path
    kaggle_exe: Path
    raw_data_dir: Path
    processed_data_dir: Path
    application_filename: str = "application_record.csv"
    credit_filename: str = "credit_record.csv"


@dataclass(frozen=True)
class TemporalWindowConfig:
    """Configuration for strict pre-cutoff observation and future performance windows."""

    t_obs: int = -12
    min_obs_months: int = 3
    min_perf_months: int = 6
    default_statuses: tuple[str, ...] = ("2", "3", "4", "5")
    cutoff_grid: tuple[int, ...] = (-24, -18, -12, -6)


@dataclass(frozen=True)
class ModelingConfig:
    """Configuration for grouped cross-validation and OOF probability calibration."""

    n_splits: int = 5
    inner_cal_splits: int = 3
    random_seed: int = 42
    model_names: tuple[str, ...] = (
        "logistic_regression",
        "lightgbm",
        "xgboost",
        "catboost",
    )


@dataclass(frozen=True)
class EconomicsConfig:
    """Unit-economics parameters for credit-card expected profit optimization."""

    margin_good: float = 180.0
    lgd: float = 0.75
    exposure_at_default: float = 2000.0
    threshold_step: float = 0.002
    sensitivity_margins: tuple[float, ...] = (120.0, 150.0, 180.0, 220.0, 260.0)
    sensitivity_lgds: tuple[float, ...] = (0.60, 0.70, 0.75, 0.85)


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for 300 DPI static charts and institutional Plotly dashboard."""

    images_dir: Path
    dashboard_path: Path
    results_json_path: Path
    dpi: int = 300
    bg_page: str = "#F7F8FA"
    bg_surface: str = "#FFFFFF"
    border_color: str = "#D8DEE6"
    ink_primary: str = "#0F172A"
    ink_muted: str = "#475569"
    navy: str = "#0F2942"
    emerald: str = "#0E7C5B"
    amber: str = "#B45309"
    crimson: str = "#B91C1C"
    steel_blue: str = "#1D4ED8"


@dataclass(frozen=True)
class PipelineConfig:
    """Master immutable configuration aggregating all CCARP domain configs."""

    root_dir: Path
    dataset: DatasetConfig
    temporal: TemporalWindowConfig
    modeling: ModelingConfig
    economics: EconomicsConfig
    plots: PlotConfig

    @classmethod
    def create_default(cls, root_dir: Path) -> PipelineConfig:
        """Build the canonical CCARP pipeline configuration rooted at root_dir."""
        resolved = root_dir.resolve()
        dataset_cfg = DatasetConfig(
            kaggle_slug="rikdifos/credit-card-approval-prediction",
            kaggle_config_dir=Path(r"c:\Users\elmas\Desktop\Projets\DataScience\PCCF"),
            kaggle_exe=Path(
                r"C:\Users\elmas\AppData\Local\Programs\Python\Python310\Scripts\kaggle.exe"
            ),
            raw_data_dir=resolved / "data" / "raw",
            processed_data_dir=resolved / "data" / "processed",
        )
        plot_cfg = PlotConfig(
            images_dir=resolved / "images",
            dashboard_path=resolved / "artifacts" / "dashboard.html",
            results_json_path=resolved / "artifacts" / "results.json",
        )
        return cls(
            root_dir=resolved,
            dataset=dataset_cfg,
            temporal=TemporalWindowConfig(),
            modeling=ModelingConfig(),
            economics=EconomicsConfig(),
            plots=plot_cfg,
        )
