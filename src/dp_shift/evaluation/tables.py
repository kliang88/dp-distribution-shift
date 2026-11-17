"""The four flat CSV tables shared by experiments, sweeps, and plotting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dp_shift.config import ResolvedConfig

from dp_shift.data.utils import read_json, require
from dp_shift.data.shifts import SHIFTS
from dp_shift.stages.common import atomic_path
from .metrics import CONDITIONS, curve_key, validate_curves
from .summarize import validate_runs


COLUMNS = {
    "runs": ["run_id", "repeat", "method", "pretraining", "shift_type", "shift_fraction", "source_fraction", "source_prevalence", "source_size",
             "comparison_epsilon", "epsilon_achieved", "auprc", "auroc", "brier", "calibration_bias", "test_fraction", "test_prevalence"],
    "precision_recall": ["run_id", "point", "recall", "precision"],
    "calibration": ["run_id", "bin", "patient_count", "positive_count", "mean_prediction"],
    "risk_distribution": ["run_id", "bin_lower", "bin_upper", "patient_count"],
}
FILES = tuple(f"data/{name}.csv" for name in COLUMNS)
PRETRAINING = {"scratch": "none", "s0": "baseline", "s1": "shifted"}
PATHS = {(method, PRETRAINING[initialization]): path for path, (method, initialization) in CONDITIONS.items()}
GRID = ["shift_type", "shift_fraction", "comparison_epsilon"]


def export(runs, curves, source_pairs, config: ResolvedConfig, privacy, *, test_fraction=None):
    """Convert existing numerical results without changing calculations or precision."""
    rates = {(row.repeat_id, row.condition): row.positive_rate for row in source_pairs.itertuples()}
    fraction_column = "shift_fraction" if "shift_fraction" in source_pairs else "positive_rate"
    fractions = {(row.repeat_id, row.condition): getattr(row, fraction_column) for row in source_pairs.itertuples()}
    require(test_fraction is not None or config.requested.source_conditions.shift_type == "readmission", "Test shift fraction is required")
    private = {(row.repeat_id, row.path_id): row._asdict() for row in privacy.itertuples(index=False)}
    rows = {name: [] for name in COLUMNS}
    ledger = {}
    for row in runs.itertuples():
        pretraining = PRETRAINING[row.initialization]
        run_id = f"r{row.repeat_id:02d}_{row.method}_{pretraining}"
        account = private.get((row.repeat_id, row.path_id))
        if account is not None:
            ledger[run_id] = {key: value for key, value in account.items() if key not in {"repeat_id", "path_id"}}
            ledger[run_id]["accountant_history"] = json.loads(account["accountant_history"])
        rows["runs"].append([run_id, row.repeat_id, row.method, pretraining,
                             config.requested.source_conditions.shift_type, config.requested.source_conditions.shift_fraction, fractions.get((row.repeat_id, row.initialization)),
                             rates.get((row.repeat_id, row.initialization)),
                             config.requested.source_conditions.pretrain_size if pretraining != "none" else None,
                             config.requested.privacy.epsilon, account["epsilon_achieved"] if account else None,
                             row.average_precision, row.auroc, row.brier, row.calibration_bias,
                             row.test_prevalence if test_fraction is None else test_fraction, row.test_prevalence])
        values = {field: curves[curve_key(row.repeat_id, row.path_id, field)] for field in (
            "precision", "recall", "calibration_n", "calibration_positive_n", "calibration_mean_prediction",
            "histogram_edges", "histogram_n")}
        rows["precision_recall"].extend([run_id, i, recall, precision] for i, (recall, precision) in
                                        enumerate(zip(values["recall"], values["precision"])))
        rows["calibration"].extend([run_id, i, int(n), int(positive), prediction] for i, (n, positive, prediction) in
                                   enumerate(zip(values["calibration_n"], values["calibration_positive_n"],
                                                 values["calibration_mean_prediction"])))
        rows["risk_distribution"].extend([run_id, lower, upper, int(n)] for lower, upper, n in
                                         zip(values["histogram_edges"][:-1], values["histogram_edges"][1:], values["histogram_n"]))
    return {name: pd.DataFrame(values, columns=COLUMNS[name]) for name, values in rows.items()}, ledger


def write(output, tables):
    for name, columns in COLUMNS.items():
        require(list(tables[name].columns) == columns, f"Unexpected {name} columns")
        with atomic_path(Path(output) / f"data/{name}.csv") as temporary:
            tables[name].to_csv(temporary, index=False)


def read(input):
    directory = Path(input)
    if not (directory / "runs.csv").exists():
        directory = directory / "data"
    tables = {name: pd.read_csv(directory / f"{name}.csv", float_precision="round_trip") for name in COLUMNS}
    runs = tables["runs"]
    if "shift_prevalence" in runs:
        runs = runs.rename(columns={"shift_prevalence": "shift_fraction"})
        runs["shift_type"] = "readmission"
        runs["source_fraction"] = runs.source_prevalence
        runs["test_fraction"] = runs.test_prevalence
    if "source_size" not in runs:
        metadata = directory.parent / "protocol.json"
        require(metadata.is_file(), "runs.csv needs source_size; older exports also need their protocol.json")
        saved = read_json(metadata)
        size = (saved["stages"]["prepare"]["resolved"]["pretrain_size"] if "stages" in saved
                else saved["plan"]["base"]["source_conditions"]["pretrain_size"])
        runs["source_size"] = np.where(runs.pretraining.eq("none"), np.nan, size)
    require(set(runs.columns) == set(COLUMNS["runs"]), "Unexpected runs columns")
    tables["runs"] = runs[COLUMNS["runs"]]
    validate(tables)
    return tables


def cells(tables):
    for key, runs in tables["runs"].groupby(GRID, sort=True):
        ids = set(runs.run_id)
        yield key, {name: table.loc[table.run_id.isin(ids)] for name, table in tables.items()}


def internal(tables):
    """Adapt one cell to the established plotting/statistics functions."""
    groups = {name: table.groupby("run_id", sort=False) for name, table in tables.items() if name != "runs"}
    rows, curves, references = [], {}, []
    for row in tables["runs"].itertuples():
        path = PATHS[row.method, row.pretraining]
        pr = groups["precision_recall"].get_group(row.run_id).sort_values("point")
        bins = groups["calibration"].get_group(row.run_id).sort_values("bin")
        histogram = groups["risk_distribution"].get_group(row.run_id).sort_values("bin_lower")
        require(np.array_equal(pr.point, np.arange(len(pr))), "PR point order must be consecutive")
        require(np.array_equal(bins.bin, np.arange(len(bins))), "Calibration bin order must be consecutive")
        for table, fields in ((bins, ("patient_count", "positive_count")), (histogram, ("patient_count",))):
            for field in fields:
                values = table[field].to_numpy()
                require(np.isfinite(values).all() and (values >= 0).all() and (values == np.floor(values)).all(),
                        f"Invalid {field} counts")
        require(np.allclose(histogram.bin_upper.to_numpy()[:-1], histogram.bin_lower.to_numpy()[1:], rtol=0, atol=1e-12),
                "Risk histogram bins are not contiguous")
        n, positive = int(bins.patient_count.sum()), int(bins.positive_count.sum())
        require(n > 0, "Calibration data must contain patients")
        prediction = np.average(bins.mean_prediction, weights=bins.patient_count)
        rows.append(dict(run_id=row.run_id, repeat_id=row.repeat, path_id=path, status="complete", method=row.method,
                         initialization=CONDITIONS[path][1], average_precision=row.auprc, auroc=row.auroc, brier=row.brier,
                         calibration_bias=row.calibration_bias, mean_prediction=prediction, test_prevalence=row.test_prevalence,
                         shift_type=row.shift_type, shift_fraction=row.shift_fraction, grid_epsilon=row.comparison_epsilon,
                         source_fraction=row.source_fraction, test_fraction=row.test_fraction,
                         source_positive_rate=row.source_prevalence, epsilon_achieved=row.epsilon_achieved))
        values = dict(precision=pr.precision.to_numpy(), recall=pr.recall.to_numpy(),
                      calibration_n=bins.patient_count.to_numpy(dtype=np.int64),
                      calibration_positive_n=bins.positive_count.to_numpy(dtype=np.int64),
                      calibration_mean_prediction=bins.mean_prediction.to_numpy(),
                      calibration_observed_rate=bins.positive_count.to_numpy() / bins.patient_count.to_numpy(),
                      histogram_edges=np.r_[histogram.bin_lower.to_numpy(), histogram.bin_upper.iloc[-1]],
                      histogram_n=histogram.patient_count.to_numpy(dtype=np.int64))
        curves.update({curve_key(row.repeat, path, field): value for field, value in values.items()})
        references.append((n, positive))
    require(len(set(references)) == 1, "Models have different test counts")
    n, positive = references[0]
    prevalence = positive / n
    reference = dict(n=n, positive_n=positive, test_prevalence=prevalence,
                     constant_ap=prevalence, constant_auroc=.5, constant_brier=prevalence * (1 - prevalence))
    return pd.DataFrame(rows), curves, reference


def validate(tables):
    for name, columns in COLUMNS.items():
        require(list(tables[name].columns) == columns and len(tables[name]) > 0, f"Invalid {name} table")
        numeric = [field for field in columns if field not in {"run_id", "method", "pretraining", "shift_type"}]
        require(all(pd.api.types.is_numeric_dtype(tables[name][field])
                    and not pd.api.types.is_bool_dtype(tables[name][field]) for field in numeric),
                f"Non-numeric values in {name}")
        require(tables[name].run_id.map(lambda value: isinstance(value, str) and bool(value)).all(),
                f"Invalid run IDs in {name}")
    runs = tables["runs"]
    require(runs.run_id.notna().all() and runs.run_id.is_unique, "Run IDs must be present and unique")
    require(set(zip(runs.method, runs.pretraining)) <= set(PATHS), "Unknown method or pretraining")
    require(np.isfinite(runs[GRID[1:]].to_numpy()).all() and runs.shift_fraction.between(0, 1, inclusive="neither").all()
            and runs.comparison_epsilon.gt(0).all(), "Invalid shift fraction or comparison epsilon")
    require(runs.shift_type.isin(SHIFTS).all(), "Unknown shift type")
    require(runs.test_fraction.between(0, 1).all(), "Invalid test fraction")
    require(runs.repeat.ge(0).all() and runs.repeat.eq(np.floor(runs.repeat)).all(), "Invalid repeat numbers")
    scratch, private = runs.pretraining.eq("none"), runs.method.eq("dp_sgd")
    require(runs.loc[scratch, "source_prevalence"].isna().all()
            and runs.loc[~scratch, "source_prevalence"].between(0, 1, inclusive="neither").all(), "Invalid source prevalence")
    require(runs.loc[scratch, "source_fraction"].isna().all()
            and runs.loc[~scratch, "source_fraction"].between(0, 1).all(), "Invalid source fraction")
    sizes = runs.loc[~scratch, "source_size"]
    require(runs.loc[scratch, "source_size"].isna().all() and np.isfinite(sizes).all()
            and sizes.ge(2).all() and sizes.eq(np.floor(sizes)).all(), "Invalid source size")
    positives = sizes * runs.loc[~scratch, "source_prevalence"]
    members = sizes * runs.loc[~scratch, "source_fraction"]
    require(np.allclose(positives, np.round(positives), rtol=0, atol=1e-9)
            and np.allclose(members, np.round(members), rtol=0, atol=1e-9), "Source counts must be integral")
    outcome_shift = runs.shift_type.eq("readmission")
    require(np.allclose(runs.loc[outcome_shift & ~scratch, "source_fraction"],
                        runs.loc[outcome_shift & ~scratch, "source_prevalence"], rtol=0, atol=1e-12)
            and np.allclose(runs.loc[outcome_shift, "test_fraction"], runs.loc[outcome_shift, "test_prevalence"], rtol=0, atol=1e-12),
            "Readmission fractions disagree with outcome prevalence")
    require(runs.loc[~private, "epsilon_achieved"].isna().all()
            and runs.loc[private, "epsilon_achieved"].gt(0).all()
            and runs.loc[private, "epsilon_achieved"].le(runs.loc[private, "comparison_epsilon"]).all(), "Invalid achieved epsilon")
    for name, table in tables.items():
        require(table.run_id.notna().all() and set(table.run_id) == set(runs.run_id), f"{name} run IDs differ")
        fields = [column for column in table if column != "run_id"]
        if name != "runs":
            require(np.isfinite(table[fields].to_numpy()).all(), f"Nonfinite {name} data")
    test_counts, trait_counts = set(), {}
    for (kind, _, _), cell in cells(tables):
        scores, curves, reference = internal(cell)
        repeats = scores.repeat_id.nunique()
        require(repeats >= 1, "At least one repeat is required")
        validate_runs(scores, repeats)
        validate_curves(curves, scores, reference)
        require(np.allclose(scores.test_prevalence, reference["test_prevalence"], rtol=0, atol=1e-12), "Test prevalence mismatch")
        shifted = cell["runs"].loc[cell["runs"].pretraining.eq("shifted")]
        require(np.allclose(shifted.source_fraction, shifted.shift_fraction, rtol=0, atol=1e-12), "Shifted source fraction mismatch")
        fractions = cell["runs"].test_fraction
        expected = trait_counts.setdefault(kind, fractions.iloc[0])
        require(np.allclose(fractions, expected, rtol=0, atol=1e-12), "Test characteristic changed across cells")
        require(np.isclose(expected * reference["n"], round(expected * reference["n"]), rtol=0, atol=1e-9),
                "Test member count must be integral")
        test_counts.add((reference["n"], reference["positive_n"]))
    require(len(test_counts) == 1, "Grid cells have different test counts")


def privacy_frame(tables, ledger):
    """Keep accounting details in metadata, joining only for verification."""
    private = tables["runs"].loc[tables["runs"].method.eq("dp_sgd")]
    require(set(ledger) == set(private.run_id), "Missing or extra privacy records")
    rows = []
    for row in private.itertuples():
        account = ledger[row.run_id]
        require(abs(account["epsilon_achieved"] - row.epsilon_achieved) <= 1e-12, "Privacy epsilon mismatch")
        rows.append({"run_id": row.run_id, **account, "accountant_history": json.dumps(account["accountant_history"])})
    return pd.DataFrame(rows)
