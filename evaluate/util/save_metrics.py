"""Excel writer that appends per-model metrics into a multi-sheet workbook.

Each call writes one row (the model under test) into the sheet named
after the run id (e.g. ``en_2603__math``). The metric column layout is
preserved across calls, so the same workbook accumulates results from
many models and many runs.
"""

import os

import pandas as pd


def save_dict_to_excel(model, paper, metric_dict, save_path):
    """Persist a per-model metric row into the workbook at ``save_path``.

    Args:
        model:       Model name. Used as the row label (workbook index).
        paper:       Run id. Used as the worksheet name.
        metric_dict: Nested dict ``{group: {metric: value, ...}, ...}``;
                     groups become the upper level of a MultiIndex,
                     metrics become the lower level.
        save_path:   Destination .xlsx path. Created if it doesn't exist.
    """
    # Ensure the parent directory exists; pd.ExcelWriter will not create it.
    parent_dir = os.path.dirname(os.path.abspath(save_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Flatten the nested dict into (column-tuple, value) pairs.
    def flatten_dict(d, parent_key=""):
        items = []
        for k, v in d.items():
            if isinstance(v, dict):
                items.extend(flatten_dict(v, k))
            else:
                items.append(((parent_key, k) if parent_key else (k,), v))
        return items

    flattened = flatten_dict(metric_dict)

    # Build the column index. Two levels when the caller passes a nested
    # dict; single level otherwise.
    if flattened and len(flattened[0][0]) == 2:
        columns = pd.MultiIndex.from_tuples([item[0] for item in flattened])
        data = {model: [item[1] for item in flattened]}
    else:
        columns = [item[0][0] for item in flattened]
        data = {model: [item[1] for item in flattened]}

    df_new = pd.DataFrame(data, index=columns).T

    if os.path.exists(save_path):
        # Workbook exists: merge with whichever sheets are already there.
        try:
            with pd.ExcelFile(save_path, engine="openpyxl") as xls:
                if paper in xls.sheet_names:
                    df_existing = pd.read_excel(
                        xls,
                        sheet_name=paper,
                        header=[0, 1] if isinstance(df_new.columns, pd.MultiIndex) else 0,
                        index_col=0,
                    )

                    # Refuse to merge incompatible schemas (column headers must match).
                    if not df_existing.columns.equals(df_new.columns):
                        raise ValueError(
                            "Column-header mismatch.\n"
                            f"  existing: {df_existing.columns.tolist()}\n"
                            f"  new:      {df_new.columns.tolist()}"
                        )

                    if model in df_existing.index:
                        # Overwrite the existing row for this model.
                        df_existing.loc[model] = df_new.loc[model]
                        df_combined = df_existing
                    else:
                        df_combined = pd.concat([df_existing, df_new])

                    # Preserve all other sheets unchanged.
                    sheets_dict = {}
                    for sheet in xls.sheet_names:
                        if sheet != paper:
                            sheets_dict[sheet] = pd.read_excel(
                                xls,
                                sheet_name=sheet,
                                header=[0, 1] if isinstance(df_new.columns, pd.MultiIndex) else 0,
                                index_col=0,
                            )

                    with pd.ExcelWriter(save_path, engine="openpyxl", mode="w") as writer:
                        df_combined.to_excel(writer, sheet_name=paper)
                        for sheet_name, sheet_df in sheets_dict.items():
                            sheet_df.to_excel(writer, sheet_name=sheet_name)

                else:
                    # Sheet doesn't exist yet; append a fresh one.
                    with pd.ExcelWriter(save_path, engine="openpyxl",
                                        mode="a", if_sheet_exists="overlay") as writer:
                        df_new.to_excel(writer, sheet_name=paper)

        except Exception as e:
            print(f"Error while reading existing workbook: {e}")
            raise

    else:
        # First write: create the workbook.
        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            df_new.to_excel(writer, sheet_name=paper)

    print(f"Saved metrics for model={model!r} into sheet={paper!r} of {save_path}")


if __name__ == "__main__":
    # Minimal smoke test: write two models into one sheet.
    sample_metric_dict_a = {
        "Accuracy": {
            "accuracy": "0.85",
            "accuracy_within_16k_tokens": "0.82",
            "latency_weighted_accuracy": "0.80",
        },
        "Efficiency": {
            "throughput_tokens_per_sec": "1234.56",
            "avg_solve_time_sec": "2.34",
            "avg_response_length_tokens": "456.78",
        },
        "Process": {
            "process_accuracy": "0.90",
            "error_count": "1.23",
        },
        "Exam simulation": {
            "total_score": "85.50",
            "multiple_choice_score": "40/50",
            "non_multiple_choice_score": "45.5/50",
        },
    }

    sample_metric_dict_b = {
        "Accuracy": {
            "accuracy": "0.75",
            "accuracy_within_16k_tokens": "0.72",
            "latency_weighted_accuracy": "0.70",
        },
        "Efficiency": {
            "throughput_tokens_per_sec": "2345.67",
            "avg_solve_time_sec": "1.23",
            "avg_response_length_tokens": "345.67",
        },
        "Process": {
            "process_accuracy": "0.80",
            "error_count": "2.34",
        },
        "Exam simulation": {
            "total_score": "75.50",
            "multiple_choice_score": "35/50",
            "non_multiple_choice_score": "40.5/50",
        },
    }

    save_path = "experiment_results.xlsx"
    save_dict_to_excel("GPT-4", "sample_run", sample_metric_dict_a, save_path)
    save_dict_to_excel("GPT-3.5", "sample_run", sample_metric_dict_b, save_path)
