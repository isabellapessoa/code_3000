import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    identifiers = ["age", "gender", "zip3"]
    required_anon = ["anon_id", *identifiers]
    required_aux = ["name", *identifiers]
    missing_anon = [column for column in required_anon if column not in anon_df.columns]
    missing_aux = [column for column in required_aux if column not in aux_df.columns]
    if missing_anon or missing_aux:
        missing = []
        if missing_anon:
            missing.append(f"anonymized data: {', '.join(missing_anon)}")
        if missing_aux:
            missing.append(f"auxiliary data: {', '.join(missing_aux)}")
        raise ValueError("Missing required columns (" + "; ".join(missing) + ")")

    # Only keys that identify exactly one auxiliary record can be linked.
    aux_keys = aux_df[identifiers].copy()
    aux_keys["matched_name"] = aux_df["name"]
    aux_keys["_aux_row"] = range(len(aux_keys))
    aux_counts = aux_keys.groupby(identifiers, dropna=False)["_aux_row"].transform("size")
    unique_aux = aux_keys.loc[aux_counts == 1, identifiers + ["matched_name"]]

    linked = anon_df[["anon_id", *identifiers]].merge(
        unique_aux, on=identifiers, how="inner", sort=False
    )
    return linked[["anon_id", "matched_name"]].reset_index(drop=True)


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0
    return len(matches_df) / len(anon_df)