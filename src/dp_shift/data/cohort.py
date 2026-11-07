"""One index encounter per patient, selected before eligibility exclusions."""

from .utils import require


def describe(frame):
    return {"n": len(frame), "positive_n": int(frame.y.sum()),
            "positive_rate": float(frame.y.mean()) if len(frame) else 0.0}


def build_cohort(raw, config, descriptions=None):
    index = raw.sort_values("encounter_id").drop_duplicates("patient_nbr")
    excluded_codes = {str(code) for code in config["excluded_discharges"]}
    excluded = index.discharge_disposition_id.isin(excluded_codes)
    exclusions = index.loc[excluded, ["patient_nbr", "encounter_id", "discharge_disposition_id"]].copy()
    exclusions["reason"] = "death_or_hospice_index_encounter"
    exclusions["description"] = exclusions.discharge_disposition_id.map(descriptions or {}).fillna("")
    cohort = index.loc[~excluded].sort_values("patient_nbr").reset_index(drop=True)
    require(cohort.patient_nbr.is_unique, "Cohort contains duplicate patients")
    audit = {"raw_n": len(raw), "index_n": len(index), "excluded_n": int(excluded.sum()),
             **describe(cohort)}
    return cohort, exclusions.reset_index(drop=True), audit
