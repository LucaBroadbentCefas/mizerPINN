"""Authoritative catalogue for the final HPC suite.

This module deliberately contains no inference from run folder names: the task
mapping is an experimental-design constant shared by discovery and the UI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SuiteTask:
    task_id: int
    run_label: str
    family: str
    display_label: str
    species: str | None = None
    state: str | None = None
    architecture: str | None = None
    cv: float | None = None
    replicate: int | None = None
    noise_seed: int | None = None
    gate: bool | None = None
    start_value: float | None = None


def build_catalogue() -> list[SuiteTask]:
    tasks: list[SuiteTask] = []
    species = ("sp_3", "sp_7", "sp_11")
    for task_id, sp in enumerate(species):
        tasks.append(SuiteTask(task_id, f"single_{sp}_logu_fourier", "Single-species main/ablations", "log-u Fourier", sp, "log-u", "fourier"))
    ablations = (("logu_mlp", "log-u MLP", "log-u", "mlp"), ("logn_fourier", "log-n Fourier", "log-n", "fourier"), ("logn_mlp", "log-n MLP", "log-n", "mlp"))
    task_id = 3
    for sp in species:
        for tag, display, state, architecture in ablations:
            tasks.append(SuiteTask(task_id, f"single_{sp}_{tag}", "Single-species main/ablations", display, sp, state, architecture))
            task_id += 1
    tasks.extend([
        SuiteTask(12, "ms_no_data", "Multispecies baseline", "No data"),
        SuiteTask(13, "ms_perfect", "Multispecies baseline", "Perfect data"),
    ])
    cvs = (0.1, 0.2, 0.3, 0.4, 0.5)
    for cv_i, cv in enumerate(cvs):
        for rep in range(1, 6):
            tid = 14 + cv_i * 5 + rep - 1
            tasks.append(SuiteTask(tid, f"noise_cv{str(cv).replace('.', 'p')}_rep{rep}_gate_on", "Noise + discrepancy gate", f"rep{rep}", cv=cv, replicate=rep, noise_seed=41000 + rep, gate=True))
    for cv_i, cv in enumerate(cvs):
        tasks.append(SuiteTask(39 + cv_i, f"noise_cv{str(cv).replace('.', 'p')}_rep1_gate_off", "Noise + discrepancy gate", "gate OFF rep1", cv=cv, replicate=1, noise_seed=41001, gate=False))
    missing = (
        ("data_every_3yr", "Every third year"), ("data_gap_10_20", "Gap years 10–20"),
        ("data_gap_30_40", "Gap years 30–40"), ("data_missing_species_3", "Missing species 3"),
        ("data_missing_species_7", "Missing species 7"), ("data_missing_species_11", "Missing species 11"),
    )
    tasks.extend(SuiteTask(44 + i, label, "Missing/sparse data", display) for i, (label, display) in enumerate(missing))
    tasks.extend([
        SuiteTask(50, "lambdaPDE0_perfect", "No-PDE NN", "Perfect data"),
        SuiteTask(51, "lambdaPDE0_noise_cv0p3_rep1", "No-PDE NN", "CV 0.3 rep1", cv=0.3, replicate=1, noise_seed=41001),
        SuiteTask(52, "lambdaPDE0_gap_30_40", "No-PDE NN", "Gap years 30–40"),
    ])
    for i, factor in enumerate((0.25, 0.5, 2.0, 4.0)):
        tasks.append(SuiteTask(53 + i, f"rmax_start_factor_{str(factor).replace('.', 'p')}", "Rmax recovery", f"Perfect, start {factor:g}×", start_value=factor))
    for rep in range(1, 6):
        tasks.append(SuiteTask(56 + rep, f"rmax_noise_cv0p3_rep{rep}_start0p5", "Rmax recovery", f"CV 0.3 rep{rep}, start 0.5×", cv=0.3, replicate=rep, noise_seed=41000 + rep, start_value=0.5))
    for i, start in enumerate((0.1, 0.3, 0.6, 1.0)):
        tasks.append(SuiteTask(62 + i, f"cv_recovery_true0p3_start{str(start).replace('.', 'p')}", "CV recovery", f"Start {start:g}", cv=0.3, start_value=start))
    for i, start in enumerate((0.05, 0.5, 1.0, 3.0)):
        tasks.append(SuiteTask(66 + i, f"effort_recovery_start{str(start).replace('.', 'p')}", "Effort recovery", f"Start {start:g}", start_value=start))
    assert [task.task_id for task in tasks] == list(range(70))
    return tasks


CATALOGUE = tuple(build_catalogue())
BY_TASK_ID = {task.task_id: task for task in CATALOGUE}
BY_LABEL = {task.run_label: task for task in CATALOGUE}


def catalogue_records() -> list[dict[str, Any]]:
    return [asdict(task) for task in CATALOGUE]
