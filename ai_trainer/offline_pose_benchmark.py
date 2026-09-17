"""Research-only reference leave-one-out benchmark; never imported by production."""
from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .dtw_compare import phase_aware_weighted_dtw, resolve_weights
from .features import extract_all_features
from .angle_domain_diagnostic import frame_angles
from .two_d_diagnostic import extract_2d_features


def stable_derivative(values: np.ndarray, *, smooth: bool = False) -> np.ndarray:
    """Keogh-style local derivative, with optional fixed [1,2,1]/4 smoothing."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("derivative input must be finite (T,D) with T >= 3")
    if smooth:
        padded = np.pad(x, ((1, 1), (0, 0)), mode="edge")
        x = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
    out = np.empty_like(x)
    out[0] = x[1] - x[0]
    out[-1] = x[-1] - x[-2]
    out[1:-1] = ((x[1:-1] - x[:-2]) + (x[2:] - x[:-2]) / 2.0) / 2.0
    return out


def derivative_features(features: dict[str, np.ndarray], *, smooth=False) -> dict[str, np.ndarray]:
    return {name: stable_derivative(value, smooth=smooth) for name, value in features.items()}


def geometry_vector(coords3d: np.ndarray, raw2d: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    coords = np.asarray(coords3d, dtype=float)
    if coords.ndim != 3 or coords.shape[1:] != (18, 3) or len(coords) < 3 or not np.isfinite(coords).all():
        raise ValueError("3D geometry input must be finite (T,18,3) with T >= 3")
    angles = [frame_angles(row) for row in coords]
    hip_lr = np.asarray([row["hip_lr"] for row in angles]); knee_lr = np.asarray([row["knee_lr"] for row in angles])
    head = min(5, len(coords)); hip_stand = np.median(hip_lr[:head], axis=0); knee_stand = np.median(knee_lr[:head], axis=0)
    hip_min = np.min(hip_lr, axis=0); knee_min = np.min(knee_lr, axis=0)
    feat = extract_all_features(coords)
    pelvis = np.asarray(feat["pelvis_trajectory"])[:, 0]
    torso = np.asarray(feat["torso_inclination"])[:, 0] * 180.0
    ankle = np.asarray(feat["ankle_angle"]) * 180.0
    names = [
        "standing_hip", "standing_knee", "bottom_hip", "bottom_knee",
        "hip_excursion", "knee_excursion", "hip_lr_excursion_diff",
        "knee_lr_excursion_diff", "pelvis_excursion", "torso_bottom",
        "torso_excursion", "ankle_bottom",
    ]
    values = [
        hip_stand.mean(), knee_stand.mean(), hip_min.mean(), knee_min.mean(),
        hip_stand.mean() - hip_lr.mean(axis=1).min(),
        knee_stand.mean() - knee_lr.mean(axis=1).min(),
        abs((hip_stand[0]-hip_min[0])-(hip_stand[1]-hip_min[1])),
        abs((knee_stand[0]-knee_min[0])-(knee_stand[1]-knee_min[1])),
        np.ptp(pelvis), np.max(torso), np.ptp(torso), np.min(ankle.mean(axis=1)),
    ]
    if raw2d is not None:
        _, summary = extract_2d_features(raw2d)
        names += ["heel_relative_motion_2d", "heel_ankle_motion_2d", "heel_toe_delta_2d"]
        values += [summary["heel_relative_motion"],
                   max(summary["left_heel_ankle_motion"], summary["right_heel_ankle_motion"]),
                   max(abs(summary["left_heel_toe_delta"]), abs(summary["right_heel_toe_delta"]))]
    return np.asarray(values, dtype=float), names


def robust_location_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(matrix, dtype=float)
    median = np.median(x, axis=0)
    iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    fallback = np.maximum(np.abs(median) * 0.05, 1e-6)
    return median, np.maximum(iqr, fallback)


def confusion_and_metrics(y_true, predictions, labels):
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, pred in zip(y_true, predictions): matrix[index[truth], index[pred]] += 1
    per_class = {}
    for label, i in index.items():
        tp = matrix[i, i]; fp = matrix[:, i].sum()-tp; fn = matrix[i, :].sum()-tp
        precision = tp/max(tp+fp, 1); recall = tp/max(tp+fn, 1)
        f1 = 2*precision*recall/max(precision+recall, 1e-12)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[i].sum())}
    return matrix, {"accuracy": float(np.trace(matrix)/max(matrix.sum(), 1)),
                    "macro_precision": float(np.mean([x["precision"] for x in per_class.values()])),
                    "macro_recall": float(np.mean([x["recall"] for x in per_class.values()])),
                    "macro_f1": float(np.mean([x["f1"] for x in per_class.values()])),
                    "per_class": per_class}


def _stats(values):
    a = np.asarray(values, dtype=float)
    keys = ("min", "p10", "median", "p90", "max", "mean", "std")
    vals = (*np.percentile(a, (0, 10, 50, 90, 100)), np.mean(a), np.std(a))
    return {k: float(v) for k, v in zip(keys, vals)}


class OfflinePoseBenchmark:
    """Operational-reference LOO evaluator for DTW, DDTW and geometry."""

    def __init__(self, root: str | Path, raw2d_by_id: dict[str, np.ndarray] | None = None,
                 db_dir: str | Path | None = None):
        self.root = Path(root); self.db_dir = Path(db_dir) if db_dir is not None else self.root/"output"/"reference_db"
        self.manifest = json.loads((self.db_dir/"manifest.json").read_text(encoding="utf-8"))["entries"]
        self.config = json.loads((self.root/"configs"/"dtw_feature_weights.json").read_text(encoding="utf-8"))
        arrays = np.load(self.db_dir/"sequences.npz")
        self.samples = []
        for entry in self.manifest:
            if entry["tier"] != "operational": continue
            coords = np.asarray(arrays[entry["array_key"]], dtype=float)
            sample_id = entry["medoid_id"]
            raw2d = None if raw2d_by_id is None else raw2d_by_id.get(sample_id)
            geometry, geometry_names = geometry_vector(coords, raw2d)
            self.samples.append({"id": sample_id, "label": entry["class_label"], "entry": entry,
                                 "coords": coords, "features": extract_all_features(coords),
                                 "bounds": entry["phase_boundaries"], "raw2d": raw2d,
                                 "geometry": geometry})
        self.geometry_names = geometry_names
        self.labels = list(dict.fromkeys(s["label"] for s in self.samples))

    def _temporal_distance(self, query, reference, *, derivative=False, smooth=False):
        q = derivative_features(query["features"], smooth=smooth) if derivative else query["features"]
        r = derivative_features(reference["features"], smooth=smooth) if derivative else reference["features"]
        weights = resolve_weights(self.config, self.config["default_profile"], reference["label"])
        return float(phase_aware_weighted_dtw(q, query["bounds"], r, reference["bounds"], weights, self.config)["total"])

    @staticmethod
    def _class_min(pair_distances, candidates, labels):
        return {label: min(pair_distances[c["id"]] for c in candidates if c["label"] == label) for label in labels}

    def run(self, output_dir: str | Path):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=False)
        method_names = ("DTW", "DDTW", "Geometry", "DTW+DDTW", "DTW+Geometry", "DDTW+Geometry", "DTW+DDTW+Geometry")
        rows = []; timings = Counter(); pair_cache = {"DTW": {}, "DDTW": {}}
        for method, derivative in (("DTW", False), ("DDTW", True)):
            started = time.perf_counter()
            for i, left in enumerate(self.samples):
                for right in self.samples[i + 1:]:
                    key = tuple(sorted((left["id"], right["id"])))
                    pair_cache[method][key] = self._temporal_distance(left, right, derivative=derivative)
            timings[method] = time.perf_counter() - started
        for test in self.samples:
            train = [s for s in self.samples if s["id"] != test["id"]]
            fold = {}
            for method in ("DTW", "DDTW"):
                pairs={r["id"]: pair_cache[method][tuple(sorted((test["id"],r["id"])))] for r in train}
                fold[method]=self._class_min(pairs,train,self.labels)
            started=time.perf_counter(); loc,scale=robust_location_scale(np.stack([s["geometry"] for s in train]))
            gpairs={r["id"]: float(np.linalg.norm((test["geometry"]-r["geometry"])/scale)) for r in train}
            timings["Geometry"] += time.perf_counter()-started; fold["Geometry"]=self._class_min(gpairs,train,self.labels)
            # Method-distance scales are estimated exclusively from train/train pairs.
            method_scales={}
            for method, derivative in (("DTW",False),("DDTW",True)):
                vals=[pair_cache[method][tuple(sorted((a["id"],b["id"])))]
                      for i,a in enumerate(train) for b in train[i+1:]]
                method_scales[method]=max(float(np.median(vals)),1e-8)
            gvals=[float(np.linalg.norm((a["geometry"]-b["geometry"])/scale)) for i,a in enumerate(train) for b in train[i+1:]]
            method_scales["Geometry"]=max(float(np.median(gvals)),1e-8)
            combos={"DTW+DDTW":("DTW","DDTW"), "DTW+Geometry":("DTW","Geometry"),
                    "DDTW+Geometry":("DDTW","Geometry"), "DTW+DDTW+Geometry":("DTW","DDTW","Geometry")}
            for name,parts in combos.items():
                fold[name]={label: float(np.mean([fold[p][label]/method_scales[p] for p in parts])) for label in self.labels}
            item={"sample_id":test["id"],"actual_class":test["label"],"methods":{},"geometry_values":dict(zip(self.geometry_names,test["geometry"]))}
            for method in method_names:
                ordered=sorted(fold[method],key=fold[method].get); pred=ordered[0]
                item["methods"][method]={"predicted_class":pred,"class_distances":fold[method],"best_class":pred,
                    "second_best_class":ordered[1],"margin":float(fold[method][ordered[1]]-fold[method][ordered[0]]),"correct":pred==test["label"]}
            rows.append(item)
        timings["Hybrid"] = sum(timings[x] for x in ("DTW","DDTW","Geometry"))
        results={}
        for method in method_names:
            matrix,metrics=confusion_and_metrics([r["actual_class"] for r in rows],[r["methods"][method]["predicted_class"] for r in rows],self.labels)
            results[method]={"metrics":metrics,"confusion_matrix":matrix.tolist()}
            self._write_matrix(out/f"confusion_matrix_{method.lower().replace('+','_')}.csv",matrix)
        disagreements=[]
        for row in rows:
            a=row["methods"]["DTW"]; b=row["methods"]["DDTW"]
            if a["predicted_class"] != b["predicted_class"]: disagreements.append(row)
        geometry_stats=[]
        for label in self.labels:
            selected=[r for r in rows if r["actual_class"]==label]
            for name in self.geometry_names: geometry_stats.append({"class":label,"feature":name,**_stats([r["geometry_values"][name] for r in selected])})
        summary={"benchmark_name":"Reference Leave-One-Out Benchmark","accuracy_note":"Classification Accuracy != Posture Match %",
                 "sample_count":len(rows),"labels":self.labels,"results":results,
                 "timings_seconds":{k:float(v) for k,v in timings.items()},
                 "timings_ms_per_sample":{k:float(v*1000/len(rows)) for k,v in timings.items()},
                 "dtw_ddtw":dict(Counter(("both_correct" if r["methods"]["DTW"]["correct"] and r["methods"]["DDTW"]["correct"] else
                    "dtw_only" if r["methods"]["DTW"]["correct"] else "ddtw_only" if r["methods"]["DDTW"]["correct"] else "both_wrong") for r in rows))}
        (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        (out/"sample_results.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
        self._write_flat_samples(out/"sample_results.csv",rows,method_names)
        self._write_flat_samples(out/"dtw_vs_ddtw_disagreements.csv",disagreements,("DTW","DDTW"))
        self._write_dicts(out/"geometry_class_stats.csv",geometry_stats)
        self._write_dicts(out/"feature_distributions.csv",geometry_stats)
        heel=[x for x in geometry_stats if x["feature"].startswith("heel_")]; self._write_dicts(out/"heel_feature_stats.csv",heel)
        distance_rows=[]
        for method in method_names:
            for label in self.labels:
                values=[r["methods"][method]["class_distances"][label] for r in rows]
                distance_rows.append({"method":method,"candidate_class":label,**_stats(values)})
            distance_rows.append({"method":method,"candidate_class":"__margin__",
                                  **_stats([r["methods"][method]["margin"] for r in rows])})
        self._write_dicts(out/"distance_distributions.csv",distance_rows)
        data_profile=self._data_profile()
        (out/"data_profile.json").write_text(json.dumps(data_profile,ensure_ascii=False,indent=2),encoding="utf-8")
        (out/"report.md").write_text(self._markdown(summary),encoding="utf-8")
        return summary, rows

    def _data_profile(self):
        by_class={}
        for label in self.labels:
            selected=[s for s in self.samples if s["label"]==label]
            source_count=sum(int(s["entry"].get("cluster_size",0)) for s in selected)
            by_class[label]={"source_sequence_count_from_cluster_size":source_count,
                "operational_reference_count":len(selected),
                "frame_count":_stats([len(s["coords"]) for s in selected]),
                "3d_shapes":[list(s["coords"].shape) for s in selected],
                "2d_shapes":[None if s["raw2d"] is None else list(s["raw2d"].shape) for s in selected],
                "finite_3d":all(np.isfinite(s["coords"]).all() for s in selected),
                "finite_2d":all(s["raw2d"] is None or np.isfinite(s["raw2d"]).all() for s in selected),
                "visibility_available":False}
        return {"classes":by_class,"sequences_npz":"normalized 3D (T,18,3)",
                "2d_storage":"not in reference DB; loaded from TL/VL source ZIP using manifest identity"}

    def _write_matrix(self,path,matrix):
        with path.open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.writer(f);w.writerow(["actual\\predicted",*self.labels]);[w.writerow([label,*matrix[i]]) for i,label in enumerate(self.labels)]

    @staticmethod
    def _write_dicts(path,rows):
        if not rows: return
        with path.open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    @staticmethod
    def _write_flat_samples(path,rows,methods):
        fields=["sample_id","actual_class"]+[f"{m}_{x}" for m in methods for x in ("predicted_class","margin","correct")]
        with path.open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for r in rows:
                row={"sample_id":r["sample_id"],"actual_class":r["actual_class"]}
                for m in methods:
                    for x in ("predicted_class","margin","correct"):row[f"{m}_{x}"]=r["methods"][m][x]
                w.writerow(row)

    def _markdown(self,summary):
        lines=["# Reference Leave-One-Out Benchmark","","Classification Accuracy != Posture Match %","",
               "| Method | Accuracy | Macro Precision | Macro Recall | Macro F1 |","|---|---:|---:|---:|---:|"]
        for name,data in summary["results"].items():
            m=data["metrics"];lines.append(f"| {name} | {m['accuracy']:.4f} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | {m['macro_f1']:.4f} |")
        return "\n".join(lines)+"\n"


__all__=["OfflinePoseBenchmark","stable_derivative","derivative_features","geometry_vector",
         "robust_location_scale","confusion_and_metrics"]
