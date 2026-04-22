"""
Hooks for the Locate-3D referring-expression task:

- ``Locate3DVizHook``: after each evaluation epoch, runs the model on a fixed
  set of validation annotations (scene + caption) and writes a plotly HTML
  file containing the RGB point cloud, the ground-truth boxes, the top-K
  predicted boxes and the caption, so the reviewer can visually confirm that
  queries do not collapse and that the selected query's box converges to the
  referred entity over epochs.
"""

import os
import copy
import numpy as np
import torch

from .default import HookBase
from .builder import HOOKS
import pointcept.utils.comm as comm


def _box_edges(box_xyzxyz):
    """Return 16-point polyline that traces all 12 edges of an axis-aligned 3D
    box. ``None`` separators are used to break the polyline between edges so a
    single plotly ``Scatter3d(mode='lines')`` trace can draw the full cuboid."""
    x0, y0, z0, x1, y1, z1 = [float(v) for v in box_xyzxyz]
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs.extend([corners[a][0], corners[b][0], None])
        ys.extend([corners[a][1], corners[b][1], None])
        zs.extend([corners[a][2], corners[b][2], None])
    return xs, ys, zs


def _make_figure(coord, color, gt_boxes, gt_primary_idx, pred_boxes,
                 pred_scores, selected_idx, caption, title):
    import plotly.graph_objects as go

    color = np.asarray(color, dtype=np.float32)
    if color.max() > 1.5:
        color = color / 255.0
    color = np.clip(color, 0, 1)
    rgb = [
        f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        for r, g, b in color
    ]

    # sub-sample for plotting speed
    max_pts = 60000
    if coord.shape[0] > max_pts:
        sel = np.random.choice(coord.shape[0], max_pts, replace=False)
        coord = coord[sel]
        rgb = [rgb[i] for i in sel]

    traces = [
        go.Scatter3d(
            x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
            mode="markers",
            marker=dict(size=1.2, color=rgb, opacity=0.85),
            name="scene",
            hoverinfo="skip",
        )
    ]

    for g, box in enumerate(gt_boxes):
        xs, ys, zs = _box_edges(box)
        is_primary = (g == gt_primary_idx)
        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(
                    color="lime" if is_primary else "green",
                    width=6 if is_primary else 3,
                    dash="solid" if is_primary else "dash",
                ),
                name=f"gt_{g}{' (primary)' if is_primary else ''}",
            )
        )

    for k, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
        xs, ys, zs = _box_edges(box)
        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(
                    color="red" if k == 0 and selected_idx is not None else "orange",
                    width=6 if k == 0 else 3,
                    dash="solid" if k == 0 else "dot",
                ),
                name=f"pred@top{k+1} (s={score:.2f})",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title}<br><sup>{caption}</sup>",
        scene=dict(
            aspectmode="data",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        showlegend=True,
    )
    return fig


@HOOKS.register_module()
class Locate3DVizHook(HookBase):
    """Visualize predictions of fixed val samples every ``viz_freq`` epochs.

    Parameters
    ----------
    num_scenes : int
        How many validation annotations to visualize.
    top_k : int
        Plot the top-K queries by (positive-map · sigmoid-logits) score.
    save_subdir : str
        Subdirectory under ``cfg.save_path`` to write HTMLs into.
    viz_freq : int
        Plot every N epochs (1 = every epoch).
    """

    def __init__(self, num_scenes=3, top_k=5, save_subdir="viz", viz_freq=1):
        self.num_scenes = num_scenes
        self.top_k = top_k
        self.save_subdir = save_subdir
        self.viz_freq = viz_freq
        self._fixed_indices = None

    def _pick_indices(self, dataset):
        """Pick the first ``num_scenes`` annotations, preferring distinct
        ``scene_id`` values so we do not visualize the same scene twice."""
        if self._fixed_indices is not None:
            return self._fixed_indices
        chosen = []
        seen_scenes = set()
        anns = getattr(dataset, "anns", None)
        if anns is None:
            self._fixed_indices = list(range(min(self.num_scenes, len(dataset))))
            return self._fixed_indices
        for i, ann in enumerate(anns):
            sid = ann.get("scene_id", i)
            if sid in seen_scenes:
                continue
            chosen.append(i)
            seen_scenes.add(sid)
            if len(chosen) >= self.num_scenes:
                break
        # fall back – fill with earliest indices if we could not find enough
        while len(chosen) < self.num_scenes and len(chosen) < len(anns):
            for i in range(len(anns)):
                if i not in chosen:
                    chosen.append(i)
                    break
        self._fixed_indices = chosen
        return chosen

    def after_epoch(self):
        if not self.trainer.cfg.evaluate:
            return
        epoch = self.trainer.epoch + 1
        if epoch % self.viz_freq != 0:
            return
        if not comm.is_main_process():
            return
        try:
            import plotly.graph_objects as go  # noqa: F401
        except ImportError:
            self.trainer.logger.warning(
                "Locate3DVizHook: plotly not installed, skipping visualization."
            )
            return

        save_dir = os.path.join(self.trainer.cfg.save_path, self.save_subdir)
        os.makedirs(save_dir, exist_ok=True)

        dataset = self.trainer.val_loader.dataset
        indices = self._pick_indices(dataset)
        self.trainer.logger.info(
            f"Locate3DVizHook: rendering {len(indices)} scenes at epoch {epoch}"
        )

        self.trainer.model.eval()
        from pointcept.datasets.locate3d_collate import locate3d_collate_fn

        for rank, ds_idx in enumerate(indices):
            try:
                sample = dataset[ds_idx]
            except Exception as e:
                self.trainer.logger.warning(
                    f"Locate3DVizHook: failed to load sample {ds_idx}: {e}"
                )
                continue
            # build a batch of one via our collate
            batch = locate3d_collate_fn([copy.deepcopy(sample)])
            gpu_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    gpu_batch[k] = v.cuda(non_blocking=True)
                else:
                    gpu_batch[k] = v

            with torch.no_grad():
                out = self.trainer.model(gpu_batch)

            # Predictions
            pred_logits = out["pred_logits"][0]   # (Q, T)
            pred_boxes = out["pred_boxes"][0]     # (Q, 6)

            positive_map = batch["positive_map"][0]  # (G, T)
            gt_boxes = batch["boxes_xyzxyz"][0]       # (G, 6)
            pid_raw = batch.get("primary_object_id", [0])[0]
            if isinstance(pid_raw, torch.Tensor):
                primary = int(pid_raw.flatten()[0].item())
            else:
                primary = int(pid_raw)
            caption = batch["caption"][0] if isinstance(batch["caption"], list) else str(batch["caption"])
            scene_id = batch["scene_id"][0] if isinstance(batch["scene_id"], list) else str(batch["scene_id"])

            prob = pred_logits.sigmoid().float().cpu()
            pos = positive_map[primary].float().cpu() if primary < positive_map.shape[0] else positive_map[0].float().cpu()
            scores = prob @ pos
            top_scores, top_idx = torch.topk(scores, min(self.top_k, scores.shape[0]))
            top_boxes = pred_boxes[top_idx].float().cpu().numpy()

            # Points fed to the encoder (after GridSample/NormalizeColor).
            # We use the "feat" tensor - it is (N, 9) = coord+color+normal.
            feat_cpu = batch["feat"].float().cpu().numpy()
            coord = feat_cpu[:, :3]
            color = feat_cpu[:, 3:6]

            fig = _make_figure(
                coord=coord,
                color=color,
                gt_boxes=gt_boxes.float().cpu().numpy(),
                gt_primary_idx=primary,
                pred_boxes=top_boxes,
                pred_scores=top_scores.tolist(),
                selected_idx=int(top_idx[0].item()),
                caption=caption,
                title=f"epoch{epoch:04d} | scene={scene_id} | ann_idx={ds_idx}",
            )
            out_path = os.path.join(
                save_dir, f"scene{rank:02d}_{scene_id}_epoch{epoch:04d}.html"
            )
            fig.write_html(out_path)
            self.trainer.logger.info(f"Locate3DVizHook: wrote {out_path}")


@HOOKS.register_module()
class Locate3DMetricsLogger(HookBase):
    """Lightweight scalar metrics writer that does not depend on wandb.

    Writes three files under ``cfg.save_path``:

    - ``metrics_train_iter.jsonl`` : one JSON line per training iteration
      containing ``{epoch, iter, global_iter, lr, **scalar_outputs}``.
    - ``metrics_train_epoch.jsonl``: one line per epoch with the average of
      each training scalar (pulled from Pointcept's ``storage``).
    - ``metrics_val.jsonl``        : one line per evaluation epoch with the
      primary metric (set via ``trainer.comm_info['current_metric_value']``)
      plus optional extras pushed by the evaluator (``val_extras``).

    Also writes parallel CSV files for easy pandas/spreadsheet import.
    """

    TRAIN_ITER = ("metrics_train_iter.jsonl", "metrics_train_iter.csv")
    TRAIN_EPOCH = ("metrics_train_epoch.jsonl", "metrics_train_epoch.csv")
    VAL = ("metrics_val.jsonl", "metrics_val.csv")

    def __init__(self, log_train_every=1):
        import json as _json
        import csv as _csv
        self._json = _json
        self._csv = _csv
        self.log_train_every = max(1, int(log_train_every))
        self._global_iter = 0
        self._train_keys = None          # CSV column order
        self._epoch_keys = None
        self._val_keys = None

    def _path(self, pair):
        return (
            os.path.join(self.trainer.cfg.save_path, pair[0]),
            os.path.join(self.trainer.cfg.save_path, pair[1]),
        )

    def _append_jsonl(self, path, record):
        with open(path, "a") as f:
            f.write(self._json.dumps(record, default=float) + "\n")

    def _append_csv(self, path, record, keys_attr):
        keys = getattr(self, keys_attr)
        write_header = keys is None or not os.path.exists(path)
        if keys is None:
            keys = list(record.keys())
            setattr(self, keys_attr, keys)
        with open(path, "a", newline="") as f:
            w = self._csv.writer(f)
            if write_header:
                w.writerow(keys)
            w.writerow([record.get(k, "") for k in keys])

    # ---------- hook entry points ----------
    def before_train(self):
        if not comm.is_main_process():
            return
        os.makedirs(self.trainer.cfg.save_path, exist_ok=True)
        self._global_iter = self.trainer.start_epoch * len(self.trainer.train_loader)

    def after_step(self):
        self._global_iter += 1
        if not comm.is_main_process():
            return
        if self._global_iter % self.log_train_every != 0:
            return
        out = self.trainer.comm_info.get("model_output_dict", None)
        if out is None:
            return
        rec = {
            "epoch": self.trainer.epoch + 1,
            "iter": int(self.trainer.comm_info.get("iter", -1)) + 1,
            "global_iter": self._global_iter,
            "lr": float(
                self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
            ),
        }
        for k, v in out.items():
            if isinstance(v, torch.Tensor) and v.ndim == 0:
                rec[k] = float(v.item())
        j_path, c_path = self._path(self.TRAIN_ITER)
        self._append_jsonl(j_path, rec)
        self._append_csv(c_path, rec, "_train_keys")

    def after_epoch(self):
        if not comm.is_main_process():
            return
        epoch = self.trainer.epoch + 1

        # training scalars averaged over the epoch
        train_rec = {"epoch": epoch}
        storage = getattr(self.trainer, "storage", None)
        if storage is not None:
            histories = storage.histories()
            for key, hist in histories.items():
                try:
                    train_rec[key] = float(hist.avg)
                except Exception:
                    continue
        if len(train_rec) > 1:
            j_path, c_path = self._path(self.TRAIN_EPOCH)
            self._append_jsonl(j_path, train_rec)
            self._append_csv(c_path, train_rec, "_epoch_keys")

        # val metric (set by evaluator)
        if self.trainer.cfg.evaluate:
            current_metric_value = self.trainer.comm_info.get(
                "current_metric_value", None
            )
            current_metric_name = self.trainer.comm_info.get(
                "current_metric_name", "metric"
            )
            if current_metric_value is not None:
                val_rec = {
                    "epoch": epoch,
                    current_metric_name: float(current_metric_value),
                }
                for k, v in self.trainer.comm_info.get("val_extras", {}).items():
                    try:
                        val_rec[k] = float(v)
                    except Exception:
                        pass
                j_path, c_path = self._path(self.VAL)
                self._append_jsonl(j_path, val_rec)
                self._append_csv(c_path, val_rec, "_val_keys")


@HOOKS.register_module()
class Locate3DDebugPrinter(HookBase):
    """Lightweight logger: prints the debug scalars reported by the model every
    ``print_every`` train iterations so ``dbg_match_iou`` / ``dbg_query_entropy``
    can be eyeballed without wandb/tensorboard."""

    def __init__(self, print_every=50):
        self.print_every = print_every
        self._iter = 0

    def after_step(self):
        self._iter += 1
        if self._iter % self.print_every != 0:
            return
        out = self.trainer.comm_info.get("model_output_dict", None)
        if out is None:
            return
        msg = []
        for k in [
            "dbg_match_iou",
            "dbg_match_iou_primary",
            "dbg_gt_covered25",
            "dbg_center_std",
            "dbg_size_std",
            "dbg_query_entropy",
            "dbg_query_unique_ratio",
        ]:
            if k in out and isinstance(out[k], torch.Tensor):
                msg.append(f"{k}={out[k].item():.3f}")
        if msg:
            self.trainer.logger.info("[Locate3D debug] " + " ".join(msg))
