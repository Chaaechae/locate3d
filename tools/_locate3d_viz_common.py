"""
Plotly rendering helpers shared between
``tools/visualize_locate3d.py`` (Locate3DSegDetector / 0f / 0h) and
``tools/visualize_locate3d_baseline.py`` (Meta open-weight Locate-3D).

Kept dependency-light (numpy + plotly only) so the baseline viz can be
imported without pulling in pointcept / torch_scatter / spconv.
"""

import numpy as np


# Two entity-indexed palettes. ``_VIVID_PALETTE`` is the high-visibility
# neon set used for GT (the box the user is trying to hit must be the
# dominant visual element); ``_MUTED_PALETTE`` is the softer D3 Cat10
# set used for Pred so they're easy to compare without stealing focus.
_VIVID_PALETTE = [
    "#ff00ff",  # magenta
    "#00ff00",  # lime
    "#ffff00",  # yellow
    "#ff4500",  # orangered
    "#00ffff",  # cyan
    "#ff1493",  # deep pink
    "#adff2f",  # green-yellow
    "#ff8c00",  # dark orange
    "#7fff00",  # chartreuse
    "#ff69b4",  # hot pink
]
_MUTED_PALETTE = [
    "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f",
    "#17becf", "#bcbd22", "#e377c2", "#aec7e8", "#98df8a",
]


def _box_edges(b):
    """16-point polyline tracing all 12 edges of an axis-aligned xyzxyz box."""
    x0, y0, z0, x1, y1, z1 = [float(v) for v in b]
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs.extend([corners[a][0], corners[b][0], None])
        ys.extend([corners[a][1], corners[b][1], None])
        zs.extend([corners[a][2], corners[b][2], None])
    return xs, ys, zs


def _box_corners(b):
    """Return the 8 corner xyz tuples of an xyzxyz box. Used to place
    marker dots at the corners of GT boxes for extra visibility."""
    x0, y0, z0, x1, y1, z1 = [float(v) for v in b]
    return (
        [x0, x1, x1, x0, x0, x1, x1, x0],
        [y0, y0, y1, y1, y0, y0, y1, y1],
        [z0, z0, z0, z0, z1, z1, z1, z1],
    )


def _render_scene(
    out_path, coord, color, gt_boxes, pred_boxes,
    pred_logits=None, infer_threshold=0.5,
    caption="", scene_id="", primary_idx=0,
    entity_names=None,
    entity_tokens=None,
    caption_token_colormap=None,
    caption_word_list=None,
    max_points=120000, draw_masks=True,
    draw_boxes=True,
    scene_point_size=2.2,
    scene_opacity=0.9,
    mask_coord=None,
):
    """Plotly render of a 3D scene with GT / Pred boxes and optional
    per-entity mask overlay.

    ``coord`` / ``color`` are the *background* point cloud. When
    ``mask_coord`` is provided, the per-entity ``pred_logits`` are
    interpreted as scoring those points (e.g. the cache's downsampled
    points the model actually ran on) and rendered on top of the
    background. When ``mask_coord`` is None, ``pred_logits`` is
    expected to align with ``coord`` itself (pre-existing behavior).
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        print(f"[error] plotly not installed: {e}")
        return

    color = np.asarray(color, dtype=np.float32)
    if color.size > 0 and color.max() > 1.5:
        color = color / 255.0
    color = np.clip(color, 0.0, 1.0)

    N = coord.shape[0]
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
    else:
        idx = np.arange(N)
    c_sub = coord[idx]
    col_sub = color[idx]

    # Mask overlay coordinates. If a separate mask_coord was provided
    # (the points the model actually ran inference on), keep the full
    # set -- they're already small (~30k). Otherwise reuse c_sub.
    if mask_coord is not None:
        mask_coord_arr = np.asarray(mask_coord, dtype=np.float32)
        logit_for_mask = pred_logits   # logits already align with mask_coord
    else:
        mask_coord_arr = c_sub
        if pred_logits is not None and pred_logits.shape[1] == N:
            logit_for_mask = pred_logits[:, idx]
        else:
            logit_for_mask = None

    rgb_str = [
        f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
        for r, g, b in col_sub
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=c_sub[:, 0], y=c_sub[:, 1], z=c_sub[:, 2],
            mode="markers",
            marker=dict(
                size=scene_point_size,
                color=rgb_str,
                opacity=scene_opacity,
            ),
            name="scene",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    G = max(len(gt_boxes), len(pred_boxes))
    for g in range(G):
        gt_color = _VIVID_PALETTE[g % len(_VIVID_PALETTE)]
        pred_color = _MUTED_PALETTE[g % len(_MUTED_PALETTE)]
        if entity_names is not None and g < len(entity_names):
            name = entity_names[g]
        else:
            name = f"entity_{g}"
        toks = ""
        if entity_tokens is not None and g < len(entity_tokens):
            toks = "/".join(entity_tokens[g])
            if toks:
                toks = f" [← '{toks}']"
        is_primary = g == primary_idx
        suffix = " (primary)" if is_primary else ""

        if draw_boxes and g < len(gt_boxes):
            xs, ys, zs = _box_edges(gt_boxes[g])
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(
                        color=gt_color,
                        width=10 if is_primary else 8,
                    ),
                    name=f"GT: {name}{toks}{suffix}",
                    legendgroup=f"e{g}",
                )
            )
            cx, cy, cz = _box_corners(gt_boxes[g])
            fig.add_trace(
                go.Scatter3d(
                    x=cx, y=cy, z=cz, mode="markers",
                    marker=dict(
                        size=6 if is_primary else 4,
                        color=gt_color,
                        symbol="diamond",
                        line=dict(color="black", width=1),
                    ),
                    name=f"GT corners {name}",
                    legendgroup=f"e{g}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        if draw_boxes and g < len(pred_boxes):
            xs, ys, zs = _box_edges(pred_boxes[g])
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(
                        color=pred_color,
                        width=5 if is_primary else 4,
                    ),
                    name=f"Pred: {name}{toks}{suffix}",
                    legendgroup=f"e{g}",
                )
            )

        if (draw_masks and logit_for_mask is not None
                and g < logit_for_mask.shape[0]):
            prob_g = 1.0 / (1.0 + np.exp(-logit_for_mask[g]))
            mask_g = prob_g > infer_threshold
            if mask_g.any():
                mask_pts = mask_coord_arr[mask_g]
                fig.add_trace(
                    go.Scatter3d(
                        x=mask_pts[:, 0], y=mask_pts[:, 1], z=mask_pts[:, 2],
                        mode="markers",
                        marker=dict(
                            size=5 if is_primary else 4,
                            color=pred_color,
                            opacity=1.0,
                            line=dict(color="black", width=0.3),
                        ),
                        name=f"Pred points: {name}{suffix}",
                        legendgroup=f"e{g}",
                        hoverinfo="skip",
                    )
                )

    colored_caption = caption
    if caption_token_colormap is not None:
        words = (caption_word_list if caption_word_list is not None
                 else caption.split(" "))
        rendered = []
        for wi, w in enumerate(words):
            if not w:
                continue
            cc = None
            if wi < len(caption_token_colormap):
                cc = caption_token_colormap[wi]
            if cc is not None:
                rendered.append(
                    f"<span style='color:{cc};font-weight:bold'>{w}</span>"
                )
            else:
                rendered.append(w)
        colored_caption = " ".join(rendered)

    title = f"<b>{scene_id}</b><br><sub>{colored_caption}</sub>"
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
        ),
        height=850,
        margin=dict(l=0, r=0, t=80, b=0),
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
    )
    fig.write_html(out_path)
    print(f"[wrote] {out_path}")
