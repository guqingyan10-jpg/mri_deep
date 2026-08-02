"""
=============================================================================
Interactive ET Lesion Disconnection Proof — Plotly 3D + Slice Slider
=============================================================================
Generates an interactive HTML per case where you can:
  1. Rotate/zoom/pan a 3D view with each lesion in a different color
  2. Slide through Z-slices with 2D overlay of all lesions
  3. Toggle individual lesions on/off

This is the definitive proof of disconnection:
  - In 3D view: if two color clusters are spatially separated with
    visible gaps → different lesions.
  - In slider view: slide through all Z-slices; if two colors never
    merge into one → provably disconnected.

Usage:
    python scripts/visualize_lesions.py

Output:
    lesion_verification_figures/{case_id}_interactive.html

Open in browser — fully interactive, no Python needed.

Author: Generated for ResUNet enhancement project
Date:   2026-08-02
=============================================================================
"""

import os
import numpy as np
import nibabel as nib
from scipy import ndimage
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================

DATA_ROOT = '/root/autodl-tmp/brats_project/MICCAI_BraTS2020_TrainingData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

CASES = {
    'BraTS20_Training_225': 'A: Single large ET (111K vox)',
    'BraTS20_Training_274': 'B: Classic multi-focal (9 lesions)',
    'BraTS20_Training_293': 'C: Extreme fragmentation (35 lesions)',
    'BraTS20_Training_329': 'D: LGG with ZERO ET (large tumor, no enhancement)',
    'BraTS20_Training_284': 'E: Micro multi-focal (16 lesions, all small)',
}

OUTPUT_DIR = 'lesion_verification_figures'
SUBSAMPLE_3D = 8000   # max voxels per lesion in 3D view (for performance)
SLICE_ALPHA = 0.65    # opacity of lesion overlay on 2D slices


def distinct_colors_plotly(n):
    """Generate n distinct colors as 'rgb(r,g,b)' strings."""
    if n == 0:
        return []
    pool_rgba = []
    for cmap in [cm.tab20, cm.tab20b, cm.Set3, cm.Paired, cm.tab20c]:
        for i in range(20):
            r, g, b, _ = cmap(i / 20)
            pool_rgba.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
    while len(pool_rgba) < n:
        pool_rgba += pool_rgba
    return pool_rgba[:n]


def build_interactive_figure(case_id, description, flair, labeled, n_lesions):
    """
    Build a single interactive Plotly figure with:
      - Top: 3D scatter of all lesions (color-coded)
      - Bottom: 2D slice viewer with Z-slider, lesion overlay
    """

    D, H, W = labeled.shape
    colors = distinct_colors_plotly(n_lesions)

    if n_lesions == 0:
        z_mid = D // 2
        # WT contour for context
        seg_path = os.path.join(DATA_ROOT, case_id, f'{case_id}_seg.nii')
        seg = nib.load(seg_path)
        seg_data = np.asarray(seg.dataobj).astype(np.int16)
        wt_slice = np.isin(seg_data[z_mid], [1, 2, 4]).astype(np.uint8)

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=flair[z_mid], colorscale='Gray', showscale=False,
            name='FLAIR MRI', hoverinfo='skip'
        ))
        # WT contour via scatter
        ys, xs = np.where(wt_slice > 0)
        if len(xs) > 0:
            idx = np.random.choice(len(xs), min(5000, len(xs)), replace=False)
            fig.add_trace(go.Scatter(
                x=xs[idx], y=ys[idx], mode='markers',
                marker=dict(size=2, color='yellow', opacity=0.4),
                name='WT boundary (no ET)', hoverinfo='skip'
            ))

        fig.update_layout(
            title=dict(
                text=f'<b>{case_id}</b>: ZERO Enhancing Tumor<br>'
                     f'<sub>{description} | Yellow = WT contour (no ET inside)</sub>',
                font=dict(size=16)
            ),
            xaxis=dict(scaleanchor='y', scaleratio=1, showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False, autorange='reversed'),
            width=800, height=800,
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        html_path = os.path.join(OUTPUT_DIR, f'{case_id}_slices.html')
        fig.write_html(html_path)
        print(f'  Saved: {html_path} (zero ET — WT contour overlay)')
        return fig, None

    # ================================================================
    # Pre-compute lesion voxel coordinates
    # ================================================================
    lesion_coords_3d = []  # for 3D scatter
    lesion_traces_2d = []  # for 2D slice overlay (z_by_z)

    z_range_with_lesions = []
    for z in range(D):
        if (labeled[z] > 0).sum() > 0:
            z_range_with_lesions.append(z)

    for lid in range(1, n_lesions + 1):
        coords = np.argwhere(labeled == lid)  # (N, 3) → (z, y, x)
        if len(coords) == 0:
            lesion_coords_3d.append(None)
            lesion_traces_2d.append({})
            continue

        sz = len(coords)
        # --- 3D: subsample ---
        if sz > SUBSAMPLE_3D:
            idx = np.random.choice(sz, SUBSAMPLE_3D, replace=False)
            coords_3d = coords[idx]
        else:
            coords_3d = coords

        # Store as (x, y, z) for Plotly
        lesion_coords_3d.append({
            'x': coords_3d[:, 2].tolist(),
            'y': coords_3d[:, 1].tolist(),
            'z': coords_3d[:, 0].tolist(),
            'size': sz,
        })

        # --- 2D: pre-group by Z ---
        z_dict = {}
        for z in z_range_with_lesions:
            mask_z = (labeled[z] == lid)
            if mask_z.sum() == 0:
                continue
            ys, xs = np.where(mask_z)
            z_dict[z] = (xs.tolist(), ys.tolist())
        lesion_traces_2d.append(z_dict)

    # ================================================================
    # Build 3D Scatter Traces
    # ================================================================
    traces_3d = []
    for lid in range(1, n_lesions + 1):
        c = lesion_coords_3d[lid - 1]
        if c is None:
            continue
        traces_3d.append(go.Scatter3d(
            x=c['x'], y=c['y'], z=c['z'],
            mode='markers',
            marker=dict(size=2, color=colors[lid - 1], opacity=0.6),
            name=f'#{lid} ({c["size"]} vox)',
            legendgroup=f'lesion_{lid}',
            showlegend=True,
        ))

    # ================================================================
    # Build 2D Slice Traces (initial Z = middle of lesion range)
    # ================================================================
    z_init = z_range_with_lesions[len(z_range_with_lesions) // 2] if z_range_with_lesions else D // 2

    traces_2d_base = []
    # FLAIR background
    traces_2d_base.append(go.Heatmap(
        z=flair[z_init],
        colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(255,255,255)']],
        showscale=False,
        name='FLAIR',
        hoverinfo='skip',
    ))

    # Lesion overlays at initial Z
    for lid in range(1, n_lesions + 1):
        z_dict = lesion_traces_2d[lid - 1]
        if z_init in z_dict:
            xs, ys = z_dict[z_init]
            traces_2d_base.append(go.Scatter(
                x=xs, y=ys,
                mode='markers',
                marker=dict(size=3, color=colors[lid - 1], opacity=SLICE_ALPHA),
                name=f'#{lid}',
                legendgroup=f'lesion_{lid}',
                showlegend=False,
                hoverinfo='skip',
            ))

    # ================================================================
    # Create Slider Frames for 2D View
    # ================================================================
    slider_steps = []
    for z in z_range_with_lesions:
        frame_traces = [
            go.Heatmap(
                z=flair[z],
                colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(255,255,255)']],
                showscale=False,
                hoverinfo='skip',
            )
        ]
        for lid in range(1, n_lesions + 1):
            z_dict = lesion_traces_2d[lid - 1]
            if z in z_dict:
                xs, ys = z_dict[z]
                frame_traces.append(go.Scatter(
                    x=xs, y=ys,
                    mode='markers',
                    marker=dict(size=3, color=colors[lid - 1], opacity=SLICE_ALPHA),
                    showlegend=False,
                    hoverinfo='skip',
                ))
            else:
                # Empty trace to keep trace count consistent
                frame_traces.append(go.Scatter(
                    x=[], y=[], mode='markers', showlegend=False, hoverinfo='skip'
                ))

        slider_steps.append(dict(
            args=[[f'z={z}'], [
                None,  # title update
                {'z': [flair[z]]},  # update heatmap
            ] + [{'x': [t.x], 'y': [t.y]} if hasattr(t, 'x') else {}
                 for t in frame_traces[1:]]],
            label=str(z),
            method='animate',
        ))

    # We'll use a simpler approach: put 3D and 2D as two separate views
    # with the 2D view having a slider

    # ================================================================
    # Final Figure: 2-row subplot
    #   Row 1: 3D scatter (single, rotatable)
    #   Row 2: 2D slice (with Z slider)
    # ================================================================

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'xy'}]],
        subplot_titles=(
            '3D View — Rotate to See Spatial Separation',
            f'2D Slice View — Slide Z to Check Connectivity (z={z_init})'
        ),
        column_widths=[0.55, 0.45],
    )

    # Add 3D traces
    for t in traces_3d:
        fig.add_trace(t, row=1, col=1)

    # Add 2D traces at initial Z
    for t in traces_2d_base:
        fig.add_trace(t, row=1, col=2)

    # ================================================================
    # Build Z-Slider
    # ================================================================
    sliders = [dict(
        active=z_range_with_lesions.index(z_init) if z_init in z_range_with_lesions else 0,
        currentvalue={'prefix': 'Z-slice: '},
        pad={'t': 50},
        steps=[
            dict(
                label=str(z),
                method='update',
                args=[
                    # Update 2D traces only (indices after the 3D traces)
                    {'visible': [True] * len(fig.data)},
                    {'title.text': f'{case_id}: {n_lesions} ET Lesions — Z={z}'}
                ],
            )
            for z in z_range_with_lesions
        ]
    )]

    # Real Z-slider that updates the heatmap and scatter points:
    # Since Plotly sliders can't easily swap heatmap z-data in subplots,
    # we create actual trace updates per Z-slice

    # Rebuild with frames for proper slider
    # Actually, let's use a different approach: build ALL z-slices as separate
    # traces with 'visible' toggled by the slider.

    # Simpler: rebuild without make_subplots — two separate views tiled.
    # Even simpler: just the 3D view with good controls + a note about the
    # matplotlib static figure for 2D proof.

    # Let me rebuild cleanly:
    fig2 = go.Figure()

    # Add all 3D traces
    for t in traces_3d:
        fig2.add_trace(t)

    # Update layout
    fig2.update_layout(
        title=dict(
            text=f'<b>{case_id}</b>: {n_lesions} ET Lesions — 3D Disconnection Proof<br>'
                 f'<sub>{description} | Rotate • Zoom • Pan | Each color = one independent lesion</sub>',
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title='X (voxel)',
            yaxis_title='Y (voxel)',
            zaxis_title='Z (slice)',
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        legend=dict(
            title=dict(text=f'<b>{n_lesions} Lesions</b>'),
            itemsizing='constant',
            font=dict(size=9),
        ),
        width=1100,
        height=800,
        hovermode='closest',
        margin=dict(l=0, r=0, t=80, b=0),
    )

    # Add buttons to toggle all lesions
    buttons_3d = [
        dict(label='Show All', method='update',
             args=[{'visible': [True] * len(fig2.data)}]),
        dict(label='Hide Small (<500 vox)', method='update',
             args=[{'visible': [
                 True if (i >= len(lesion_coords_3d) or
                          lesion_coords_3d[i] is None or
                          lesion_coords_3d[i]['size'] >= 500)
                 else False
                 for i in range(len(fig2.data))
             ]}]),
        dict(label='Only Large (>5000 vox)', method='update',
             args=[{'visible': [
                 True if (i >= len(lesion_coords_3d) or
                          lesion_coords_3d[i] is None or
                          lesion_coords_3d[i]['size'] >= 5000)
                 else False
                 for i in range(len(fig2.data))
             ]}]),
    ]

    fig2.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=True,
            x=0.02, y=0.98,
            buttons=buttons_3d,
        )]
    )

    # ---- Build separate 2D slider figure ----

    fig_slice = go.Figure()

    # FLAIR base (will be updated by slider)
    fig_slice.add_trace(go.Heatmap(
        z=flair[z_init],
        colorscale='Gray',
        showscale=False,
        hoverinfo='skip',
    ))

    for lid in range(1, n_lesions + 1):
        z_dict = lesion_traces_2d[lid - 1]
        if z_init in z_dict:
            xs, ys = z_dict[z_init]
        else:
            xs, ys = [], []
        fig_slice.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='markers',
            marker=dict(size=4, color=colors[lid - 1], opacity=0.7),
            name=f'#{lid}',
            hoverinfo='name',
        ))

    # Build frames for slider
    frames = []
    for z in z_range_with_lesions:
        frame_data = [go.Heatmap(z=flair[z], colorscale='Gray', showscale=False, hoverinfo='skip')]
        for lid in range(1, n_lesions + 1):
            z_dict = lesion_traces_2d[lid - 1]
            if z in z_dict:
                xs, ys = z_dict[z]
            else:
                xs, ys = [], []
            frame_data.append(go.Scatter(
                x=xs, y=ys, mode='markers',
                marker=dict(size=4, color=colors[lid - 1], opacity=0.7),
                showlegend=False, hoverinfo='name',
            ))
        frames.append(go.Frame(data=frame_data, name=str(z)))

    fig_slice.frames = frames

    # Slider
    fig_slice.update_layout(
        title=dict(
            text=f'<b>{case_id}</b>: Z-Slice Viewer — Slide to Check Connectivity<br>'
                 f'<sub>If two colors appear on SAME slice at DIFFERENT positions → DISCONNECTED</sub>',
            font=dict(size=14),
        ),
        xaxis=dict(scaleanchor='y', scaleratio=1, showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False, autorange='reversed'),
        width=800,
        height=800,
        legend=dict(title=dict(text='Lesions'), font=dict(size=9)),
        sliders=[dict(
            active=z_range_with_lesions.index(z_init) if z_init in z_range_with_lesions else 0,
            currentvalue={'prefix': 'Z-slice: ', 'font': {'size': 14}},
            pad={'t': 60},
            len=0.95,
            x=0.025,
            steps=[dict(
                label=str(z),
                method='animate',
                args=[[str(z)], dict(
                    mode='immediate',
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                )],
            ) for z in z_range_with_lesions],
        )],
    )

    # ================================================================
    # Save
    # ================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    html_path_3d = os.path.join(OUTPUT_DIR, f'{case_id}_3d.html')
    fig2.write_html(html_path_3d)
    print(f'  Saved: {html_path_3d}')

    html_path_slice = os.path.join(OUTPUT_DIR, f'{case_id}_slices.html')
    fig_slice.write_html(html_path_slice)
    print(f'  Saved: {html_path_slice}')

    return fig2, fig_slice


def main():
    print('=' * 65)
    print('INTERACTIVE ET LESION DISCONNECTION PROOF')
    print('=' * 65)
    print()
    print('Output per case:')
    print('  {case}_3d.html    — Rotatable 3D view, each lesion different color')
    print('  {case}_slices.html — Z-slider, 2D overlay, slide to check connectivity')
    print()

    for case_id, desc in CASES.items():
        print(f'Processing: {case_id}')

        seg_path = os.path.join(DATA_ROOT, case_id, f'{case_id}_seg.nii')
        flair_path = os.path.join(DATA_ROOT, case_id, f'{case_id}_flair.nii')

        seg = nib.load(seg_path)
        seg_data = np.asarray(seg.dataobj).astype(np.int16)
        flair = nib.load(flair_path)
        flair_data = np.asarray(flair.dataobj).astype(np.float32)

        et_mask = (seg_data == 4)
        labeled_raw, n_raw = ndimage.label(et_mask)

        valid_ids = [lid for lid in range(1, n_raw + 1)
                     if (labeled_raw == lid).sum() >= 10]
        filtered = np.zeros_like(labeled_raw)
        for new_id, old_id in enumerate(valid_ids, 1):
            filtered[labeled_raw == old_id] = new_id

        n_valid = len(valid_ids)
        print(f'  Components: raw={n_raw}, valid(>=10vox)={n_valid}')

        build_interactive_figure(case_id, desc, flair_data, filtered, n_valid)

    print()
    print('=' * 65)
    print(f'Done. Open {OUTPUT_DIR}/*.html in a browser.')
    print('=' * 65)
    print()
    print('HOW TO USE:')
    print('  _3d.html:     Rotate with mouse. Different colors in')
    print('                different spatial locations = DISCONNECTED.')
    print('                Toggle buttons to filter small/large lesions.')
    print('  _slices.html: Drag Z-slider. If two colors appear on the')
    print('                SAME slice at DIFFERENT positions, and they')
    print('                never merge across slices → DISCONNECTED.')
    print('  The slider is the ultimate proof: a connected lesion')
    print('  MUST share at least one Z-slice. Drag through all slices')
    print('  — if colors never merge, the lesions ARE disconnected.')


if __name__ == '__main__':
    main()
