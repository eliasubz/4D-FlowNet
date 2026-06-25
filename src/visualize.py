from __future__ import annotations

import torch


def speed(v: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(v, dim=0)


def make_interactive_3d_flow_figure(
    velocity: torch.Tensor,
    wall_threshold: float = 0.03,
    vector_threshold: float = 0.12,
    vector_step: int = 4,
    arrow_scale: float = 0.18,
    ref_velocity: torch.Tensor | None = None,
    vessel_mask: torch.Tensor | None = None,
):
    """Create a Plotly 3D velocity figure with a translucent vessel wall.

    Args:
        velocity: Tensor shaped [3, D, H, W].
        ref_velocity: Optional reference velocity tensor to mask the background.
        vessel_mask: Optional binary vessel mask shaped [D, H, W]. When given,
            the wall is drawn from this mask instead of from velocity magnitude.
    """
    import plotly.graph_objects as go

    velocity = velocity.detach().cpu()[:3]
    # Permute velocity from (3, Z, Y, X) to Cartesian (3, X, Y, Z)
    velocity = velocity.permute(0, 3, 2, 1)

    if vessel_mask is not None:
        wall_mask = vessel_mask.detach().cpu().float()
        # Permute mask from (Z, Y, X) to Cartesian (X, Y, Z)
        wall_mask = wall_mask.permute(2, 1, 0)
        mask = wall_mask > 0.5
        wall_spd = wall_mask
        velocity = velocity * mask
        wall_threshold = 0.5
    elif ref_velocity is not None:
        ref_val = ref_velocity.detach().cpu()[:3]
        # Permute ref_val from (3, Z, Y, X) to Cartesian (3, X, Y, Z)
        ref_val = ref_val.permute(0, 3, 2, 1)
        wall_spd = speed(ref_val)
        # Create a binary fluid domain mask from the reference speed
        mask = (wall_spd > 1e-4)
        velocity = velocity * mask
    else:
        wall_spd = speed(velocity)
        
    spd = speed(velocity)
    w, h, d = spd.shape

    axis_x = torch.linspace(-1.0, 1.0, w)
    axis_y = torch.linspace(-1.0, 1.0, h)
    axis_z = torch.linspace(-1.0, 1.0, d)
    # Generate Cartesian meshgrid with indexing="ij"
    xx, yy, zz = torch.meshgrid(axis_x, axis_y, axis_z, indexing="ij")

    sampled = torch.zeros_like(spd, dtype=torch.bool)
    sampled[::vector_step, ::vector_step, ::vector_step] = True
    vector_mask = (spd > vector_threshold) & sampled

    x = xx[vector_mask].numpy()
    y = yy[vector_mask].numpy()
    z = zz[vector_mask].numpy()
    u = velocity[0][vector_mask].numpy()
    v = velocity[1][vector_mask].numpy()
    wv = velocity[2][vector_mask].numpy()
    c = (spd[vector_mask] * 100.0).numpy()
    color_min = float(spd[vector_mask].min()) if c.size else 0.0
    color_max = float(spd[vector_mask].max()) if c.size else 1.0
    tick_count = 5
    if color_max > color_min:
        tick_vals = torch.linspace(color_min, color_max, tick_count).tolist()
    else:
        tick_vals = [color_min]
    tick_text = [f"{val * 100.0:.0f}" for val in tick_vals]

    fig = go.Figure()
    fig.add_trace(
        go.Isosurface(
            x=xx.flatten().numpy(),
            y=yy.flatten().numpy(),
            z=zz.flatten().numpy(),
            value=wall_spd.flatten().numpy(),
            isomin=wall_threshold,
            isomax=float(wall_spd.max()),
            surface_count=1,
            opacity=0.16,
            colorscale="Greys",
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            name="translucent wall",
        )
    )
    fig.add_trace(
        go.Cone(
            x=x,
            y=y,
            z=z,
            u=u,
            v=v,
            w=wv,
            sizemode="absolute",
            sizeref=arrow_scale,
            anchor="tail",
            colorscale="Viridis",
            cauto=False,
            cmin=color_min,
            cmax=color_max,
            colorbar=dict(
                title="speed (cm/s)",
                tickmode="array",
                tickvals=tick_vals,
                ticktext=tick_text,
            ),
            customdata=c,
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>speed=%{customdata:.1f} cm/s<extra></extra>",
            name="velocity vectors",
        )
    )
    fig.update_layout(
        width=900,
        height=720,
        scene=dict(
            xaxis=dict(visible=False, range=[-1, 1]),
            yaxis=dict(visible=False, range=[-1, 1]),
            zaxis=dict(visible=False, range=[-1, 1]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=35, b=0),
        title=dict(
            text="Synthetic 3D Velocity Field",
            x=0.5,
            xanchor="center",
            yanchor="top",
        ),
    )
    return fig
