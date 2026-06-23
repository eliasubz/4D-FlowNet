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
):
    """Create a Plotly 3D velocity figure with a translucent vessel wall.

    Args:
        velocity: Tensor shaped [3, D, H, W].
        ref_velocity: Optional reference velocity tensor to mask the background.
    """
    import plotly.graph_objects as go

    velocity = velocity.detach().cpu()
    if ref_velocity is not None:
        ref_val = ref_velocity.detach().cpu()
        wall_spd = speed(ref_val)
        # Create a binary fluid domain mask from the reference speed
        mask = (wall_spd > 1e-4)
        velocity = velocity * mask
    else:
        wall_spd = speed(velocity)
        
    spd = speed(velocity)
    d, h, w = spd.shape

    axis_z = torch.linspace(-1.0, 1.0, d)
    axis_y = torch.linspace(-1.0, 1.0, h)
    axis_x = torch.linspace(-1.0, 1.0, w)
    zz, yy, xx = torch.meshgrid(axis_z, axis_y, axis_x, indexing="ij")

    sampled = torch.zeros_like(spd, dtype=torch.bool)
    sampled[::vector_step, ::vector_step, ::vector_step] = True
    vector_mask = (spd > vector_threshold) & sampled

    x = xx[vector_mask].numpy()
    y = yy[vector_mask].numpy()
    z = zz[vector_mask].numpy()
    u = velocity[0][vector_mask].numpy()
    v = velocity[1][vector_mask].numpy()
    wv = velocity[2][vector_mask].numpy()
    c = spd[vector_mask].numpy()

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
            cmin=0.0,
            cmax=float(spd.max()),
            colorbar=dict(title="speed"),
            customdata=c,
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>speed=%{customdata:.3f}<extra></extra>",
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
        title="Synthetic 3D velocity field with translucent wall",
    )
    return fig
