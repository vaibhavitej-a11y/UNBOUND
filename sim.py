"""
STELLAR DISRUPTION
A rogue star performs a close gravitational flyby through a planetary system.
Real Newtonian N-body physics. Velocity Verlet integration. Astronomical units.

Physics Setup:
- Units: Astronomical Units (AU), Solar Masses (M_sun), Years (yr).
- Gravitational Constant G = 4*pi^2 ~ 39.478 AU^3 / (M_sun * yr^2).
- Bodies:
  - Body 0: Fixed Host Star (1.0 M_sun) at origin (0, 0) AU.
  - Body 1: Inner Planet (~1 Earth Mass) at r1 = 1.0 AU.
  - Body 2: Outer Planet (~3.3 Earth Masses) at r2 = 1.8 AU.
  - Body 3: Rogue Star (0.6 M_sun) — hyperbolic flyby trajectory.
- Integrator: 2nd-order Symplectic Velocity Verlet scheme.
- Softening: eps^2 = 1e-6 AU^2 (prevents singularity at close approach).

Controls:
- SPACE : Pause / Resume simulation
- R     : Reset simulation state
- S     : Toggle Rogue Star (ACTIVE / DISABLED)
- UP    : Increase simulation speed
- DOWN  : Decrease simulation speed
"""

import math
import taichi as ti

# -----------------------------------------------------------------------------
# 1. Taichi & Physics Constants
# -----------------------------------------------------------------------------
try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)

NUM_BODIES = 4                     # 0: Star, 1: Planet 1, 2: Planet 2, 3: Rogue Star
NUM_TRAILS = 3                     # Planets 1 & 2 + Rogue Star
G_CONST = 4.0 * math.pi * math.pi  # AU^3 / (M_sun * yr^2)
EPS_SQ = 1e-6                      # Gravitational softening factor (AU^2)
BASE_DT = 0.001                    # Fundamental simulation time step (years per step)
SUBSTEPS_PER_FRAME = 20            # Sub-stepping for integration stability
MAX_TRAIL_LEN = 180                # Trail history cap — prevents tangled spaghetti

# Window dimensions
WIN_W, WIN_H = 1280, 800

# Starfield config — rendered once into a background image
NUM_STARS = 320
STAR_IMG_W, STAR_IMG_H = 1280, 800

# -----------------------------------------------------------------------------
# 2. Taichi Fields (GPU Data Allocation)
# -----------------------------------------------------------------------------
pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_BODIES)
vel = ti.Vector.field(2, dtype=ti.f32, shape=NUM_BODIES)
acc = ti.Vector.field(2, dtype=ti.f32, shape=NUM_BODIES)
mass = ti.field(dtype=ti.f32, shape=NUM_BODIES)

# Initial state backup for reset functionality
init_pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_BODIES)
init_vel = ti.Vector.field(2, dtype=ti.f32, shape=NUM_BODIES)
init_mass = ti.field(dtype=ti.f32, shape=NUM_BODIES)

# Screen coordinate fields [0, 1]^2 for rendering
render_pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_BODIES)
render_radius = ti.field(dtype=ti.f32, shape=NUM_BODIES)
body_colors = ti.Vector.field(3, dtype=ti.f32, shape=NUM_BODIES)

# Orbit Trail Fields (Ring buffer storing physical AU positions)
trail_history = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_TRAILS, MAX_TRAIL_LEN))
trail_head = ti.field(dtype=ti.i32, shape=())   # Current write index in ring buffer
trail_count = ti.field(dtype=ti.i32, shape=())  # Total points stored (up to MAX_TRAIL_LEN)

# Renderable line segment vertices and colors for canvas.lines()
TRAIL_VERTICES_PER_BODY = 2 * (MAX_TRAIL_LEN - 1)
TOTAL_TRAIL_VERTICES = NUM_TRAILS * TRAIL_VERTICES_PER_BODY

trail_line_vertices = ti.Vector.field(2, dtype=ti.f32, shape=TOTAL_TRAIL_VERTICES)
trail_line_colors = ti.Vector.field(3, dtype=ti.f32, shape=TOTAL_TRAIL_VERTICES)

# Static backdrop tensor (near-black #05070c + low-density stars)
star_bg_field = ti.Vector.field(3, dtype=ti.f32, shape=(STAR_IMG_W, STAR_IMG_H))

# Final output RGBA image tensor uploaded to canvas
star_img = ti.Vector.field(3, dtype=ti.f32, shape=(STAR_IMG_W, STAR_IMG_H))

# -----------------------------------------------------------------------------
# 3. Physics & Simulation Kernels  (UNCHANGED)
# -----------------------------------------------------------------------------

@ti.kernel
def compute_accelerations(rogue_enabled: ti.i32):
    """Computes N-body gravitational accelerations using real Newtonian gravity."""
    for i in range(NUM_BODIES):
        if i == 0:
            # Body 0 is the fixed host star at origin
            acc[i] = ti.Vector([0.0, 0.0])
        elif i == 3 and rogue_enabled == 0:
            # Body 3 (Rogue Star) acceleration is zero when disabled
            acc[i] = ti.Vector([0.0, 0.0])
        else:
            acc_i = ti.Vector([0.0, 0.0])
            for j in range(NUM_BODIES):
                if i != j:
                    if j == 3 and rogue_enabled == 0:
                        continue
                    r_vec = pos[j] - pos[i]
                    r_sq = r_vec.norm_sqr() + EPS_SQ
                    r_dist = ti.sqrt(r_sq)
                    acc_i += G_CONST * mass[j] * r_vec / (r_dist * r_sq)
            acc[i] = acc_i


@ti.kernel
def verlet_half_step_1(dt: ti.f32, rogue_enabled: ti.i32):
    """First half of Symplectic Velocity Verlet integration: update positions and half-step velocities."""
    for i in range(NUM_BODIES):
        if i == 0:
            continue
        if i == 3 and rogue_enabled == 0:
            continue
        pos[i] += vel[i] * dt + 0.5 * acc[i] * (dt * dt)
        vel[i] += 0.5 * acc[i] * dt


@ti.kernel
def verlet_half_step_2(dt: ti.f32, rogue_enabled: ti.i32):
    """Second half of Symplectic Velocity Verlet integration: update velocities with new accelerations."""
    for i in range(NUM_BODIES):
        if i == 0:
            continue
        if i == 3 and rogue_enabled == 0:
            continue
        vel[i] += 0.5 * acc[i] * dt


def verlet_step(dt: ti.f32, rogue_enabled: ti.i32):
    """Executes a full 2nd-order Velocity Verlet timestep."""
    verlet_half_step_1(dt, rogue_enabled)
    compute_accelerations(rogue_enabled)
    verlet_half_step_2(dt, rogue_enabled)


@ti.kernel
def reset_to_initial():
    """Resets body positions, velocities, masses, and trail histories to initial values."""
    for i in range(NUM_BODIES):
        pos[i] = init_pos[i]
        vel[i] = init_vel[i]
        acc[i] = ti.Vector([0.0, 0.0])
        mass[i] = init_mass[i]

    trail_head[None] = 0
    trail_count[None] = 0

    for t, i in ti.ndrange(NUM_TRAILS, MAX_TRAIL_LEN):
        trail_history[t, i] = ti.Vector([-999.0, -999.0])


# -----------------------------------------------------------------------------
# 4. Visual Shaders & Trail Projection
# -----------------------------------------------------------------------------

@ti.kernel
def update_render_positions(scale_x: ti.f32, scale_y: ti.f32, rogue_enabled: ti.i32):
    """Maps physical AU coordinates to screen space [0, 1]^2."""
    for i in range(NUM_BODIES):
        if i == 3 and rogue_enabled == 0:
            render_pos[i] = ti.Vector([-10.0, -10.0])
        else:
            rx = 0.5 + pos[i][0] / (2.0 * scale_x)
            ry = 0.5 + pos[i][1] / (2.0 * scale_y)
            render_pos[i] = ti.Vector([rx, ry])


@ti.func
def smoothstep(edge0: ti.f32, edge1: ti.f32, x: ti.f32) -> ti.f32:
    t = ti.max(0.0, ti.min(1.0, (x - edge0) / (edge1 - edge0 + 1e-6)))
    return t * t * (3.0 - 2.0 * t)


@ti.kernel
def render_scene_pixel_shader(scale_x: ti.f32, scale_y: ti.f32, rogue_enabled: ti.i32):
    """Per-pixel GPU rendering shader with layered blooms, ambient illumination, and 3D planet shading."""
    aspect = float(STAR_IMG_W) / float(STAR_IMG_H)

    # Host Star Center (body 0 at origin 0,0)
    cx0 = 0.5 + pos[0][0] / (2.0 * scale_x)
    cy0 = 0.5 + pos[0][1] / (2.0 * scale_y)

    # Compact Stellar Intruder Center (body 3)
    cx3 = 0.5 + pos[3][0] / (2.0 * scale_x)
    cy3 = 0.5 + pos[3][1] / (2.0 * scale_y)

    # Planet Centers (body 1 & body 2)
    cx1 = 0.5 + pos[1][0] / (2.0 * scale_x)
    cy1 = 0.5 + pos[1][1] / (2.0 * scale_y)

    cx2 = 0.5 + pos[2][0] / (2.0 * scale_x)
    cy2 = 0.5 + pos[2][1] / (2.0 * scale_y)

    for px, py in star_img:
        # Base backdrop color (#05070c + starfield)
        col = star_bg_field[px, py]

        u = float(px) / float(STAR_IMG_W)
        v = float(py) / float(STAR_IMG_H)

        # ---------------------------------------------------------------------
        # 1. Host Star: Layered Radial Bloom & Ambient Space Illumination (Visually Dominant)
        # ---------------------------------------------------------------------
        dx0 = (u - cx0) * aspect
        dy0 = (v - cy0)
        r0_sq = dx0 * dx0 + dy0 * dy0
        r0 = ti.sqrt(r0_sq)

        # Layer 1: Core (radius 0.015, incandescent white-gold core)
        R_core0 = 0.015
        t_core0 = smoothstep(R_core0, R_core0 * 0.55, r0)
        core_col0 = ti.Vector([1.00, 0.98, 0.91])

        # Layer 2: Intense Inner Halo
        sigma0_inner = 0.032
        I0_inner = 0.85 * ti.exp(-r0_sq / (sigma0_inner * sigma0_inner))
        glow0_inner = ti.Vector([0.98, 0.82, 0.36]) * I0_inner

        # Layer 3: Warm Amber Mid-Bloom
        sigma0_mid = 0.075
        I0_mid = 0.38 * ti.exp(-r0_sq / (sigma0_mid * sigma0_mid))
        glow0_mid = ti.Vector([0.92, 0.65, 0.22]) * I0_mid

        # Layer 4: Soft Ambient Illumination cast subtly on nearby space
        sigma0_amb = 0.165
        I0_amb = 0.12 * ti.exp(-r0_sq / (sigma0_amb * sigma0_amb))
        glow0_amb = ti.Vector([0.80, 0.50, 0.16]) * I0_amb

        col += (glow0_inner + glow0_mid + glow0_amb)
        col = col * (1.0 - t_core0) + core_col0 * t_core0

        # ---------------------------------------------------------------------
        # 2. Compact Stellar Intruder: Compact Blue-White Glow (Subordinate to Host Star)
        # ---------------------------------------------------------------------
        if rogue_enabled == 1:
            dx3 = (u - cx3) * aspect
            dy3 = (v - cy3)
            r3_sq = dx3 * dx3 + dy3 * dy3
            r3 = ti.sqrt(r3_sq)

            # Compact Core (radius 0.0115 — distinct and clearly visible, but subordinate to Host Star 0.015)
            R_core3 = 0.0115
            t_core3 = smoothstep(R_core3, R_core3 * 0.45, r3)
            core3_col = ti.Vector([0.92, 0.95, 1.00])

            # Layer 1: Compact Inner Halo
            sigma3_inner = 0.022
            I3_inner = 0.48 * ti.exp(-r3_sq / (sigma3_inner * sigma3_inner))
            glow3_inner = ti.Vector([0.76, 0.84, 1.00]) * I3_inner

            # Layer 2: Muted Blue-White Mid-Bloom
            sigma3_mid = 0.045
            I3_mid = 0.18 * ti.exp(-r3_sq / (sigma3_mid * sigma3_mid))
            glow3_mid = ti.Vector([0.55, 0.68, 0.95]) * I3_mid

            # Layer 3: Faint Ambient Space Cast
            sigma3_amb = 0.080
            I3_amb = 0.06 * ti.exp(-r3_sq / (sigma3_amb * sigma3_amb))
            glow3_amb = ti.Vector([0.40, 0.52, 0.85]) * I3_amb

            col += (glow3_inner + glow3_mid + glow3_amb)
            col = col * (1.0 - t_core3) + core3_col * t_core3

        # ---------------------------------------------------------------------
        # 3. Planet 3D Sphere Shading & Soft Ambient Glow
        # ---------------------------------------------------------------------
        for p in range(2):
            cx_p = cx1 if p == 0 else cx2
            cy_p = cy1 if p == 0 else cy2

            dx_p = (u - cx_p) * aspect
            dy_p = (v - cy_p)
            dist_p_sq = dx_p * dx_p + dy_p * dy_p

            # Visual radii for planets
            R_p = 0.0145 if p == 0 else 0.0160
            r_p_sq = dist_p_sq / (R_p * R_p)

            # Palette Base Colors: Cyan (Inner) & Coral (Outer)
            p_base = ti.Vector([0.37, 0.78, 0.85]) if p == 0 else ti.Vector([0.88, 0.54, 0.435])

            # Soft ambient illumination cast subtly on nearby space around planets
            p_amb = 0.22 * ti.exp(-dist_p_sq / ((R_p * 1.9) * (R_p * 1.9)))
            col += p_base * p_amb

            if r_p_sq <= 1.0:
                # 3D Normal vector N = (nx, ny, nz)
                nx = dx_p / R_p
                ny = dy_p / R_p
                nz = ti.sqrt(ti.max(0.0, 1.0 - r_p_sq))
                N = ti.Vector([nx, ny, nz])

                # Direction vector FROM planet center TO Host Star in 3D
                dir_x = (cx0 - cx_p) * aspect
                dir_y = (cy0 - cy_p)
                dir_len = ti.sqrt(dir_x * dir_x + dir_y * dir_y) + 1e-6
                L2D_x = dir_x / dir_len
                L2D_y = dir_y / dir_len
                L = ti.Vector([L2D_x, L2D_y, 0.25]).normalized()

                # Lambertian Diffuse N dot L
                NdotL = ti.max(0.0, N.dot(L))
                diffuse = 0.03 + 0.97 * ti.pow(NdotL, 0.85)

                # Specular Peak H = normalize(L + (0, 0, 1))
                V = ti.Vector([0.0, 0.0, 1.0])
                H = (L + V).normalized()
                NdotH = ti.max(0.0, N.dot(H))
                specular = ti.pow(NdotH, 20.0)

                # Limb Darkening
                limb = 1.0 - 0.30 * (1.0 - nz) * (1.0 - nz)

                p_spec_col = ti.Vector([1.0, 1.0, 1.0])
                shaded_col = p_base * diffuse * limb + p_spec_col * specular * 0.45

                # Edge Anti-Aliasing
                edge_alpha = smoothstep(1.0, 0.82, ti.sqrt(r_p_sq))
                col = col * (1.0 - edge_alpha) + shaded_col * edge_alpha

        # Write final pixel color
        star_img[px, py] = col


@ti.kernel
def record_trail_history(rogue_enabled: ti.i32):
    """Records current body positions in AU into ring buffers."""
    head = trail_head[None]
    for t in range(NUM_TRAILS):
        body_idx = t + 1  # 0: Planet 1 (idx 1), 1: Planet 2 (idx 2), 2: Rogue Star (idx 3)
        if body_idx == 3 and rogue_enabled == 0:
            trail_history[t, head] = ti.Vector([-999.0, -999.0])
        else:
            trail_history[t, head] = pos[body_idx]

    trail_head[None] = (head + 1) % MAX_TRAIL_LEN
    if trail_count[None] < MAX_TRAIL_LEN:
        trail_count[None] += 1


@ti.kernel
def build_trail_lines(scale_x: ti.f32, scale_y: ti.f32, rogue_enabled: ti.i32):
    """Builds vertex line segments and colors for canvas.lines() dynamically projected to screen coordinates."""
    count = trail_count[None]
    head = trail_head[None]

    for t in range(NUM_TRAILS):
        body_idx = t + 1
        color = body_colors[body_idx]
        base_vert_idx = t * TRAIL_VERTICES_PER_BODY

        for seg in range(MAX_TRAIL_LEN - 1):
            v_idx = base_vert_idx + seg * 2

            if seg < count - 1 and not (body_idx == 3 and rogue_enabled == 0):
                # Ring buffer indices relative to current head
                idx0 = (head - count + seg) % MAX_TRAIL_LEN
                if idx0 < 0:
                    idx0 += MAX_TRAIL_LEN
                idx1 = (idx0 + 1) % MAX_TRAIL_LEN

                p_au0 = trail_history[t, idx0]
                p_au1 = trail_history[t, idx1]

                # Don't draw segment if either point is inactive/off-screen
                if p_au0[0] < -900.0 or p_au1[0] < -900.0:
                    trail_line_vertices[v_idx] = ti.Vector([-1.0, -1.0])
                    trail_line_vertices[v_idx + 1] = ti.Vector([-1.0, -1.0])
                    trail_line_colors[v_idx] = ti.Vector([0.0, 0.0, 0.0])
                    trail_line_colors[v_idx + 1] = ti.Vector([0.0, 0.0, 0.0])
                else:
                    nx0 = 0.5 + p_au0[0] / (2.0 * scale_x)
                    ny0 = 0.5 + p_au0[1] / (2.0 * scale_y)
                    nx1 = 0.5 + p_au1[0] / (2.0 * scale_x)
                    ny1 = 0.5 + p_au1[1] / (2.0 * scale_y)

                    trail_line_vertices[v_idx] = ti.Vector([nx0, ny0])
                    trail_line_vertices[v_idx + 1] = ti.Vector([nx1, ny1])

                    # Smooth cubic ease-in with radiant head glow
                    alpha = float(seg) / float(MAX_TRAIL_LEN - 1)
                    fade = alpha * alpha * (3.0 - 2.0 * alpha)
                    glow = 1.0 + 0.40 * ti.pow(alpha, 4.0)
                    seg_color = color * (fade * glow)
                    trail_line_colors[v_idx] = seg_color
                    trail_line_colors[v_idx + 1] = seg_color
            else:
                # Degenerate segments off-screen
                trail_line_vertices[v_idx] = ti.Vector([-1.0, -1.0])
                trail_line_vertices[v_idx + 1] = ti.Vector([-1.0, -1.0])
                trail_line_colors[v_idx] = ti.Vector([0.0, 0.0, 0.0])
                trail_line_colors[v_idx + 1] = ti.Vector([0.0, 0.0, 0.0])


def generate_starfield():
    """Generates space-agency style near-black background #05070c with soft blurred haze blooms and starfield."""
    import numpy as np

    # 1. Base Deep Space Canvas (WIN_W x WIN_H x 3)
    img = np.zeros((STAR_IMG_W, STAR_IMG_H, 3), dtype=np.float32)

    # Base background #05070c: RGB [0.016, 0.022, 0.040]
    img[:, :, 0] = 0.016
    img[:, :, 1] = 0.022
    img[:, :, 2] = 0.040

    # 2. Subtle atmospheric hints: heavily reduced opacity to prevent blue cast across scene
    uu, vv = np.meshgrid(np.linspace(0.0, 1.0, STAR_IMG_W), np.linspace(0.0, 1.0, STAR_IMG_H), indexing='ij')
    xx = (uu - 0.5) * 1.6
    yy = vv - 0.5

    haze_blooms = [
        (-0.30,  0.12, 0.45, [0.004, 0.002, 0.006]),  # Faint deep violet hint
        ( 0.38, -0.15, 0.40, [0.002, 0.005, 0.006]),  # Faint deep teal hint
        ( 0.05, -0.25, 0.48, [0.002, 0.003, 0.007]),  # Faint deep navy hint
        (-0.18, -0.30, 0.38, [0.003, 0.002, 0.005]),  # Faint slate-purple hint
        ( 0.35,  0.28, 0.40, [0.003, 0.002, 0.004]),  # Faint muted plum hint
    ]

    for cx, cy, sigma, col in haze_blooms:
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        g = np.exp(-dist_sq / (2.0 * sigma * sigma)).astype(np.float32)
        for c in range(3):
            img[:, :, c] += col[c] * g

    # 3. Dense Low-Opacity Starfield (~1450 stars)
    np.random.seed(42)

    # Dense micro background stars (1100 stars, 1x1, subtle low opacity: brightness 0.10 - 0.35)
    n_micro = 1100
    mx = np.random.randint(0, STAR_IMG_W, size=n_micro)
    my = np.random.randint(0, STAR_IMG_H, size=n_micro)
    mb = 0.10 + 0.25 * np.random.rand(n_micro)
    for i in range(n_micro):
        b = float(mb[i])
        img[mx[i], my[i]] += [b * 0.70, b * 0.80, b * 1.0]

    # Medium field stars (240 stars, 2x2 soft anti-aliased dots)
    n_med = 240
    sx = np.random.randint(0, STAR_IMG_W - 1, size=n_med)
    sy = np.random.randint(0, STAR_IMG_H - 1, size=n_med)
    sb = 0.35 + 0.35 * np.random.rand(n_med)
    hue = np.random.rand(n_med)
    for i in range(n_med):
        b = float(sb[i])
        x, y = sx[i], sy[i]
        col = [b * 0.78, b * 0.86, b * 1.0] if hue[i] < 0.65 else [b * 1.0, b * 0.88, b * 0.72]
        img[x, y] += col
        img[x + 1, y] += [col[0] * 0.45, col[1] * 0.45, col[2] * 0.45]
        img[x, y + 1] += [col[0] * 0.45, col[1] * 0.45, col[2] * 0.45]

    # Prominent field stars (40 stars, 3x3 soft core)
    n_prom = 40
    px_arr = np.random.randint(1, STAR_IMG_W - 1, size=n_prom)
    py_arr = np.random.randint(1, STAR_IMG_H - 1, size=n_prom)
    pb = 0.55 + 0.30 * np.random.rand(n_prom)
    for i in range(n_prom):
        b = float(pb[i])
        x, y = px_arr[i], py_arr[i]
        c_core = [b * 0.88, b * 0.92, b * 1.0]
        c_soft = [b * 0.30, b * 0.35, b * 0.45]
        img[x, y] += c_core
        img[x - 1, y] += c_soft; img[x + 1, y] += c_soft
        img[x, y - 1] += c_soft; img[x, y + 1] += c_soft

    np.clip(img, 0.0, 1.0, out=img)
    star_bg_field.from_numpy(img)


# -----------------------------------------------------------------------------
# 5. Initialization Helper & Baseline Capture  (UNCHANGED)
# -----------------------------------------------------------------------------

def setup_initial_conditions():
    """Configures the initial orbital state of host star, 2 planets, and rogue star."""
    m0 = 1.0
    p0 = [0.0, 0.0]
    v0 = [0.0, 0.0]

    r1 = 1.0
    m1 = 3.0e-6
    v1_mag = math.sqrt(G_CONST * m0 / r1)
    p1 = [r1, 0.0]
    v1 = [0.0, v1_mag]

    r2 = 1.8
    m2 = 1.0e-5
    v2_mag = math.sqrt(G_CONST * m0 / r2)
    p2 = [0.0, r2]
    v2 = [-v2_mag, 0.0]

    m3 = 0.6
    p3 = [-4.5, -2.0]
    v3 = [3.2, 1.1]

    init_pos[0] = p0; init_vel[0] = v0; init_mass[0] = m0
    init_pos[1] = p1; init_vel[1] = v1; init_mass[1] = m1
    init_pos[2] = p2; init_vel[2] = v2; init_mass[2] = m2
    init_pos[3] = p3; init_vel[3] = v3; init_mass[3] = m3

    body_colors[0] = [0.98, 0.94, 0.82]
    render_radius[0] = 0.015

    body_colors[1] = [0.37, 0.78, 0.85]
    render_radius[1] = 0.0145

    body_colors[2] = [0.88, 0.54, 0.435]
    render_radius[2] = 0.0160

    body_colors[3] = [0.78, 0.84, 1.0]
    render_radius[3] = 0.0115

    reset_to_initial()


# -----------------------------------------------------------------------------
# 6. Main Execution & Interactive Loop
# -----------------------------------------------------------------------------

def main():
    import time as _time
    setup_initial_conditions()
    generate_starfield()

    m0 = 1.0
    r0_1 = 1.0
    v0_1 = math.sqrt(G_CONST * m0 / r0_1)
    E0_1 = 0.5 * (v0_1 ** 2) - (G_CONST * m0 / r0_1)

    r0_2 = 1.8
    v0_2 = math.sqrt(G_CONST * m0 / r0_2)
    E0_2 = 0.5 * (v0_2 ** 2) - (G_CONST * m0 / r0_2)

    min_rogue_dist = float('inf')

    window = ti.ui.Window("STELLAR DISRUPTION  |  Rogue Star Flyby Simulation",
                          res=(WIN_W, WIN_H),
                          vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    BASE_SCALE_Y = 3.2
    MAX_SCALE_Y = 8.5
    scale_y = BASE_SCALE_Y
    scale_x = scale_y * (WIN_W / WIN_H)

    paused = False
    rogue_enabled = False
    speed_scale = 0.3  # Starts at 0.3x speed as requested
    sim_time = 0.0

    print("=" * 60, flush=True)
    print("  STELLAR DISRUPTION — Rogue Star Flyby Simulation", flush=True)
    print("=" * 60, flush=True)
    print("  SPACE  : Pause / Resume", flush=True)
    print("  S      : Toggle Rogue Star (ACTIVE / DISABLED)", flush=True)
    print("  R      : Reset system", flush=True)
    print("  UP/DN  : Speed +/-", flush=True)
    print("=" * 60, flush=True)

    trail_sample_counter = 0

    NARRATION_DURATION = 4.5
    narrations   = []
    E1_prev_sign = None
    E2_prev_sign = None

    def _launch_intruder():
        nonlocal rogue_enabled
        rogue_enabled = True

    while window.running:
        # --- Handle User Controls ---
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.SPACE:
                paused = not paused

            elif event.key == 's' or event.key == 'S':
                if not rogue_enabled:
                    _launch_intruder()
                else:
                    rogue_enabled = False

            elif event.key == 'r' or event.key == 'R':
                reset_to_initial()
                sim_time         = 0.0
                speed_scale      = 0.3
                min_rogue_dist   = float('inf')
                scale_y          = BASE_SCALE_Y
                scale_x          = scale_y * (WIN_W / WIN_H)
                rogue_enabled    = False
                narrations.clear()
                E1_prev_sign     = None
                E2_prev_sign     = None

            elif event.key == ti.ui.UP:
                speed_scale = min(speed_scale * 1.5, 8.0)
            elif event.key == ti.ui.DOWN:
                speed_scale = max(speed_scale / 1.5, 0.125)

        rogue_flag = 1 if rogue_enabled else 0

        # --- Physics Integration Step (UNCHANGED) ---
        if not paused:
            dt = BASE_DT * speed_scale
            for _ in range(SUBSTEPS_PER_FRAME):
                verlet_step(dt, rogue_flag)
                sim_time += dt

            trail_sample_counter += 1
            if trail_sample_counter % 2 == 0:
                record_trail_history(rogue_flag)

        # --- Real-Time Scientific Impact Calculations (UNCHANGED) ---
        pos_arr = pos.to_numpy()
        vel_arr = vel.to_numpy()

        d_rogue            = 0.0
        encounter_in_progress = False
        post_encounter     = False

        if rogue_enabled:
            d_rogue = math.sqrt(pos_arr[3][0]**2 + pos_arr[3][1]**2)
            if d_rogue < min_rogue_dist:
                min_rogue_dist = d_rogue
            if d_rogue <= 4.0:
                encounter_in_progress = True
            elif min_rogue_dist < 4.0 and d_rogue > 4.0:
                post_encounter = True

        r1 = math.sqrt(pos_arr[1][0]**2 + pos_arr[1][1]**2)
        v1 = math.sqrt(vel_arr[1][0]**2 + vel_arr[1][1]**2)
        E1 = 0.5 * (v1**2) - (G_CONST * m0 / max(r1, 1e-4))
        dE1 = ((E1 - E0_1) / abs(E0_1)) * 100.0

        if E1 >= 0.0:
            status1 = "POTENTIALLY UNBOUND" if encounter_in_progress else "UNBOUND / ESCAPED"
        else:
            status1 = "BOUND"

        r2 = math.sqrt(pos_arr[2][0]**2 + pos_arr[2][1]**2)
        v2 = math.sqrt(vel_arr[2][0]**2 + vel_arr[2][1]**2)
        E2 = 0.5 * (v2**2) - (G_CONST * m0 / max(r2, 1e-4))
        dE2 = ((E2 - E0_2) / abs(E0_2)) * 100.0

        if E2 >= 0.0:
            status2 = "POTENTIALLY UNBOUND" if encounter_in_progress else "UNBOUND / ESCAPED"
        else:
            status2 = "BOUND"

        # --- Dynamic Camera Zoom-Out (Smooth exponential tracking) ---
        r_max_planets = max(r1, r2)
        if rogue_enabled and d_rogue <= 5.5:
            r_interest = max(r_max_planets, d_rogue)
        else:
            r_interest = r_max_planets

        req_scale_y = r_interest * 1.30
        target_scale_y = min(max(req_scale_y, BASE_SCALE_Y), MAX_SCALE_Y)

        zoom_rate = 2.5
        frame_dt = BASE_DT * speed_scale if not paused else 0.016
        zoom_factor = 1.0 - math.exp(-zoom_rate * min(frame_dt * 50.0, 0.2))
        scale_y += (target_scale_y - scale_y) * zoom_factor
        scale_x = scale_y * (WIN_W / WIN_H)

        # Update screen coordinates and trail line geometry with dynamically scaled camera
        update_render_positions(scale_x, scale_y, rogue_flag)
        build_trail_lines(scale_x, scale_y, rogue_flag)

        # --- Encounter Narration on Energy Sign-Crossings ---
        now_wall = _time.perf_counter()
        if encounter_in_progress:
            E1_sign = (E1 >= 0.0)
            E2_sign = (E2 >= 0.0)

            if E1_prev_sign is not None and E1_sign != E1_prev_sign:
                msg = ("  Inner Planet has become gravitationally unbound"
                       if E1_sign else
                       "  Inner Planet recaptured into a bound orbit")
                narrations.append({'msg': msg,
                                   'color': (0.937, 0.424, 0.459) if E1_sign else (0.596, 0.765, 0.475),
                                   'born': now_wall})

            if E2_prev_sign is not None and E2_sign != E2_prev_sign:
                msg = ("  Outer Planet has become gravitationally unbound"
                       if E2_sign else
                       "  Outer Planet recaptured into a bound orbit")
                narrations.append({'msg': msg,
                                   'color': (0.937, 0.424, 0.459) if E2_sign else (0.596, 0.765, 0.475),
                                   'born': now_wall})

            E1_prev_sign = E1_sign
            E2_prev_sign = E2_sign

        narrations = [n for n in narrations if (now_wall - n['born']) < NARRATION_DURATION]

        # =====================================================================
        # --- Drawing Canvas (Path 1 Per-Pixel GPU Shader Rendering) ---
        # =====================================================================
        render_scene_pixel_shader(scale_x, scale_y, rogue_flag)
        canvas.set_image(star_img)
        canvas.lines(trail_line_vertices, width=0.0016, per_vertex_color=trail_line_colors)

        # =====================================================================
        # --- HUD: Single Compact Left-Anchored Panel (Responsive Layout) ---
        # =====================================================================
        CLR_HEAD  = (0.930, 0.950, 0.980)
        CLR_SEC   = (0.850, 0.880, 0.920)
        CLR_BODY  = (0.847, 0.867, 0.890)
        CLR_LABEL = (0.478, 0.510, 0.565)
        CLR_HOST  = (0.910, 0.725, 0.290)
        CLR_INNER = (0.369, 0.784, 0.847)
        CLR_OUTER = (0.878, 0.541, 0.435)
        CLR_ROGUE = (0.788, 0.839, 1.000)
        CLR_BOUND = (0.596, 0.765, 0.475)
        CLR_ALERT = (0.937, 0.424, 0.459)
        DIV = "-------------------"

        PX = 0.006
        PY = 0.008
        PW = 0.190

        _LH = 0.026
        _PAD = 0.014
        _n = 26
        if rogue_enabled:
            _n += 3
            if min_rogue_dist < 999.0:
                _n += 1
        PH = min(_n * _LH + _PAD, 0.984)

        gui.begin("##hud", PX, PY, PW, PH)

        # Title
        gui.text("STELLAR DISRUPTION", color=CLR_HEAD)
        gui.text("Rogue Star Flyby", color=CLR_LABEL)
        gui.text(DIV, color=CLR_LABEL)

        # Phase & Status
        gui.text("PHASE", color=CLR_LABEL)
        if not rogue_enabled:
            gui.text("  Pre-Encounter", color=CLR_BODY)
        elif encounter_in_progress:
            gui.text("  FLYBY IN PROGRESS", color=CLR_ALERT)
        elif post_encounter:
            gui.text("  Post-Encounter", color=CLR_BOUND)
        else:
            gui.text("  Intruder Approaching", color=CLR_BODY)

        state_str = "PAUSED" if paused else "Running"
        gui.text(f"  {state_str}  {sim_time:.1f} yr  {speed_scale:.1f}x", color=CLR_ALERT if paused else CLR_BODY)
        gui.text(DIV, color=CLR_LABEL)

        # Orbital Data
        gui.text("ORBITAL DATA", color=CLR_SEC)
        gui.text("  Inner Planet", color=CLR_INNER)
        gui.text(f"    r={r1:.3f} AU  v={v1:.2f}", color=CLR_BODY)
        gui.text(f"    E={E1:+.2f}  dE={dE1:+.1f}%", color=CLR_BODY)
        bound_clr1 = CLR_ALERT if "UNBOUND" in status1 else CLR_BOUND
        gui.text(f"    {status1}", color=bound_clr1)

        gui.text("  Outer Planet", color=CLR_OUTER)
        gui.text(f"    r={r2:.3f} AU  v={v2:.2f}", color=CLR_BODY)
        gui.text(f"    E={E2:+.2f}  dE={dE2:+.1f}%", color=CLR_BODY)
        bound_clr2 = CLR_ALERT if "UNBOUND" in status2 else CLR_BOUND
        gui.text(f"    {status2}", color=bound_clr2)

        # Intruder Status (when active)
        if rogue_enabled:
            gui.text(DIV, color=CLR_LABEL)
            gui.text("INTRUDER", color=CLR_ROGUE)
            gui.text(f"  Dist: {d_rogue:.3f} AU", color=CLR_BODY)
            if min_rogue_dist < 999.0:
                gui.text(f"  Min:  {min_rogue_dist:.3f} AU", color=CLR_BODY)

        # Legend
        gui.text(DIV, color=CLR_LABEL)
        gui.text("LEGEND", color=CLR_LABEL)
        gui.text("  * Host Star  1.0 Msun", color=CLR_HOST)
        gui.text("  o Inner      1.0 AU",   color=CLR_INNER)
        gui.text("  o Outer      1.8 AU",   color=CLR_OUTER)
        gui.text("  * Intruder   0.6 Msun", color=CLR_ROGUE)

        # Controls
        gui.text(DIV, color=CLR_LABEL)
        gui.text("CONTROLS", color=CLR_LABEL)
        gui.text("  Spc Pause  S Launch", color=CLR_LABEL)
        gui.text("  R Reset  Up/Dn Speed", color=CLR_LABEL)

        gui.end()

        # Encounter Narration Banner (bottom center)
        if narrations:
            n = narrations[-1]
            age = now_wall - n['born']
            fade_start = NARRATION_DURATION - 2.0
            if age > fade_start:
                alpha = max(0.0, 1.0 - (age - fade_start) / 2.0)
                c = n['color']
                nc = (c[0] * alpha, c[1] * alpha, c[2] * alpha)
            else:
                nc = n['color']
            gui.begin("##narration", 0.200, 0.955, 0.600, 0.038)
            gui.text(n['msg'].strip(), color=nc)
            gui.end()

        window.show()


if __name__ == "__main__":
    main()
