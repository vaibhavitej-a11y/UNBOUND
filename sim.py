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

# Orbit Trail Fields (Ring buffer for 3 trails)
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
            a_sum = ti.Vector([0.0, 0.0])
            for j in range(NUM_BODIES):
                if i != j:
                    # Exclude Rogue Star (j=3) from forces if disabled
                    if (j == 3 or i == 3) and rogue_enabled == 0:
                        continue
                    r_vec = pos[j] - pos[i]
                    r2 = r_vec.norm_sqr() + EPS_SQ
                    r = ti.sqrt(r2)
                    # Gravitational force field: a_i = sum_j (G * m_j * r_ij / r_ij^3)
                    a_sum += (G_CONST * mass[j] / (r2 * r)) * r_vec
            acc[i] = a_sum


@ti.kernel
def verlet_half_step_1(dt: ti.f32, rogue_enabled: ti.i32):
    """First half of Velocity Verlet integrator: update velocity by half step and position by full step."""
    for i in range(NUM_BODIES):
        if i == 0:
            continue
        if i == 3 and rogue_enabled == 0:
            continue
        vel[i] += 0.5 * acc[i] * dt
        pos[i] += vel[i] * dt


@ti.kernel
def verlet_half_step_2(dt: ti.f32, rogue_enabled: ti.i32):
    """Second half of Velocity Verlet integrator: update velocity by remaining half step."""
    for i in range(NUM_BODIES):
        if i == 0:
            continue
        if i == 3 and rogue_enabled == 0:
            continue
        vel[i] += 0.5 * acc[i] * dt


def verlet_step(dt: float, rogue_enabled: int):
    """Executes one complete Velocity Verlet integration step."""
    verlet_half_step_1(dt, rogue_enabled)
    compute_accelerations(rogue_enabled)
    verlet_half_step_2(dt, rogue_enabled)


@ti.kernel
def reset_to_initial():
    """Resets positions, velocities, and masses to initial configurations."""
    for i in range(NUM_BODIES):
        pos[i] = init_pos[i]
        vel[i] = init_vel[i]
        mass[i] = init_mass[i]
        acc[i] = ti.Vector([0.0, 0.0])
    trail_head[None] = 0
    trail_count[None] = 0


# -----------------------------------------------------------------------------
# 4. Rendering & Coordinate Mapping Kernels  (UNCHANGED PHYSICS — visual only)
# -----------------------------------------------------------------------------

@ti.kernel
def update_render_positions(scale_x: ti.f32, scale_y: ti.f32, rogue_enabled: ti.i32):
    """Transforms simulation coordinates (AU) into normalized screen coordinates [0, 1]^2."""
    for i in range(NUM_BODIES):
        if i == 3 and rogue_enabled == 0:
            # Hide rogue star off-screen when disabled
            render_pos[i] = ti.Vector([-10.0, -10.0])
        else:
            nx = 0.5 + pos[i][0] / (2.0 * scale_x)
            ny = 0.5 + pos[i][1] / (2.0 * scale_y)
            render_pos[i] = ti.Vector([nx, ny])


@ti.func
def smoothstep(edge0: ti.f32, edge1: ti.f32, x: ti.f32) -> ti.f32:
    t = ti.max(0.0, ti.min(1.0, (x - edge0) / (edge1 - edge0 + 1e-6)))
    return t * t * (3.0 - 2.0 * t)


@ti.kernel
def render_scene_pixel_shader(scale_x: ti.f32, scale_y: ti.f32, rogue_enabled: ti.i32):
    """Per-pixel GPU rendering shader for true Gaussian star bloom and 3D planet sphere shading."""
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
        # 1. Host Star Gaussian Bloom I(r) = I0 * exp(-r^2 / sigma^2)
        # ---------------------------------------------------------------------
        dx0 = (u - cx0) * aspect
        dy0 = (v - cy0)
        r0_sq = dx0 * dx0 + dy0 * dy0
        r0 = ti.sqrt(r0_sq)

        # Continuous Gaussian bloom (sigma = 0.028)
        sigma0 = 0.028
        I0_bloom = 0.65 * ti.exp(-r0_sq / (sigma0 * sigma0))
        amber_glow = ti.Vector([0.91, 0.725, 0.29]) * I0_bloom
        col += amber_glow

        # Host Star Anti-Aliased Core Circle (radius 0.012)
        if r0 <= 0.012:
            t_core = smoothstep(0.012, 0.007, r0)
            core_col = ti.Vector([0.98, 0.94, 0.82])
            col = col * (1.0 - t_core) + core_col * t_core

        # ---------------------------------------------------------------------
        # 2. Compact Stellar Intruder Bloom (Body 3, if active)
        # ---------------------------------------------------------------------
        if rogue_enabled == 1:
            dx3 = (u - cx3) * aspect
            dy3 = (v - cy3)
            r3_sq = dx3 * dx3 + dy3 * dy3
            r3 = ti.sqrt(r3_sq)

            sigma3 = 0.020
            I3_bloom = 0.45 * ti.exp(-r3_sq / (sigma3 * sigma3))
            blue_white_glow = ti.Vector([0.79, 0.84, 1.00]) * I3_bloom
            col += blue_white_glow

            if r3 <= 0.010:
                t_core3 = smoothstep(0.010, 0.005, r3)
                core3_col = ti.Vector([0.88, 0.92, 1.00])
                col = col * (1.0 - t_core3) + core3_col * t_core3

        # ---------------------------------------------------------------------
        # 3. Planet 3D Sphere Shading (Inner Planet: Cyan, Outer Planet: Coral)
        # ---------------------------------------------------------------------
        for p in range(2):
            cx_p = cx1 if p == 0 else cx2
            cy_p = cy1 if p == 0 else cy2

            dx_p = (u - cx_p) * aspect
            dy_p = (v - cy_p)

            # Visual radii for planets (~45% larger)
            R_p = 0.0115 if p == 0 else 0.0125
            r_p_sq = (dx_p * dx_p + dy_p * dy_p) / (R_p * R_p)

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

                # Smooth Lambertian Terminator (no visible banding)
                diffuse = 0.03 + 0.97 * ti.pow(NdotL, 0.85)

                # Specular Peak H = normalize(L + (0, 0, 1))
                V = ti.Vector([0.0, 0.0, 1.0])
                H = (L + V).normalized()
                NdotH = ti.max(0.0, N.dot(H))
                specular = ti.pow(NdotH, 20.0)

                # Limb Darkening
                limb = 1.0 - 0.30 * (1.0 - nz) * (1.0 - nz)

                # Palette Base Colors:
                # Inner planet (p=0): cool cyan #5ec8d8 -> [0.37, 0.78, 0.85]
                # Outer planet (p=1): soft coral #e08a6f -> [0.88, 0.54, 0.435]
                p_base = ti.Vector([0.37, 0.78, 0.85]) if p == 0 else ti.Vector([0.88, 0.54, 0.435])
                p_spec_col = ti.Vector([1.0, 1.0, 1.0])

                # Shaded surface
                shaded_col = p_base * diffuse * limb + p_spec_col * specular * 0.45

                # Edge Anti-Aliasing
                edge_alpha = smoothstep(1.0, 0.85, ti.sqrt(r_p_sq))
                col = col * (1.0 - edge_alpha) + shaded_col * edge_alpha

        # Write final pixel color
        star_img[px, py] = col


@ti.kernel
def record_trail_history(scale_x: ti.f32, scale_y: ti.f32, rogue_enabled: ti.i32):
    """Records current normalized body positions into ring buffers."""
    head = trail_head[None]
    for t in range(NUM_TRAILS):
        body_idx = t + 1  # 0: Planet 1 (idx 1), 1: Planet 2 (idx 2), 2: Rogue Star (idx 3)
        if body_idx == 3 and rogue_enabled == 0:
            trail_history[t, head] = ti.Vector([-10.0, -10.0])
        else:
            nx = 0.5 + pos[body_idx][0] / (2.0 * scale_x)
            ny = 0.5 + pos[body_idx][1] / (2.0 * scale_y)
            trail_history[t, head] = ti.Vector([nx, ny])

    trail_head[None] = (head + 1) % MAX_TRAIL_LEN
    if trail_count[None] < MAX_TRAIL_LEN:
        trail_count[None] += 1


@ti.kernel
def build_trail_lines(rogue_enabled: ti.i32):
    """Builds vertex line segments and colors for canvas.lines() from ring buffer."""
    count = trail_count[None]
    head = trail_head[None]

    for t in range(NUM_TRAILS):
        body_idx = t + 1
        color = body_colors[body_idx]
        base_vert_idx = t * TRAIL_VERTICES_PER_BODY

        # Fill line segment pairs
        for seg in range(MAX_TRAIL_LEN - 1):
            v_idx = base_vert_idx + seg * 2

            if seg < count - 1 and not (body_idx == 3 and rogue_enabled == 0):
                # Ring buffer indices relative to current head
                idx0 = (head - count + seg) % MAX_TRAIL_LEN
                if idx0 < 0:
                    idx0 += MAX_TRAIL_LEN
                idx1 = (idx0 + 1) % MAX_TRAIL_LEN

                pt0 = trail_history[t, idx0]
                pt1 = trail_history[t, idx1]

                # Don't draw segment if either point is off-screen
                if pt0[0] < -1.0 or pt1[0] < -1.0:
                    trail_line_vertices[v_idx] = ti.Vector([-1.0, -1.0])
                    trail_line_vertices[v_idx + 1] = ti.Vector([-1.0, -1.0])
                    trail_line_colors[v_idx] = ti.Vector([0.0, 0.0, 0.0])
                    trail_line_colors[v_idx + 1] = ti.Vector([0.0, 0.0, 0.0])
                else:
                    trail_line_vertices[v_idx] = pt0
                    trail_line_vertices[v_idx + 1] = pt1

                    # Aggressive power-law fade: full opacity near planet, near-fully transparent at tail
                    alpha = float(seg) / float(MAX_TRAIL_LEN - 1)
                    fade = ti.pow(alpha, 2.2)
                    seg_color = color * (0.005 + 0.995 * fade)
                    trail_line_colors[v_idx] = seg_color
                    trail_line_colors[v_idx + 1] = seg_color
            else:
                # Degenerate segments off-screen
                trail_line_vertices[v_idx] = ti.Vector([-1.0, -1.0])
                trail_line_vertices[v_idx + 1] = ti.Vector([-1.0, -1.0])
                trail_line_colors[v_idx] = ti.Vector([0.0, 0.0, 0.0])
                trail_line_colors[v_idx + 1] = ti.Vector([0.0, 0.0, 0.0])


def generate_starfield():
    """Generates space-agency style near-black background #05070c with low-density stars once at startup."""
    import numpy as np

    # 1. Base Deep Space Canvas (WIN_W x WIN_H x 3)
    img = np.zeros((STAR_IMG_W, STAR_IMG_H, 3), dtype=np.float32)

    # Base background: near-black #05070c -> RGB [0.016, 0.022, 0.040]
    for py in range(STAR_IMG_H):
        t = py / float(STAR_IMG_H)
        img[:, py, 0] = 0.016 + 0.005 * (1.0 - t)  # Red
        img[:, py, 1] = 0.022 + 0.006 * (1.0 - t)  # Green
        img[:, py, 2] = 0.040 + 0.010 * (1.0 - t)  # Blue

    # 2. Restored Space-Agency Low-Density Starfield (750 stars, 1x1 to 3x3 soft dots)
    np.random.seed(42)

    # Micro background stars (550 stars, 1x1)
    n_micro = 550
    mx = np.random.randint(0, STAR_IMG_W, size=n_micro)
    my = np.random.randint(0, STAR_IMG_H, size=n_micro)
    mb = 0.25 + 0.45 * np.random.rand(n_micro)
    for i in range(n_micro):
        b = float(mb[i])
        img[mx[i], my[i]] += [b * 0.72, b * 0.82, b * 1.0]

    # Medium field stars (160 stars, 2x2 soft dots)
    n_med = 160
    sx = np.random.randint(0, STAR_IMG_W - 1, size=n_med)
    sy = np.random.randint(0, STAR_IMG_H - 1, size=n_med)
    sb = 0.45 + 0.45 * np.random.rand(n_med)
    hue = np.random.rand(n_med)
    for i in range(n_med):
        b = float(sb[i])
        x, y = sx[i], sy[i]
        col = [b * 0.80, b * 0.88, b * 1.0] if hue[i] < 0.6 else [b * 1.0, b * 0.90, b * 0.75]
        img[x, y] += col
        img[x + 1, y] += [col[0] * 0.4, col[1] * 0.4, col[2] * 0.4]
        img[x, y + 1] += [col[0] * 0.4, col[1] * 0.4, col[2] * 0.4]

    # Prominent field stars (40 stars, 3x3 soft core)
    n_prom = 40
    px_arr = np.random.randint(1, STAR_IMG_W - 1, size=n_prom)
    py_arr = np.random.randint(1, STAR_IMG_H - 1, size=n_prom)
    pb = 0.65 + 0.30 * np.random.rand(n_prom)
    for i in range(n_prom):
        b = float(pb[i])
        x, y = px_arr[i], py_arr[i]
        c_core = [b * 0.90, b * 0.94, b * 1.0]
        c_soft = [b * 0.35, b * 0.40, b * 0.50]
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
    # Body 0: Host Star
    m0 = 1.0
    p0 = [0.0, 0.0]
    v0 = [0.0, 0.0]

    # Body 1: Planet 1 (Inner Planet - Earth-like orbit)
    r1 = 1.0
    m1 = 3.0e-6  # ~1 Earth Mass
    v1_mag = math.sqrt(G_CONST * m0 / r1)  # ~ 6.28318 AU/yr
    p1 = [r1, 0.0]
    v1 = [0.0, v1_mag]

    # Body 2: Planet 2 (Outer Planet - Mars/Super-Earth orbit)
    r2 = 1.8
    m2 = 1.0e-5  # ~3.3 Earth Masses
    v2_mag = math.sqrt(G_CONST * m0 / r2)  # ~ 4.683 AU/yr
    p2 = [0.0, r2]
    v2 = [-v2_mag, 0.0]

    # Body 3: Rogue Star (Hyperbolic Encounter Trajectory)
    m3 = 0.6     # 0.6 Solar Masses
    p3 = [-4.5, -2.0]  # Starting position at outer periphery
    v3 = [3.2, 1.1]    # Hyperbolic flyby velocity vector (AU/yr)

    # Populate host initial numpy buffers then copy into Taichi fields
    init_pos[0] = p0; init_vel[0] = v0; init_mass[0] = m0
    init_pos[1] = p1; init_vel[1] = v1; init_mass[1] = m1
    init_pos[2] = p2; init_vel[2] = v2; init_mass[2] = m2
    init_pos[3] = p3; init_vel[3] = v3; init_mass[3] = m3

    # Visual Properties & Palette Matching Pass A Specifications
    # Body 0 (Host Star): Warm Amber / Gold (#e8b94a core)
    body_colors[0] = [0.98, 0.94, 0.82]
    render_radius[0] = 0.012

    # Body 1 (Inner Planet): Cool Cyan (#5ec8d8)
    body_colors[1] = [0.37, 0.78, 0.85]
    render_radius[1] = 0.007

    # Body 2 (Outer Planet): Soft Coral (#e08a6f)
    body_colors[2] = [0.88, 0.54, 0.435]
    render_radius[2] = 0.008

    # Body 3 (Compact Stellar Intruder): Cool Blue-White (#c9d6ff)
    body_colors[3] = [0.79, 0.84, 1.00]
    render_radius[3] = 0.010

    reset_to_initial()
    compute_accelerations(0)


# -----------------------------------------------------------------------------
# 6. Main Execution & Interactive Loop
# -----------------------------------------------------------------------------

def main():
    setup_initial_conditions()

    # Generate starfield background once (static)
    generate_starfield()

    # Calculate baseline physical quantities for planets relative to Host Star
    # (UNCHANGED scientific calculations)
    m0 = 1.0
    r0_1 = 1.0
    v0_1 = math.sqrt(G_CONST * m0 / r0_1)
    E0_1 = 0.5 * (v0_1 ** 2) - (G_CONST * m0 / r0_1)

    r0_2 = 1.8
    v0_2 = math.sqrt(G_CONST * m0 / r0_2)
    E0_2 = 0.5 * (v0_2 ** 2) - (G_CONST * m0 / r0_2)

    min_rogue_dist = float('inf')

    # Create Taichi GGUI Window
    window = ti.ui.Window("STELLAR DISRUPTION  |  Rogue Star Flyby Simulation",
                          res=(WIN_W, WIN_H),
                          vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    # Viewport scaling: wider scale keeps all bodies visible during flyby
    # 3.2 AU Y-extent prevents clipping at periastron approach angles
    scale_y = 3.2
    scale_x = scale_y * (WIN_W / WIN_H)

    paused = False
    rogue_enabled = False
    speed_scale = 1.0
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

    while window.running:
        # --- Handle User Controls ---
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.SPACE:
                paused = not paused
            elif event.key == 's' or event.key == 'S':
                rogue_enabled = not rogue_enabled
            elif event.key == 'r' or event.key == 'R':
                reset_to_initial()
                sim_time = 0.0
                min_rogue_dist = float('inf')
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

            # Record trail every frame for smooth, capped-length trails
            trail_sample_counter += 1
            if trail_sample_counter % 2 == 0:
                record_trail_history(scale_x, scale_y, rogue_flag)

        # Update screen coordinates and trail line geometry
        update_render_positions(scale_x, scale_y, rogue_flag)
        build_trail_lines(rogue_flag)

        # --- Real-Time Scientific Impact Calculations (UNCHANGED) ---
        pos_arr = pos.to_numpy()
        vel_arr = vel.to_numpy()

        # Real-time encounter state
        d_rogue = 0.0
        encounter_in_progress = False
        post_encounter = False

        if rogue_enabled:
            d_rogue = math.sqrt(pos_arr[3][0]**2 + pos_arr[3][1]**2)
            if d_rogue < min_rogue_dist:
                min_rogue_dist = d_rogue

            # Encounter is in progress if rogue star is within 4.0 AU of host star
            if d_rogue <= 4.0:
                encounter_in_progress = True
            elif min_rogue_dist < 4.0 and d_rogue > 4.0:
                post_encounter = True

        # Planet 1 (Inner Planet - Index 1)
        r1 = math.sqrt(pos_arr[1][0]**2 + pos_arr[1][1]**2)
        v1 = math.sqrt(vel_arr[1][0]**2 + vel_arr[1][1]**2)
        E1 = 0.5 * (v1**2) - (G_CONST * m0 / max(r1, 1e-4))
        dE1 = ((E1 - E0_1) / abs(E0_1)) * 100.0

        if E1 >= 0.0:
            status1 = "POTENTIALLY UNBOUND" if encounter_in_progress else "UNBOUND / ESCAPED"
        else:
            status1 = "BOUND"

        # Planet 2 (Outer Planet - Index 2)
        r2 = math.sqrt(pos_arr[2][0]**2 + pos_arr[2][1]**2)
        v2 = math.sqrt(vel_arr[2][0]**2 + vel_arr[2][1]**2)
        E2 = 0.5 * (v2**2) - (G_CONST * m0 / max(r2, 1e-4))
        dE2 = ((E2 - E0_2) / abs(E0_2)) * 100.0

        if E2 >= 0.0:
            status2 = "POTENTIALLY UNBOUND" if encounter_in_progress else "UNBOUND / ESCAPED"
        else:
            status2 = "BOUND"

        # --- Encounter phase label for status bar ---
        if rogue_enabled:
            if encounter_in_progress:
                phase_label = "FLYBY IN PROGRESS"
            elif post_encounter:
                phase_label = "POST-ENCOUNTER"
            else:
                phase_label = "APPROACHING"
        else:
            phase_label = "Compact Stellar Intruder DISABLED  —  press S to activate"

        # =====================================================================
        # --- Drawing Canvas (Pass A Space-Agency Palette & Sphere Shading) ---
        # =====================================================================

        # =====================================================================
        # --- Drawing Canvas (Path 1 Per-Pixel GPU Shader Rendering) ---
        # =====================================================================

        # 1. Execute per-pixel GPU shader (Gaussian Star Bloom + 3D Planet Shading + Intruder Bloom)
        render_scene_pixel_shader(scale_x, scale_y, rogue_flag)
        canvas.set_image(star_img)

        # 2. Orbit & Flyby Trails (power-law fade to transparent at tail)
        canvas.lines(trail_line_vertices, width=0.0022, per_vertex_color=trail_line_colors)

        # =====================================================================
        # --- GUI Overlays (Pass B: Space-Agency Mission Dashboard Aesthetic) ---
        # =====================================================================

        # Mission Palette & Color Hierarchy
        CLR_HEAD  = (0.930, 0.950, 0.980)  # Header titles / crisp white
        CLR_SEC   = (0.850, 0.880, 0.920)  # Section headers
        CLR_BODY  = (0.847, 0.867, 0.890)  # Main text #d8dde3
        CLR_LABEL = (0.478, 0.510, 0.565)  # Secondary / static labels #7a8290
        CLR_HOST  = (0.910, 0.725, 0.290)  # Host star amber #e8b94a
        CLR_INNER = (0.369, 0.784, 0.847)  # Inner planet cyan #5ec8d8
        CLR_OUTER = (0.878, 0.541, 0.435)  # Outer planet coral #e08a6f
        CLR_ROGUE = (0.788, 0.839, 1.000)  # Compact Intruder #c9d6ff
        CLR_BOUND = (0.596, 0.765, 0.475)  # Green bound state
        CLR_ALERT = (0.937, 0.424, 0.459)  # Red/Coral unbound alert

        DIVIDER = "------------------------"

        # Left Dock — Title, System Status, Legend & Controls
        gui.begin("##left", 0.005, 0.005, 0.205, 0.925)
        gui.text("STELLAR DISRUPTION", color=CLR_HEAD)
        gui.text("Rogue Star Flyby Simulation", color=CLR_LABEL)
        gui.text(DIVIDER, color=CLR_LABEL)

        # Emphasized Live System Status
        gui.text("SYSTEM STATUS", color=CLR_SEC)
        gui.text(f"  State:   {'PAUSED' if paused else 'RUNNING'}", color=CLR_ALERT if paused else CLR_BOUND)
        gui.text(f"  Time:    {sim_time:7.2f} yr", color=CLR_BODY)
        gui.text(f"  Speed:   {speed_scale:7.2f}x", color=CLR_BODY)
        gui.text(DIVIDER, color=CLR_LABEL)

        # De-emphasized Static Celestial Legend
        gui.text("CELESTIAL LEGEND", color=CLR_LABEL)
        gui.text("  * Host Star   1.0 Msun", color=CLR_HOST)
        gui.text("  o Inner Planet 1.0 AU", color=CLR_INNER)
        gui.text("  o Outer Planet 1.8 AU", color=CLR_OUTER)
        gui.text("  * Intruder     0.6 Msun", color=CLR_ROGUE)
        gui.text(DIVIDER, color=CLR_LABEL)

        # De-emphasized Static Controls
        gui.text("MISSION CONTROLS", color=CLR_LABEL)
        gui.text("  Space   Pause / Resume", color=CLR_LABEL)
        gui.text("  S       Toggle Intruder", color=CLR_LABEL)
        gui.text("  R       Reset System", color=CLR_LABEL)
        gui.text("  Up/Dn   Adjust Speed", color=CLR_LABEL)
        gui.end()

        # Right Dock — Scientific Measurements & Orbital Analysis
        gui.begin("##right", 0.790, 0.005, 0.205, 0.925)
        gui.text("ORBITAL ANALYSIS", color=CLR_HEAD)
        gui.text("E = 0.5v^2 - GM/r", color=CLR_LABEL)
        gui.text(DIVIDER, color=CLR_LABEL)

        # Inner Planet Readout
        gui.text("INNER PLANET", color=CLR_INNER)
        gui.text(f"  r   {r1:7.3f} AU (ini {r0_1:.2f})", color=CLR_BODY)
        gui.text(f"  v   {v1:7.3f} AU/yr", color=CLR_BODY)
        gui.text(f"  E   {E1:+7.2f}", color=CLR_BODY)
        gui.text(f"  dE  {dE1:+7.1f}%", color=CLR_ALERT if abs(dE1) > 10 else CLR_BODY)
        gui.text(f"  {status1}", color=CLR_ALERT if "UNBOUND" in status1 else CLR_BOUND)
        gui.text(DIVIDER, color=CLR_LABEL)

        # Outer Planet Readout
        gui.text("OUTER PLANET", color=CLR_OUTER)
        gui.text(f"  r   {r2:7.3f} AU (ini {r0_2:.2f})", color=CLR_BODY)
        gui.text(f"  v   {v2:7.3f} AU/yr", color=CLR_BODY)
        gui.text(f"  E   {E2:+7.2f}", color=CLR_BODY)
        gui.text(f"  dE  {dE2:+7.1f}%", color=CLR_ALERT if abs(dE2) > 10 else CLR_BODY)
        gui.text(f"  {status2}", color=CLR_ALERT if "UNBOUND" in status2 else CLR_BOUND)
        gui.text(DIVIDER, color=CLR_LABEL)

        # Intruder Encounter Metrics
        gui.text("INTRUDER ENCOUNTER", color=CLR_ROGUE)
        if rogue_enabled:
            gui.text(f"  Dist  {d_rogue:7.3f} AU", color=CLR_BODY)
            min_str = f"{min_rogue_dist:7.3f} AU" if min_rogue_dist < 999.0 else "---"
            gui.text(f"  Min   {min_str}", color=CLR_BODY)
        else:
            gui.text("  Status: DISABLED", color=CLR_LABEL)
        gui.text(DIVIDER, color=CLR_LABEL)

        # Energy Reference Guide
        gui.text("ENERGY REFERENCE", color=CLR_LABEL)
        gui.text("  E < 0   BOUND orbit", color=CLR_BOUND)
        gui.text("  E >= 0  POTENTIALLY UNBOUND", color=CLR_ALERT)
        gui.text("  (3-body transfer)", color=CLR_LABEL)
        gui.end()

        # Bottom Dock — Status Bar Strip Across Bottom
        gui.begin("##statusbar", 0.0, 0.935, 1.0, 0.065)
        rogue_str = f"Intruder: {'ACTIVE' if rogue_enabled else 'DISABLED'}"
        gui.text(f"  {rogue_str}   |   Time: {sim_time:6.2f} yr   |   Phase: {phase_label}   |   Speed: {speed_scale:.2f}x", color=CLR_BODY)
        gui.end()

        window.show()


if __name__ == "__main__":
    main()
