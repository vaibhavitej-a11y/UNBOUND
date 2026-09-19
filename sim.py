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

# Starfield background image (RGBA rendered once on GPU)
star_img = ti.Vector.field(3, dtype=ti.f32, shape=(STAR_IMG_W, STAR_IMG_H))

# Host Star & Intruder Render Fields
host_star_pos = ti.Vector.field(2, dtype=ti.f32, shape=1)
intruder_pos  = ti.Vector.field(2, dtype=ti.f32, shape=1)

# Planet Directional Sphere Layers (shape 2 for Inner & Outer planets)
planet_center_pos   = ti.Vector.field(2, dtype=ti.f32, shape=2)
planet_shadow_pos   = ti.Vector.field(2, dtype=ti.f32, shape=2)
planet_lit_pos      = ti.Vector.field(2, dtype=ti.f32, shape=2)
planet_specular_pos = ti.Vector.field(2, dtype=ti.f32, shape=2)

planet_shadow_colors   = ti.Vector.field(3, dtype=ti.f32, shape=2)
planet_body_colors     = ti.Vector.field(3, dtype=ti.f32, shape=2)
planet_lit_colors      = ti.Vector.field(3, dtype=ti.f32, shape=2)
planet_specular_colors = ti.Vector.field(3, dtype=ti.f32, shape=2)

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


@ti.kernel
def update_glow_positions(scale_x: ti.f32, scale_y: ti.f32, rogue_enabled: ti.i32):
    """Updates rendering positions for host star, intruder, and directional planet sphere shading."""
    # Host Star (index 0)
    nx0 = 0.5 + pos[0][0] / (2.0 * scale_x)
    ny0 = 0.5 + pos[0][1] / (2.0 * scale_y)
    host_star_pos[0] = ti.Vector([nx0, ny0])

    # Compact Stellar Intruder (index 3)
    if rogue_enabled == 1:
        nx3 = 0.5 + pos[3][0] / (2.0 * scale_x)
        ny3 = 0.5 + pos[3][1] / (2.0 * scale_y)
        intruder_pos[0] = ti.Vector([nx3, ny3])
    else:
        intruder_pos[0] = ti.Vector([-10.0, -10.0])

    # Directional Sphere Shading for Planets (indices 1 & 2)
    for p in range(2):
        body_idx = p + 1
        px = pos[body_idx][0]
        py = pos[body_idx][1]

        # Normalized screen center of planet
        cx = 0.5 + px / (2.0 * scale_x)
        cy = 0.5 + py / (2.0 * scale_y)
        planet_center_pos[p] = ti.Vector([cx, cy])

        # Vector pointing FROM planet TO host star (at origin 0, 0)
        d_star = ti.sqrt(px * px + py * py) + 1e-6
        ux = -px / d_star
        uy = -py / d_star

        # Screen-space direction vector
        dx = ux / scale_x
        dy = uy / scale_y
        s_len = ti.sqrt(dx * dx + dy * dy) + 1e-6
        su_x = dx / s_len
        su_y = dy / s_len

        # Directional highlight offsets toward host star (scaled for ~45% larger planet radius)
        planet_shadow_pos[p]   = ti.Vector([cx - su_x * 0.0020, cy - su_y * 0.0020])
        planet_lit_pos[p]      = ti.Vector([cx + su_x * 0.0042, cy + su_y * 0.0042])
        planet_specular_pos[p] = ti.Vector([cx + su_x * 0.0065, cy + su_y * 0.0065])


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
    """Generates a high-fidelity cinematic deep-space background once at startup."""
    import numpy as np

    # 1. Base Deep Space Canvas (WIN_W x WIN_H x 3)
    # Coordinate grids normalized to [0, 1]
    x_coords = np.linspace(0.0, 1.0, STAR_IMG_W, dtype=np.float32)
    y_coords = np.linspace(0.0, 1.0, STAR_IMG_H, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Deep Space Void Base (Navy/Indigo gradient)
    img = np.zeros((STAR_IMG_W, STAR_IMG_H, 3), dtype=np.float32)
    img[:, :, 0] = 0.008 + 0.006 * (1.0 - yy)  # Red
    img[:, :, 1] = 0.010 + 0.008 * (1.0 - yy)  # Green
    img[:, :, 2] = 0.022 + 0.016 * (1.0 - yy)  # Blue

    # 2. Subtle Nebula Clouds (Purple, Blue, Reddish filaments)
    # Purple Nebula (Top Right)
    d_purple2 = (xx - 0.75)**2 + (yy - 0.30)**2
    neb_purple = np.exp(-d_purple2 / 0.12)
    img[:, :, 0] += neb_purple * 0.022
    img[:, :, 1] += neb_purple * 0.008
    img[:, :, 2] += neb_purple * 0.038

    # Deep Cyan/Blue Nebula (Bottom Left)
    d_cyan2 = (xx - 0.22)**2 + (yy - 0.75)**2
    neb_cyan = np.exp(-d_cyan2 / 0.15)
    img[:, :, 0] += neb_cyan * 0.006
    img[:, :, 1] += neb_cyan * 0.024
    img[:, :, 2] += neb_cyan * 0.036

    # Reddish Cosmic Filament (Center Top)
    d_red2 = (xx - 0.45)**2 + (yy - 0.18)**2
    neb_red = np.exp(-d_red2 / 0.08)
    img[:, :, 0] += neb_red * 0.020
    img[:, :, 1] += neb_red * 0.006
    img[:, :, 2] += neb_red * 0.012

    # Clip background glow so it never interferes with UI or trail readability
    np.clip(img, 0.0, 0.08, out=img)

    # 3. Dense Multi-Layer Starfield
    np.random.seed(42)

    # Tier 1: Faint background micro-stars
    n_micro = 900
    mx = np.random.randint(0, STAR_IMG_W, size=n_micro)
    my = np.random.randint(0, STAR_IMG_H, size=n_micro)
    mb = 0.15 + 0.35 * np.random.rand(n_micro)
    for i in range(n_micro):
        b = float(mb[i])
        img[mx[i], my[i]] += [b * 0.75, b * 0.85, b * 1.0]

    # Tier 2: Medium field stars
    n_med = 350
    sx = np.random.randint(0, STAR_IMG_W, size=n_med)
    sy = np.random.randint(0, STAR_IMG_H, size=n_med)
    sb = 0.40 + 0.40 * np.random.rand(n_med)
    hue = np.random.rand(n_med)
    for i in range(n_med):
        b = float(sb[i])
        x, y = sx[i], sy[i]
        # Color variation: white, cyan, yellow, warm tint
        if hue[i] < 0.6:
            col = np.array([b * 0.85, b * 0.90, b * 1.0], dtype=np.float32)
        elif hue[i] < 0.85:
            col = np.array([b * 1.0, b * 0.92, b * 0.75], dtype=np.float32)
        else:
            col = np.array([b * 1.0, b * 0.70, b * 0.80], dtype=np.float32)

        img[x, y] += col
        if b > 0.65 and x + 1 < STAR_IMG_W and y + 1 < STAR_IMG_H:
            img[x+1, y] += col * 0.35
            img[x, y+1] += col * 0.35

    # Tier 3: Bright foreground stars with soft halo glow
    n_bright = 45
    bx = np.random.randint(2, STAR_IMG_W - 2, size=n_bright)
    by = np.random.randint(2, STAR_IMG_H - 2, size=n_bright)
    bb = 0.75 + 0.25 * np.random.rand(n_bright)
    for i in range(n_bright):
        b = float(bb[i])
        x, y = bx[i], by[i]
        core_col = np.array([b, b, b], dtype=np.float32)
        halo_col = np.array([b * 0.4, b * 0.5, b * 0.7], dtype=np.float32)

        # Core
        img[x, y] += core_col
        # Cross flare & halo
        img[x-1, y] += halo_col * 0.5; img[x+1, y] += halo_col * 0.5
        img[x, y-1] += halo_col * 0.5; img[x, y+1] += halo_col * 0.5

    np.clip(img, 0.0, 1.0, out=img)
    star_img.from_numpy(img)



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

    # Directional Sphere Shading Colors for Planets (Enhanced Contrast)
    # Planet 1 (Inner - Cool Cyan #5ec8d8)
    planet_body_colors[0]     = [0.37, 0.78, 0.85]   # Base #5ec8d8
    planet_shadow_colors[0]   = [0.02, 0.09, 0.12]   # Deep limb shadow
    planet_lit_colors[0]      = [0.72, 0.94, 0.98]   # Lit highlight #b8f0fa
    planet_specular_colors[0] = [1.00, 1.00, 1.00]   # Crisp white specular peak

    # Planet 2 (Outer - Soft Coral #e08a6f)
    planet_body_colors[1]     = [0.88, 0.54, 0.435]  # Base #e08a6f
    planet_shadow_colors[1]   = [0.14, 0.05, 0.03]   # Deep limb shadow
    planet_lit_colors[1]      = [0.98, 0.78, 0.70]   # Lit highlight #fac7b3
    planet_specular_colors[1] = [1.00, 1.00, 1.00]   # Crisp white specular peak

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
        update_glow_positions(scale_x, scale_y, rogue_flag)
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

        # 1. Background (Near-black #05070c with low-density stars)
        canvas.set_image(star_img)

        # 2. Orbit & Flyby Trails (power-law fade to transparent at tail)
        canvas.lines(trail_line_vertices, width=0.0022, per_vertex_color=trail_line_colors)

        # 3. Host Star Radial Gradient & Visible Soft Bloom (~2.5x star radius)
        canvas.circles(host_star_pos, radius=0.035, color=(0.55, 0.40, 0.12))  # Outer soft bloom (~2.5x)
        canvas.circles(host_star_pos, radius=0.024, color=(0.82, 0.62, 0.18))  # Mid glow (~1.7x)
        canvas.circles(host_star_pos, radius=0.016, color=(0.95, 0.82, 0.35))  # Inner warm halo (~1.2x)

        # 4. Compact Stellar Intruder Glow & Core (if active)
        if rogue_enabled:
            canvas.circles(intruder_pos, radius=0.018, color=(0.22, 0.25, 0.38))
            canvas.circles(intruder_pos, radius=0.010, color=(0.85, 0.89, 1.00))

        # 5. Host Star Core Circle (bright white-gold center)
        canvas.circles(host_star_pos, radius=0.012, color=(0.98, 0.94, 0.82))

        # 6. Planets Directional Sphere Shading (~45% larger radii for camera zoom visibility)
        #    Layer A: Dark limb shadow base
        canvas.circles(planet_shadow_pos, radius=0.0130, per_vertex_color=planet_shadow_colors)
        #    Layer B: Main planet body circle
        canvas.circles(planet_center_pos, radius=0.0110, per_vertex_color=planet_body_colors)
        #    Layer C: Lit highlight circle (offset toward host star)
        canvas.circles(planet_lit_pos, radius=0.0070, per_vertex_color=planet_lit_colors)
        #    Layer D: Specular peak circle (offset further toward host star)
        canvas.circles(planet_specular_pos, radius=0.0035, per_vertex_color=planet_specular_colors)

        # =====================================================================
        # --- GUI Overlays ---
        # =====================================================================

        # Title bar — top center, compact
        gui.begin("##title", 0.28, 0.01, 0.44, 0.075)
        gui.text("  STELLAR DISRUPTION")
        gui.text("  Explore how a passing star reshapes planetary orbits")
        gui.end()

        # Left panel — Legend + Controls, narrow, shifted down below title
        gui.begin("##left", 0.01, 0.10, 0.22, 0.54)
        gui.text("LEGEND")
        gui.text("  * Host Star   Yellow  1.0 Msun")
        gui.text("  o Inner Planet  Cyan  1.0 AU")
        gui.text("  o Outer Planet Coral  1.8 AU")
        gui.text("  * Intruder      Blue  0.6 Msun")
        gui.text("")
        gui.text("CONTROLS")
        gui.text("  SPACE   Pause / Resume")
        gui.text("  S       Toggle Intruder")
        gui.text("  R       Reset")
        gui.text("  UP/DN   Speed  +/-")
        gui.text("")
        gui.text("SIMULATION")
        gui.text(f"  {'>> PAUSED <<' if paused else 'Running'}")
        gui.text(f"  Time  {sim_time:.2f} yr")
        gui.text(f"  Speed {speed_scale:.2f}x")
        gui.end()

        # Right panel — Scientific measurements, narrow
        gui.begin("##right", 0.77, 0.10, 0.22, 0.68)
        gui.text("ORBITAL ANALYSIS")
        gui.text("E = 0.5v^2 - GM/r")
        gui.text("")
        gui.text("INNER PLANET  (Cyan)")
        gui.text(f"  r   {r1:.3f} AU (ini {r0_1:.2f})")
        gui.text(f"  v   {v1:.3f} AU/yr")
        gui.text(f"  E   {E1:+.2f}")
        gui.text(f"  dE  {dE1:+.1f}%")
        gui.text(f"  {status1}")
        gui.text("")
        gui.text("OUTER PLANET  (Coral)")
        gui.text(f"  r   {r2:.3f} AU (ini {r0_2:.2f})")
        gui.text(f"  v   {v2:.3f} AU/yr")
        gui.text(f"  E   {E2:+.2f}")
        gui.text(f"  dE  {dE2:+.1f}%")
        gui.text(f"  {status2}")
        gui.text("")
        gui.text("INTRUDER ENCOUNTER")
        if rogue_enabled:
            gui.text(f"  Dist  {d_rogue:.3f} AU")
            if min_rogue_dist < 999.0:
                gui.text(f"  Min   {min_rogue_dist:.3f} AU")
            else:
                gui.text("  Min   ---")
        else:
            gui.text("  DISABLED")
        gui.text("")
        gui.text("ENERGY GUIDE")
        gui.text("  E<0  BOUND orbit")
        gui.text("  E>=0 POTENTIALLY UNBOUND")
        gui.text("  (3-body transfer)")
        gui.end()

        # Bottom status bar — thin strip across bottom
        gui.begin("##statusbar", 0.0, 0.93, 1.0, 0.07)
        rogue_str = f"Intruder: {'ACTIVE' if rogue_enabled else 'DISABLED'}"
        gui.text(f"  {rogue_str}   |   Time: {sim_time:.2f} yr   |   Phase: {phase_label}   |   Speed: {speed_scale:.2f}x")
        gui.end()

        window.show()


if __name__ == "__main__":
    main()
