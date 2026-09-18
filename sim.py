"""
STELLAR DISRUPTION — Phase 3: Scientific Impact Measurements

Physics Setup:
- Units: Astronomical Units (AU), Solar Masses (M_sun), Years (yr).
- Gravitational Constant G = 4 * pi^2 ~ 39.47841760435743 AU^3 / (M_sun * yr^2).
- Bodies:
  - Body 0: Fixed Host Star (1.0 M_sun) at origin (0, 0) AU.
  - Body 1: Inner Planet (~1 Earth Mass) at r1 = 1.0 AU, initial v1 = 2*pi AU/yr.
  - Body 2: Outer Planet (~3.3 Earth Masses) at r2 = 1.8 AU, initial v2 = sqrt(G * M_star / r2) AU/yr.
  - Body 3: Rogue Star (0.6 M_sun) at initial pos (-4.5, -2.0) AU, velocity (3.2, 1.1) AU/yr.
- Integrator: 2nd-order Symplectic Velocity Verlet scheme in Taichi kernel.
- Softening: eps^2 = 1e-6 AU^2 (prevents zero-division singularity).

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

MAX_TRAIL_LEN = 300                # Number of historical points per trail

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

# -----------------------------------------------------------------------------
# 3. Physics & Simulation Kernels
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
# 4. Rendering & Coordinate Mapping Kernels
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

                    # Fade color along trail length
                    alpha = float(seg) / float(MAX_TRAIL_LEN)
                    seg_color = color * (0.15 + 0.85 * alpha)
                    trail_line_colors[v_idx] = seg_color
                    trail_line_colors[v_idx + 1] = seg_color
            else:
                # Degenerate segments off-screen
                trail_line_vertices[v_idx] = ti.Vector([-1.0, -1.0])
                trail_line_vertices[v_idx + 1] = ti.Vector([-1.0, -1.0])
                trail_line_colors[v_idx] = ti.Vector([0.0, 0.0, 0.0])
                trail_line_colors[v_idx + 1] = ti.Vector([0.0, 0.0, 0.0])


# -----------------------------------------------------------------------------
# 5. Initialization Helper & Baseline Capture
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

    # Visual Properties
    # Body 0 (Host Star): Yellow, Radius 0.025
    body_colors[0] = [1.0, 0.85, 0.1]
    render_radius[0] = 0.025

    # Body 1 (Inner Planet): Cyan, Radius 0.012
    body_colors[1] = [0.1, 0.85, 1.0]
    render_radius[1] = 0.012

    # Body 2 (Outer Planet): Coral/Orange, Radius 0.014
    body_colors[2] = [1.0, 0.45, 0.2]
    render_radius[2] = 0.014

    # Body 3 (Rogue Star): Crimson Red, Radius 0.020
    body_colors[3] = [1.0, 0.1, 0.3]
    render_radius[3] = 0.020

    reset_to_initial()
    compute_accelerations(0)


# -----------------------------------------------------------------------------
# 6. Main Execution & Interactive Loop
# -----------------------------------------------------------------------------

def main():
    setup_initial_conditions()

    # Calculate baseline physical quantities for planets relative to Host Star
    m0 = 1.0
    r0_1 = 1.0
    v0_1 = math.sqrt(G_CONST * m0 / r0_1)
    E0_1 = 0.5 * (v0_1 ** 2) - (G_CONST * m0 / r0_1)

    r0_2 = 1.8
    v0_2 = math.sqrt(G_CONST * m0 / r0_2)
    E0_2 = 0.5 * (v0_2 ** 2) - (G_CONST * m0 / r0_2)

    min_rogue_dist = float('inf')

    # Create Taichi GGUI Window
    win_width, win_height = 1280, 800
    window = ti.ui.Window("STELLAR DISRUPTION — Phase 3: Scientific Impact Measurements",
                          res=(win_width, win_height),
                          vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    # Viewport scaling: 2.6 AU in Y axis; scale X to preserve 16:10 aspect ratio
    scale_y = 2.6
    scale_x = scale_y * (win_width / win_height)

    paused = False
    rogue_enabled = False
    speed_scale = 1.0
    sim_time = 0.0

    print("=" * 60, flush=True)
    print("STELLAR DISRUPTION — Phase 3 MVP Launched", flush=True)
    print("Controls:", flush=True)
    print("  [SPACE] : Pause / Resume simulation", flush=True)
    print("  [S]     : Toggle Rogue Star (ACTIVE / DISABLED)", flush=True)
    print("  [R]     : Reset system to initial unperturbed state", flush=True)
    print("  [UP]    : Speed up simulation", flush=True)
    print("  [DOWN]  : Slow down simulation", flush=True)
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

        # --- Physics Integration Step ---
        if not paused:
            dt = BASE_DT * speed_scale
            for _ in range(SUBSTEPS_PER_FRAME):
                verlet_step(dt, rogue_flag)
                sim_time += dt

            # Record trail history every 2 frames for smooth trail rendering
            trail_sample_counter += 1
            if trail_sample_counter % 2 == 0:
                record_trail_history(scale_x, scale_y, rogue_flag)

        # Update screen coordinates and trail line geometry
        update_render_positions(scale_x, scale_y, rogue_flag)
        build_trail_lines(rogue_flag)

        # --- Real-Time Scientific Impact Calculations ---
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
            status1 = "BOUND (E < 0)"

        # Planet 2 (Outer Planet - Index 2)
        r2 = math.sqrt(pos_arr[2][0]**2 + pos_arr[2][1]**2)
        v2 = math.sqrt(vel_arr[2][0]**2 + vel_arr[2][1]**2)
        E2 = 0.5 * (v2**2) - (G_CONST * m0 / max(r2, 1e-4))
        dE2 = ((E2 - E0_2) / abs(E0_2)) * 100.0
        
        if E2 >= 0.0:
            status2 = "POTENTIALLY UNBOUND" if encounter_in_progress else "UNBOUND / ESCAPED"
        else:
            status2 = "BOUND (E < 0)"

        # --- Drawing Canvas ---
        # Dark cosmic background
        canvas.set_background_color((0.05, 0.05, 0.08))

        # Render Orbit & Flyby Trails
        canvas.lines(trail_line_vertices, width=0.0025, per_vertex_color=trail_line_colors)

        # Render Bodies (Star, Planets & Rogue Star)
        canvas.circles(render_pos, radius=0.012, per_vertex_color=body_colors)

        # --- GUI Overlay 1: Simulation Controls & Status ---
        gui.begin("Simulation Controls & Status", 0.02, 0.02, 0.32, 0.32)
        gui.text("Phase 3: Scientific Impact MVP")
        gui.text("-----------------------------------")
        gui.text(f"Status       : {'[PAUSED]' if paused else '[RUNNING]'}")
        gui.text(f"Rogue Star   : {'[ACTIVE]' if rogue_enabled else '[DISABLED]'}")
        gui.text(f"Sim Time     : {sim_time:.2f} years")
        gui.text(f"Speed Multi  : {speed_scale:.2f}x")
        gui.text("")
        gui.text("Hotkeys:")
        gui.text("  SPACE  : Pause / Resume")
        gui.text("  S      : Toggle Rogue Star")
        gui.text("  R      : Reset System")
        gui.text("  UP/DN  : Speed +/-")
        gui.end()

        # --- GUI Overlay 2: Scientific Impact Measurements ---
        gui.begin("Scientific Impact Measurements", 0.60, 0.02, 0.38, 0.52)
        gui.text("Real-Time Gravitational Metrics")
        gui.text("-------------------------------------")
        gui.text("Planet 1 (Inner - Cyan):")
        gui.text(f"  Distance : {r1:.3f} AU  (Init: {r0_1:.2f} AU)")
        gui.text(f"  Speed    : {v1:.3f} AU/yr")
        gui.text(f"  Energy Δ : {dE1:+.1f}%")
        gui.text(f"  Status   : {status1}")
        gui.text("")
        gui.text("Planet 2 (Outer - Coral):")
        gui.text(f"  Distance : {r2:.3f} AU  (Init: {r0_2:.2f} AU)")
        gui.text(f"  Speed    : {v2:.3f} AU/yr")
        gui.text(f"  Energy Δ : {dE2:+.1f}%")
        gui.text(f"  Status   : {status2}")
        gui.text("")
        gui.text("Encounter Data:")
        if rogue_enabled:
            gui.text(f"  Rogue Star Dist: {d_rogue:.3f} AU")
            if min_rogue_dist < 999.0:
                gui.text(f"  Min Periastron : {min_rogue_dist:.3f} AU")
            else:
                gui.text("  Min Periastron : N/A")
            
            if encounter_in_progress:
                gui.text("  Encounter Phase: FLYBY IN PROGRESS")
            elif post_encounter:
                gui.text("  Encounter Phase: POST-ENCOUNTER (Final E)")
            else:
                gui.text("  Encounter Phase: APPROACHING")
        else:
            gui.text("  Rogue Star     : DISABLED")
        gui.end()

        window.show()


if __name__ == "__main__":
    main()
