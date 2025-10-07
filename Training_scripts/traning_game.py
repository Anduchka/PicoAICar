import pygame, math, random
import numpy as np
import argparse, os, time

WIDTH, HEIGHT = 1000, 1000
FPS = 60
BG = (25, 28, 33)
CAR_COLOR = (40, 180, 255)
BEST_CAR_COLOR = (108, 245, 39)
RAY_COLOR = (240, 220, 120)
WALL_COLOR = (200, 200, 200)

# Car real to game stats 1px = 1sm
CAR_WIDTH = 20
CAR_HEIGHT = 20

TRACK_WIDTH = 9.8

RAY_MAX_DISTANCE = 200
RAY_SEQUENCE = [-90, -45, 0, 45, 90, 45, 0, -45]
OBS_ANGLES = [-90, -45, 0, 45, 90]
RAY_DELAY = 0.10 

CAR_MAX_SPEED = 40 # pixels per second
FRICTION = 1.0

CAR_COLOR = (40, 180, 255)

WALLS = []

# AI related properties

CAR_COUNT = 30
CMD_MIN_HOLD_FRAMES = 6
GRID_CELL = 40
OBSTICLE_COUNT = 30
CHECK_DISTANCE_TIME = 2.5
MIN_TRAVEL_DISTANCE = 25
target_for_length = 10

best_ever_score = 0.0

#network policy

OBS_DIM = len(OBS_ANGLES) + 2
HID = 16
ACT_DIM = 2

class Policy:
    
    def __init__(self):
        self.w1 = np.random.normal(0, 0.5, (HID, OBS_DIM)); self.b1 = np.zeros(HID)
        self.w2 = np.random.normal(0, 0.5, (ACT_DIM, HID)); self.b2 = np.zeros(ACT_DIM)
        
    def act(self, obs_list):
        x = np.tanh(self.w1 @ np.array(obs_list) + self.b1)
        y = np.tanh(self.w2 @ x + self.b2)
        return float(y[0]), float(y[1])
        
    def clone_mutate(self, sigma = 0.20):
        child = Policy()
        child.w1 = self.w1 + np.random.normal(0, sigma, self.w1.shape)
        child.b1 = self.b1 + np.random.normal(0, sigma, self.b1.shape)
        child.w2 = self.w2 + np.random.normal(0, sigma, self.w2.shape)
        child.b2 = self.b2 + np.random.normal(0, sigma, self.b2.shape)
        return child

def quantize_ternary(x, deadband=0.3):
    if x >  deadband: return  1
    if x < -deadband: return -1
    return 0

def seg_intersect(p1, p2, p3, p4): # check if intersect obj
    (x1,y1),(x2,y2),(x3,y3),(x4,y4) = p1, p2, p3, p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    
    if abs(denom) < 1e-9: return (False, None, None, None)
    
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t*(x2-x1);  iy = y1 + t*(y2-y1)
        return (True, t, u, (ix, iy))
    
    return (False, None, None, None)

def make_borders(): # create border for the map
    return [((10, 10), (WIDTH - 10, 10)),
        ((WIDTH - 10, 10), (WIDTH - 10, HEIGHT - 10)),
        ((WIDTH - 10, HEIGHT - 10), (10, HEIGHT - 10)),
        ((10, HEIGHT - 10), (10, 10))
        ]

def rot_rect_points(cx, cy, w, h, angle_rad):
    hw, hh = w/2, h/2
    corners = [(-hw,-hh), ( hw,-hh), ( hw, hh), (-hw, hh)]
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    pts = []
    for x,y in corners:
        rx = cx + x*c - y*s
        ry = cy + x*s + y*c
        pts.append((rx, ry))
        
    return pts

def _dist_point_to_seg(px, py, x1, y1, x2, y2): # get dist from point to obj
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    vv = vx*vx + vy*vy
    if vv == 0:
        dx, dy = px - x1, py - y1
        return (dx*dx + dy*dy) ** 0.5
    t = max(0.0, min(1.0, (wx*vx + wy*vy)/vv))
    projx, projy = x1 + t*vx, y1 + t*vy
    dx, dy = px - projx, py - projy
    return (dx*dx + dy*dy) ** 0.5

def pts_to_segments(pts): # transforms obj to line list
    lines = []
    for p in range(len(pts)):

        lines.append(((pts[p]), pts[(p + 1) % len(pts)]))
        
    return lines

def is_in_safe_radius(cx, cy, safe_radius, pts): # check if obj is in safe radius
    
    r2 = safe_radius*safe_radius
    
    for(x, y) in pts:
        if (x-cx)*(x-cx) + (y-cy)*(y-cy) <= r2:
            return True
        
    for (p1, p2) in pts_to_segments(pts):
        if _dist_point_to_seg(cx, cy, p1[0], p1[1], p2[0], p2[1]) <= safe_radius:
            return True
    
    return False 


def random_obsticles(n=6): # primary obsticle generator
    walls = make_borders()
    
    spawn_cx, spawn_cy = WIDTH/2, HEIGHT/2
    safe_radius = random.randint(45, 80)
    
    for _ in range(n):
        w = random.randint(60, 180)
        h = random.randint(40, 140)
        
        angle = math.radians(random.randint(0, 180))
        
        for _try in range(200):
            cx = random.randint(10 + int(w / 2), WIDTH - 10 - int(w / 2))
            cy = random.randint(10 + int(h / 2), HEIGHT - 10 - int(h / 2))
            
            pts = rot_rect_points(cx, cy, w, h, angle)
            
            if not is_in_safe_radius(spawn_cx, spawn_cy, safe_radius, pts):
                walls.extend(pts_to_segments(pts))
                break
        
    return walls

def generate_maze(cell=GRID_CELL): # basic maze for testing
    cols = (WIDTH - 20) // cell
    rows = (HEIGHT - 20) // cell
    
    cells = [[0 for _ in range(cols)] for _ in range(rows)]
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    history = []
    
    initial = [random.randint(0, cols - 1), random.randint(0, rows - 1)]
    
    history.append(initial)
    visited[initial[0]][initial[1]] = True
    
    while True:
        x, y = history[-1]
        
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(dirs)
        
        changed = False
        
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < cols and 0 <= ny < rows:
                if not visited[nx][ny]:
                    if dx == 1:
                        cells[nx][ny] = 1
                    elif dx == -1:
                        cells[x][y] = 3 if cells[x][y] == 2 else 1
                    elif dy == -1:
                        cells[nx][ny] = 2
                    elif dy == 1:
                        cells[x][y] = 3 if cells[x][y] == 1 else 2
                    
                    visited[nx][ny] = True 
                    changed = True
                    history.append([nx, ny])
                    break 
        
        if not changed:
            history.pop()
        if len(history) <= 0:
            break 
    
    walls = make_borders()
    
    for y in range(rows):
        for x in range(cols):
            if cells[x][y] == 1:
                walls.append(((x * cell, y * cell), (x * cell + cell, y * cell)))
            elif cells[x][y] == 2:
                walls.append(((x * cell, y * cell), (x * cell, y * cell + cell)))
    
    
    return walls

def _dist_to_room_border(x, y, ang): #check distance to room border from point
    dx, dy = math.cos(ang), math.sin(ang)
    ts = []
    if dx > 0:  ts.append((WIDTH - 15.0 - x) / dx)
    if dx < 0:  ts.append((15.0 - x) / dx)
    if dy > 0:  ts.append((HEIGHT - 15.0 - y) / dy)
    if dy < 0:  ts.append((15.0 - y) / dy)
    ts = [t for t in ts if t > 0]
    return min(ts) if ts else float('inf')

def _nearest_obstacle_hit(x, y, ang, obstacle_segments, max_len=10_000): #check obsticle hit
    start = (x, y)
    end   = (x + max_len*math.cos(ang), y + max_len*math.sin(ang))
    nearest = float('inf')
    for (p1, p2) in obstacle_segments:
        hit, t, _, _ = seg_intersect(start, end, p1, p2)
        if hit:
            dist = t * math.hypot(end[0]-x, end[1]-y)
            if 0 < dist < nearest:
                nearest = dist
    return nearest

def map_has_escape(x, y, obstacle_segments, n_rays=180): # check if map has escape from center to border
    for i in range(n_rays):
        ang = 2*math.pi * i / n_rays
        d_border = _dist_to_room_border(x, y, ang)
        d_obst   = _nearest_obstacle_hit(x, y, ang, obstacle_segments, max_len=2*max(WIDTH, HEIGHT))
        if d_border < d_obst - 1e-6:
            return True
    return False

class Car:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity_left = 0.0
        self.velocity_right = 0.0
        self.ray_dists = {a: RAY_MAX_DISTANCE for a in RAY_SEQUENCE}
        self.ray_idx = random.randrange(len(RAY_SEQUENCE))
        self.ray_timer = random.uniform(0, RAY_DELAY)
        self.dead = False
        self.policy = Policy()
        self.fitness = 0.0
        self.cmd_l = 0
        self.cmd_r = 0
        self.cmd_hold = 0
        self.mirror = bool(random.getrandbits(1))
        self.previous = [x, y]
        self._trail_time = 0.0
        self.just_entered_new_cell = False
        self.visited = set()
        self._mark_cell()
        
    def _mark_cell(self):
        cx = int(self.x // GRID_CELL)
        cy = int(self.y // GRID_CELL)
        
        k = (cx, cy)
        
        if k not in self.visited:
            self.visited.add(k)
            self.just_entered_new_cell = True
        else:
            self.just_entered_new_cell = False
        
        self.current_cell = (cx, cy)
        
    def set_tracks_speed(self, vl, vr):
        self.velocity_left = max(-CAR_MAX_SPEED, min(CAR_MAX_SPEED, vl))
        self.velocity_right = max(-CAR_MAX_SPEED, min(CAR_MAX_SPEED, vr))
    
    def update(self, delta_time):
        
        v = 0.5 * (self.velocity_left + self.velocity_right)
        omega = (self.velocity_right - self.velocity_left) / TRACK_WIDTH
        
        self.x += v * math.cos(self.theta) * delta_time
        self.y += v * math.sin(self.theta) * delta_time
        
        self.theta += omega * delta_time
    
        if self.theta > math.pi: self.theta -= 2*math.pi
        if self.theta < -math.pi: self.theta += 2*math.pi
        
        self.velocity_left *= FRICTION
        self.velocity_right *= FRICTION
        
        self._mark_cell()
        
    def get_shape(self):
        W = CAR_WIDTH
        L = CAR_HEIGHT
        
        corners = [(-L/2, -W/2), ( L/2, -W/2), ( L/2, W/2), (-L/2, W/2)]
        
        pts = []
        
        for (cx, cy) in corners:
            rx = self.x + cx * math.cos(self.theta) - cy * math.sin(self.theta)
            ry = self.y + cx * math.sin(self.theta) + cy * math.cos(self.theta)
            
            pts.append((rx, ry))
        
        return pts
    
    def ray_distance(self, walls, angle_deg, max_len=RAY_MAX_DISTANCE):
        
        ang = self.theta + math.radians(angle_deg)
        start = (self.x, self.y)
        end_guess = (self.x + max_len*math.cos(ang), self.y + max_len*math.sin(ang))
        hit_pt = None
        min_t = 1e9
        
        for (p1, p2) in walls:
            hit, t, u, pt = seg_intersect(start, end_guess, p1, p2)
            if hit and t < min_t:
                min_t, hit_pt = t, pt
            
        if hit_pt is None:
            d = max_len + random.uniform(-40, 0)
        else:
            dx = hit_pt[0] - self.x
            dy = hit_pt[1] - self.y
            d = math.hypot(dx, dy) + random.uniform(-10, 10)
            
        return max(0.0, min(RAY_MAX_DISTANCE, d))

def car_observation(car):
    dists = [max(0.0, min(1.0, car.ray_dists[a] / RAY_MAX_DISTANCE)) for a in OBS_ANGLES]
    
    if car.mirror:
        dists = dists[::-1]
        cmd_tail = [car.cmd_r, car.cmd_l]
    else:
        cmd_tail = [car.cmd_l, car.cmd_r]
    
    return dists + cmd_tail

def car_reward(car): # reward car
    
    reward = 0.0
    
    forvard_dist = car.ray_dists[0]
    direction = car.cmd_l + car.cmd_r
    
    left_space = (car.ray_dists[-90] + car.ray_dists[-45]) / 2
    right_space = (car.ray_dists[90] + car.ray_dists[45]) / 2
    
    if forvard_dist >= 15 and direction == 2:
        reward += 0.03
    elif forvard_dist < 10 and direction == 2:
        reward -= 0.01
    
    abstruction = False
    
    for a in (-90, -45, 45, 90):
        if car.ray_dists[a] < 10:
            reward -= 0.05
        elif car.ray_dists[a] < 25:
            abstruction = True
        if car.ray_dists[a] > 15 and car.ray_dists[a] < 40:
            reward += 0.05
    
    if not abstruction:
        if car.cmd_l == -car.cmd_r != 0:
            reward -= 0.05
        elif car.cmd_l == car.cmd_r == -1:
            reward -= 0.05
    
    if left_space > right_space: #turn left
        if car.cmd_r > car.cmd_l and abstruction:
            reward += 0.08
        elif car.cmd_r < car.cmd_l:
            reward -= 0.08
    
    elif right_space > left_space:#turn right
        if car.cmd_l > car.cmd_r and abstruction:
            reward += 0.08
        elif car.cmd_l < car.cmd_r:
            reward -= 0.08
    
    
    return reward

def car_collides(car_pts, walls): #check car colision
    edges = pts_to_segments(car_pts)
    for e1 in edges:
        for w in walls:
            hit, *_ = seg_intersect(e1[0], e1[1], w[0], w[1])
            if hit:
                return True
    return False

def reset_world(num): #reset world
    global WALLS
    
    if num % 2 == 0:
        WALLS = generate_maze(cell=80) #temp
    else:
        WALLS = random_obsticles(n=OBSTICLE_COUNT)
    
        while not map_has_escape(WIDTH / 2, HEIGHT / 2, WALLS, n_rays=180):
            WALLS = random_obsticles(n=OBSTICLE_COUNT)

def kill_car(car): #kill
    car.fitness -= 1.0
    car.dead = True
    car.velocity_left = 0.0
    car.velocity_right = 0.0

def create_car(): #spawn
    return Car(WIDTH / 2, HEIGHT / 2, random.uniform(-math.pi, math.pi))

def load_policy_npz(path):
    z = np.load(path)
    pol = Policy()
    pol.w1 = z["w1"]; pol.b1 = z["b1"]
    pol.w2 = z["w2"]; pol.b2 = z["b2"]
    return pol

def main(render=True, mode="train", load_path=None): #-----MAIN-----#
    
    pygame.init()

    screen = None
    clock = None

    if render:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()

    cars = []
    
    for _i in range(CAR_COUNT):
        cars.append(create_car())
    
    reset_world(1)
    
    global EPISODE_LEN
    
    if mode == "play":
        if load_path is None or not os.path.exists(load_path):
            print("ERROR: --play needs a valid --load <file.npz>")
            return
        best = load_policy_npz(load_path)
        for c in cars:
            c.policy = best
    else:
        EPISODE_LEN = 500
    
    frame_count = 0
    GEN = 0
    
    run = True
    
    global WALLS
    global OBSTICLE_COUNT
    
    best_car_now = cars[0]
    
    while run: # run loop
        
        if render:
            screen.fill((0,0,0))
            delta_time = clock.tick(FPS) / 1000.0
        else:
            delta_time = 1.0 / 720.0
    
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                run = False
        
        dead_cars = 0
        
        for car in cars: # advance all cars
            if car.dead:
                dead_cars += 1
                continue
            
            car.ray_timer += delta_time #car ray logic
        
            if car.ray_timer >= RAY_DELAY: 
                dist = car.ray_distance(WALLS, RAY_SEQUENCE[car.ray_idx])
            
                car.ray_dists[RAY_SEQUENCE[car.ray_idx]] = dist
                car.ray_idx = (car.ray_idx + 1) % len(RAY_SEQUENCE)
                car.ray_timer = 0.0
        
            obs = car_observation(car)
            
            #car move logic
            a_l, a_r = car.policy.act(obs)
            
            if car.mirror:
                a_l, a_r = a_r, a_l
            
            q_l = quantize_ternary(a_l, deadband=0.3)
            q_r = quantize_ternary(a_r, deadband=0.3)
            
            if car.cmd_hold > 0:
                q_l, q_r = car.cmd_l, car.cmd_r
                car.cmd_hold -= 1
            else:
                if (q_l != car.cmd_l) or (q_r != car.cmd_r):
                    car.cmd_l, car.cmd_r = q_l, q_r
                    car.cmd_hold = CMD_MIN_HOLD_FRAMES
            
            car.set_tracks_speed(car.cmd_l * CAR_MAX_SPEED, car.cmd_r * CAR_MAX_SPEED)
            car.update(delta_time)
            
            #car reward logic
            
            car._trail_time += delta_time
            
            if car._trail_time >= CHECK_DISTANCE_TIME:
                x, y = car.previous[0], car.previous[1]
                if (x-car.x)**2 + (y-car.y)**2 < MIN_TRAVEL_DISTANCE**2:
                    kill_car(car)
                else:
                    car.previous[0], car.previous[1] = car.x, car.y
                car._trail_time = 0
            
            reward = 0
            
            reward += car_reward(car)
            car.fitness += reward
        
            if car_collides(car.get_shape(), WALLS):
                kill_car(car)
        
            if render:
        
                # Draw car
                pts = car.get_shape()
                pygame.draw.polygon(screen, CAR_COLOR, pts, 0)
            
                # Draw rays
                for ang, dist in car.ray_dists.items():
                    ang_r = car.theta + math.radians(ang)
                    ex = car.x + dist * math.cos(ang_r)
                    ey = car.y + dist * math.sin(ang_r)
                    pygame.draw.line(screen, RAY_COLOR, (car.x, car.y), (ex, ey), 2)
                    pygame.draw.circle(screen, RAY_COLOR, (int(ex), int(ey)), 3)
                
                if best_car_now.fitness < car.fitness:
                    best_car_now = car
                pygame.draw.circle(screen, BEST_CAR_COLOR, (best_car_now.x, best_car_now.y), 8)
            
        frame_count += 1
        
        if frame_count >= EPISODE_LEN or dead_cars >= CAR_COUNT: #new episode logic
            
            if mode == "play":
                cars = reset_world()
                if load_path:
                    best = load_policy_npz(load_path)
                    for c in cars:
                        c.policy = best
                frame_count = 0
                continue    
            
            GEN += 1
            
            cars.sort(key=lambda c: c.fitness, reverse=True)
            elites = cars[: max(2, CAR_COUNT // 5)]
            best = elites[0].fitness
            avg  = sum(c.fitness for c in cars) / len(cars)
            print(f"Gen {GEN:03d}  best={best:.2f}  avg={avg:.2f}")
            
            global target_for_length, best_ever_score
            
            if best > target_for_length:
                EPISODE_LEN = int(EPISODE_LEN * 1.1)
                target_for_length *= 1.10
            
            policy_name = "best_policy"
            
            if best > best_ever_score:
                best_ever_score = best
                policy_name = "new_best_policy"
            np.savez(policy_name + "(b="+ str(int(best)) +"_a="+ str(int(avg)) +")_gen" + str(GEN) +".npz",
                w1=elites[0].policy.w1, b1=elites[0].policy.b1,
                w2=elites[0].policy.w2, b2=elites[0].policy.b2)
            
            new_cars = []
            for e in elites:
                nc = create_car()
                nc.policy = e.policy.clone_mutate(sigma=0.0)
                new_cars.append(nc)
            
            while len(new_cars) < CAR_COUNT:
                parent = random.choice(elites)
                child = create_car()
                child.policy = parent.policy.clone_mutate(sigma=0.15)
                new_cars.append(child)
            
            reset_world(GEN)
            
            cars = new_cars
            best_car_now = cars[0]
            frame_count = 0
        
        if render:
            # Draw wals
            for (p1, p2) in WALLS:
                pygame.draw.line(screen, WALL_COLOR, p1, p2, 2)
        
            pygame.display.flip()

    pygame.quit()
    
#py traning_game.py --render --play --cars 10 --load 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--render", action="store_true", help="show pygame window")
    p.add_argument("--play", action="store_true", help="playback a saved policy (no training)")
    p.add_argument("--load", type=str, default=None, help="path to .npz policy to load")
    p.add_argument("--cars", type=int, default=CAR_COUNT, help="number of cars")
    args = p.parse_args()
    
    CAR_COUNT = args.cars
    
    mode = "play" if args.play else "train"
    
    if mode == "play":
        OBSTICLE_COUNT = 30
        EPISODE_LEN = 90000
        MAX_VIOLATION_COUNT = 1000
    
    main(render=args.render, mode=mode, load_path=args.load)