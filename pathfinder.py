screen_width = 640
screen_height = 480

screen_size = (screen_width, screen_height)

player_position = (screen_width // 2, screen_height)

def is_counter_clockwise(pnt1, pnt2, pnt3):
    return (pnt3[1] - pnt1[1]) * (pnt2[0] - pnt1[0]) > (pnt2[1] - pnt1[1]) * (pnt3[0] - pnt1[0])

def segments_intersect(start1, end1, start2, end2):
    return (is_counter_clockwise(start1, start2, end2) != is_counter_clockwise(end1, start2, end2)) and \
           (is_counter_clockwise(start1, end1, start2) != is_counter_clockwise(start1, end1, end2))

def detect_obstacle_collision(target_loc, player_loc, obs_x1, obs_y1, obs_x2, obs_y2):
    obs_center = ((obs_x1 + obs_x2) // 2, (obs_y1 + obs_y2) // 2)
    corners = [(obs_x1, obs_y1), (obs_x2, obs_y1), (obs_x1, obs_y2), (obs_x2, obs_y2)]
    
    for corner in corners:
        if segments_intersect(player_loc, target_loc, obs_center, corner):
            return "Left" if corner[0] < obs_center[0] else "Right"
    return None

def determine_screen_region(x_coord):
    if x_coord <= screen_width // 3:
        return "left"
    elif x_coord <= 2 * (screen_width // 3):
        return "center"
    else:
        return "right"

def compute_navigation_path(entities, target_label):
    barriers = []
    goal_position = (-1, -1)
    goal_depth = -1
    found = False
    
    for entity in entities:
        label, x_start, y_start, x_end, y_end, depth = entity
        if label == target_label:
            goal_position = ((x_start + x_end) // 2, (y_start + y_end) // 2)
            goal_depth = depth
            found = True
        else:
            barriers.append((label, x_start, y_start, x_end, y_end, depth))
    
    if not found:
        return ("Target not visible", -1, False)
    
    region = determine_screen_region(goal_position[0])
    
    barriers.sort(key=lambda item: item[5], reverse=True)
    
    for barrier in barriers:
        if barrier[5] < goal_depth:
            break
        result = detect_obstacle_collision(goal_position, player_position, barrier[1], barrier[2], barrier[3], barrier[4])
        
        if result == "Left" and region == "center":
            return (f"Obstacle {barrier[0]} ahead, turn Left and proceed.", region, True)
        elif result == "Right" and region == "center":
            return (f"Obstacle {barrier[0]} ahead, turn Right and proceed.", region, True)
        elif result and region in ["left", "right"]:
            return ("Move forward.", region, False)
    
    if region == "left":
        return ("Turn Left", region, False)
    elif region == "right":
        return ("Turn Right", region, False)
    
    return ("Move forward.", region, False)
