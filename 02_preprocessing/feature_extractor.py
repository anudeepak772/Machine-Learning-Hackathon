"""
Feature Extractor for PathSnap Map-Matching
Extracts rich spatial, kinematic, and topological features for each (GPS observation, candidate arc) pair.
"""

import math
import numpy as np
import pandas as pd
try:
    from .spatial_indexer import SpatialGridIndex
except (ImportError, ValueError):
    from spatial_indexer import SpatialGridIndex

def calculate_bearing_deg(lon1, lat1, lon2, lat2):
    """Calculates forward compass bearing from point 1 to point 2 in degrees [0, 360)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0

def angular_difference_deg(angle1, angle2):
    """Returns absolute minimal angular difference in degrees [0, 180]."""
    diff = abs(angle1 - angle2) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff

def project_point_to_segment(px, py, ax, ay, bx, by, cos_lat):
    """
    Returns (perp_dist_m, dist_start_m, dist_end_m, t_ratio, seg_len_m).
    """
    px_m, py_m = px * 111320.0 * cos_lat, py * 111320.0
    ax_m, ay_m = ax * 111320.0 * cos_lat, ay * 111320.0
    bx_m, by_m = bx * 111320.0 * cos_lat, by * 111320.0

    dx, dy = bx_m - ax_m, by_m - ay_m
    seg_len_sq = dx*dx + dy*dy
    seg_len_m = math.sqrt(seg_len_sq)

    dist_start_m = math.hypot(px_m - ax_m, py_m - ay_m)
    dist_end_m = math.hypot(px_m - bx_m, py_m - by_m)

    if seg_len_sq == 0.0:
        return dist_start_m, dist_start_m, dist_end_m, 0.0, 0.0

    t_raw = ((px_m - ax_m)*dx + (py_m - ay_m)*dy) / seg_len_sq
    t = max(0.0, min(1.0, t_raw))

    proj_x = ax_m + t * dx
    proj_y = ay_m + t * dy
    perp_dist_m = math.hypot(px_m - proj_x, py_m - proj_y)

    return perp_dist_m, dist_start_m, dist_end_m, t, seg_len_m

def match_gps_to_route_ground_truth(track, nodes, arcs, route):
    """
    Robustly maps each GPS point in track to its true arc in ground truth route.
    Handles looping trajectories, repeated arcs, and stops without window desynchronization.
    Returns: list of true_arc_ids, one per GPS point.
    """
    if not route:
        return [None] * len(track)

    cos_lat = math.cos(math.radians(track[0][1]))
    true_arc_ids = []
    
    current_route_idx = 0
    num_route = len(route)
    
    for i, p in enumerate(track):
        px, py = p[0], p[1]
        
        # Adaptive forward-progress window [current - 5, current + 60]
        start_w = max(0, current_route_idx - 5)
        end_w = min(num_route, current_route_idx + 60)
        
        best_d = 1e9
        best_idx = current_route_idx
        
        for r_i in range(start_w, end_w):
            arc_id = route[r_i]
            u, v = arcs[arc_id]
            ax, ay = nodes[u]
            bx, by = nodes[v]
            d, _, _, _, _ = project_point_to_segment(px, py, ax, ay, bx, by, cos_lat)
            if d < best_d:
                best_d = d
                best_idx = r_i
                
        # Fallback to full route if the window drifted or GPS jump occurred
        if best_d > 45.0:
            for r_i in range(num_route):
                arc_id = route[r_i]
                u, v = arcs[arc_id]
                ax, ay = nodes[u]
                bx, by = nodes[v]
                d, _, _, _, _ = project_point_to_segment(px, py, ax, ay, bx, by, cos_lat)
                if d < best_d:
                    best_d = d
                    best_idx = r_i
                    
        current_route_idx = best_idx
        true_arc_ids.append(route[best_idx])
        
    return true_arc_ids

class FeatureExtractor:
    def __init__(self, search_radius_m=50.0, max_candidates=15):
        self.search_radius_m = search_radius_m
        self.max_candidates = max_candidates

    def extract_features_for_trip(self, trip_id, track, nodes, arcs, route=None):
        """
        Extracts candidate features and ground truth labels for all GPS points in a trip.
        Returns:
            features_df: DataFrame where each row is a candidate arc for a GPS point
            miss_count: int, count of GPS points where true arc was not in candidate set
            total_points: int, number of GPS points
        """
        # Build spatial index
        indexer = SpatialGridIndex(cell_size_deg=0.0008)
        indexer.fit(nodes, arcs)

        true_arcs = match_gps_to_route_ground_truth(track, nodes, arcs, route) if route else [None] * len(track)
        
        cos_lat = math.cos(math.radians(track[0][1]))
        num_points = len(track)
        
        # Precompute GPS kinematics (headings and speeds)
        gps_headings = []
        gps_speeds = []
        time_deltas = []
        
        for i in range(num_points):
            # Time delta
            dt = track[i][2] - track[i-1][2] if i > 0 else 1.0
            dt = dt if dt > 0 else 1.0
            time_deltas.append(dt)
            
            # Heading: use centered difference when possible
            if i > 0 and i < num_points - 1:
                p_prev, p_next = track[i-1], track[i+1]
                h = calculate_bearing_deg(p_prev[0], p_prev[1], p_next[0], p_next[1])
            elif i < num_points - 1:
                p_curr, p_next = track[i], track[i+1]
                h = calculate_bearing_deg(p_curr[0], p_curr[1], p_next[0], p_next[1])
            elif i > 0:
                p_prev, p_curr = track[i-1], track[i]
                h = calculate_bearing_deg(p_prev[0], p_prev[1], p_curr[0], p_curr[1])
            else:
                h = 0.0
            gps_headings.append(h)
            
            # Speed
            if i > 0:
                p_prev, p_curr = track[i-1], track[i]
                dx = (p_curr[0] - p_prev[0]) * 111320.0 * cos_lat
                dy = (p_curr[1] - p_prev[1]) * 111320.0
                dist_m = math.hypot(dx, dy)
                gps_speeds.append(dist_m / dt)
            else:
                gps_speeds.append(0.0)

        rows = []
        miss_count = 0

        for i in range(num_points):
            lon, lat, ts = track[i]
            true_arc = true_arcs[i]
            gps_head = gps_headings[i]
            gps_spd = gps_speeds[i]
            dt = time_deltas[i]

            # Query candidates
            cands = indexer.query_radius(lon, lat, radius_m=self.search_radius_m)
            
            # Adaptive fallback if radius had too few candidates
            if len(cands) < 3:
                cands = indexer.query_radius(lon, lat, radius_m=self.search_radius_m * 1.8)
            if len(cands) == 0:
                cands = indexer.query_radius(lon, lat, radius_m=self.search_radius_m * 3.0)

            # Check candidate miss rate
            if true_arc is not None:
                if true_arc not in cands:
                    miss_count += 1

            # Candidate feature computation
            cand_features = []
            for arc_id in cands:
                u, v = arcs[arc_id]
                u_lon, u_lat = nodes[u]
                v_lon, v_lat = nodes[v]

                perp_dist, dist_start, dist_end, t_ratio, arc_len = project_point_to_segment(
                    lon, lat, u_lon, u_lat, v_lon, v_lat, cos_lat
                )
                
                arc_head = calculate_bearing_deg(u_lon, u_lat, v_lon, v_lat)
                head_diff = angular_difference_deg(gps_head, arc_head)
                cos_head_diff = math.cos(math.radians(head_diff))
                
                # Gaussian spatial emission prior (sigma = 10m)
                spatial_prior = math.exp(-0.5 * (perp_dist / 10.0)**2)
                
                label = 1 if (true_arc is not None and arc_id == true_arc) else 0

                cand_features.append({
                    'trip_id': trip_id,
                    'obs_idx': i,
                    'timestamp': ts,
                    'arc_id': arc_id,
                    'perp_dist': perp_dist,
                    'dist_start': dist_start,
                    'dist_end': dist_end,
                    't_ratio': t_ratio,
                    'arc_len': arc_len,
                    'gps_speed': gps_spd,
                    'time_delta': dt,
                    'heading_diff': head_diff,
                    'cos_heading_diff': cos_head_diff,
                    'spatial_prior': spatial_prior,
                    'label': label
                })

            if not cand_features:
                continue

            # Sort by perp_dist to compute relative rank features
            cand_features.sort(key=lambda x: x['perp_dist'])
            
            # Keep top candidates per observation
            selected_cands = cand_features[:self.max_candidates]
            min_dist = selected_cands[0]['perp_dist']

            for rank, c in enumerate(selected_cands):
                c['dist_rank'] = rank + 1
                c['dist_ratio_to_best'] = c['perp_dist'] / (min_dist + 0.1)
                c['dist_diff_to_best'] = c['perp_dist'] - min_dist
                rows.append(c)

        df = pd.DataFrame(rows)
        return df, miss_count, num_points
