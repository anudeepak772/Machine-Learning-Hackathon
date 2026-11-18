"""
Spatial Indexer for Road Networks
Enables sub-millisecond candidate arc retrieval for GPS points.
"""

import math

class SpatialGridIndex:
    def __init__(self, cell_size_deg=0.0008):
        """
        cell_size_deg: ~70m (lon) x ~90m (lat) in Central Germany (lat ~49.8).
        Provides tight candidate bounding boxes and fast lookups.
        """
        self.cell_size = cell_size_deg
        self.grid = {}
        self.nodes = None
        self.arcs = None

    def fit(self, nodes, arcs):
        """
        nodes: list of (lon, lat) tuples
        arcs: list of (from_node, to_node) tuples
        """
        self.nodes = nodes
        self.arcs = arcs
        self.grid.clear()
        
        cs = self.cell_size
        for arc_id, (u, v) in enumerate(arcs):
            u_lon, u_lat = nodes[u]
            v_lon, v_lat = nodes[v]
            min_x = u_lon if u_lon < v_lon else v_lon
            max_x = u_lon if u_lon > v_lon else v_lon
            min_y = u_lat if u_lat < v_lat else v_lat
            max_y = u_lat if u_lat > v_lat else v_lat

            gx0, gx1 = int(min_x / cs), int(max_x / cs)
            gy0, gy1 = int(min_y / cs), int(max_y / cs)

            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    key = (gx, gy)
                    if key not in self.grid:
                        self.grid[key] = []
                    self.grid[key].append(arc_id)

    def query_radius(self, lon, lat, radius_m=50.0):
        """
        Returns a list of candidate arc_ids within search radius in meters.
        """
        cos_lat = math.cos(math.radians(lat))
        # Convert meters to degrees
        d_lat_deg = radius_m / 111320.0
        d_lon_deg = radius_m / (111320.0 * cos_lat)

        cs = self.cell_size
        gx0 = int((lon - d_lon_deg) / cs)
        gx1 = int((lon + d_lon_deg) / cs)
        gy0 = int((lat - d_lat_deg) / cs)
        gy1 = int((lat + d_lat_deg) / cs)

        candidates = set()
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                key = (gx, gy)
                if key in self.grid:
                    candidates.update(self.grid[key])
        
        return list(candidates)
