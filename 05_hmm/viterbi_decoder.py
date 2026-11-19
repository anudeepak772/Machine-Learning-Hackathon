"""
PathSnap - Hidden Markov Model (HMM) & Viterbi Decoder
Performs global sequence-level map-matching using road-network topology,
classifier emission probabilities, and network shortest path transition probabilities.
"""

import math
import heapq
import numpy as np

class ViterbiMapMatchingDecoder:
    def __init__(self, sigma_z=8.0, beta=12.0, same_arc_prob=0.85, direct_conn_prob=1.0, gap_threshold_s=5.0):
        """
        sigma_z: GPS observation noise standard deviation (meters)
        beta: exponential decay parameter for network vs GPS distance differences
        same_arc_prob: prior probability for staying on the same arc
        direct_conn_prob: probability for moving along a directly connected arc
        gap_threshold_s: threshold for detecting large GPS dropouts
        """
        self.sigma_z = sigma_z
        self.beta = beta
        self.same_arc_prob = same_arc_prob
        self.direct_conn_prob = direct_conn_prob
        self.gap_threshold_s = gap_threshold_s
        self.min_log_prob = -50.0

    def _build_network_adjacency(self, arcs, nodes=None):
        """Builds forward and backward adjacency maps for arcs."""
        # node -> outgoing arc_ids
        node_outgoing = {}
        for arc_id, (u, v) in enumerate(arcs):
            if u not in node_outgoing:
                node_outgoing[u] = []
            node_outgoing[u].append((arc_id, v))
        return node_outgoing

    def _shortest_path_distance(self, from_arc, to_arc, arcs, node_outgoing, max_depth=8):
        """
        Computes hop distance or topological reachability between from_arc and to_arc.
        Returns:
            hops: number of hops (1 if directly connected, >1 for multi-hop, inf if unreachable)
        """
        if from_arc == to_arc:
            return 0

        u_from, v_from = arcs[from_arc]
        u_to, v_to = arcs[to_arc]

        # Direct connection
        if v_from == u_to:
            return 1
        # Reversed or parallel intersection connection
        if v_from == v_to or u_from == u_to:
            return 1.2

        # BFS for short path
        queue = [(v_from, 1)]
        visited = {v_from}
        
        while queue:
            curr_node, depth = queue.pop(0)
            if depth > max_depth:
                break
            for next_arc, next_node in node_outgoing.get(curr_node, []):
                if next_arc == to_arc:
                    return depth + 1
                if next_node not in visited and depth < max_depth:
                    visited.add(next_node)
                    queue.append((next_node, depth + 1))

        return float('inf')



    def compute_log_transition(self, from_arc, to_arc, arcs, node_outgoing, dt_s, d_gps_m):
        """
        Computes log transition probability from from_arc to to_arc given time delta dt_s.
        """
        if from_arc == to_arc:
            return math.log(self.same_arc_prob)

        hops = self._shortest_path_distance(from_arc, to_arc, arcs, node_outgoing, max_depth=6)

        if hops == 1:
            return math.log(self.direct_conn_prob)
        elif hops <= 3:
            # Multi-hop connection (e.g. vehicle traversed intermediate short arcs between 1s GPS points)
            return -1.5 * hops
        elif math.isinf(hops):
            # If large GPS gap occurred, be more tolerant of route jumps
            if dt_s > self.gap_threshold_s:
                return -12.0
            return self.min_log_prob
        else:
            return -2.5 * hops

    def decode(self, obs_candidates_list, arcs, nodes=None):
        """
        Viterbi log-space dynamic programming decoding.
        obs_candidates_list: list of candidate lists for each GPS observation.
            Each element is a list of dicts: {'arc_id': int, 'clf_prob': float, 'perp_dist': float, ...}
        arcs: list of (u, v) tuples for the trip road network
        Returns:
            best_path: list of arc_ids, one per GPS observation
        """
        T = len(obs_candidates_list)
        if T == 0:
            return []

        node_outgoing = self._build_network_adjacency(arcs, nodes)

        # V[t][i] stores max log probability of path ending at candidate i at step t
        # backpointer[t][i] stores index of previous candidate in V[t-1]
        V = []
        backpointer = []

        # Step 0: Initialize
        cands_0 = obs_candidates_list[0]
        v_0 = []
        for c in cands_0:
            log_emit = self.compute_log_emission(c['clf_prob'], c['perp_dist'])
            # Initial prior based on distance rank
            log_prior = -0.5 * (c['dist_rank'] - 1)
            v_0.append(log_prior + log_emit)
        V.append(v_0)
        backpointer.append([0] * len(cands_0))

        # Forward DP
        for t in range(1, T):
            prev_cands = obs_candidates_list[t - 1]
            curr_cands = obs_candidates_list[t]
            dt_s = curr_cands[0].get('time_delta', 1.0)
            d_gps_m = curr_cands[0].get('gps_speed', 10.0) * dt_s

            v_t = []
            bp_t = []

            for curr_idx, curr_c in enumerate(curr_cands):
                curr_arc = curr_c['arc_id']
                log_emit = self.compute_log_emission(curr_c['clf_prob'], curr_c['perp_dist'])

                best_prev_score = -float('inf')
                best_prev_idx = 0

                for prev_idx, prev_c in enumerate(prev_cands):
                    prev_arc = prev_c['arc_id']
                    log_trans = self.compute_log_transition(
                        prev_arc, curr_arc, arcs, node_outgoing, dt_s, d_gps_m
                    )
                    score = V[t - 1][prev_idx] + log_trans

                    if score > best_prev_score:
                        best_prev_score = score
                        best_prev_idx = prev_idx

                v_t.append(best_prev_score + log_emit)
                bp_t.append(best_prev_idx)

            V.append(v_t)
            backpointer.append(bp_t)

        # Backward tracking
        best_last_idx = int(np.argmax(V[-1]))
        best_path_indices = [best_last_idx]

        for t in range(T - 1, 0, -1):
            prev_idx = backpointer[t][best_path_indices[-1]]
            best_path_indices.append(prev_idx)

        best_path_indices.reverse()
        best_path_arcs = [obs_candidates_list[t][idx]['arc_id'] for t, idx in enumerate(best_path_indices)]

        return best_path_arcs
