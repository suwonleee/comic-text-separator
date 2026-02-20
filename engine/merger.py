# GPL-3.0
import itertools
from typing import List, Set
from collections import Counter

import numpy as np
import networkx as nx
from shapely.geometry import Polygon

from .types import TextLine, TextRegion
from .geometry import can_merge_textlines


def _split_text_region(bboxes, connected_indices, width, height, gamma=0.5, sigma=2):
    connected_indices = list(connected_indices)
    if len(connected_indices) == 1:
        return [set(connected_indices)]

    if len(connected_indices) == 2:
        fs = max(bboxes[connected_indices[0]].font_size, bboxes[connected_indices[1]].font_size)
        if (bboxes[connected_indices[0]].distance(bboxes[connected_indices[1]]) < (1 + gamma) * fs
                and abs(bboxes[connected_indices[0]].angle - bboxes[connected_indices[1]].angle) < 0.2 * np.pi):
            return [set(connected_indices)]
        return [{connected_indices[0]}, {connected_indices[1]}]

    G = nx.Graph()
    for idx in connected_indices:
        G.add_node(idx)
    for u, v in itertools.combinations(connected_indices, 2):
        G.add_edge(u, v, weight=bboxes[u].distance(bboxes[v]))

    edges = sorted(nx.algorithms.tree.minimum_spanning_edges(G, algorithm='kruskal', data=True),
                   key=lambda a: a[2]['weight'], reverse=True)
    distances_sorted = [a[2]['weight'] for a in edges]
    fontsize = np.mean([bboxes[idx].font_size for idx in connected_indices])
    distances_std = np.std(distances_sorted)
    distances_mean = np.mean(distances_sorted)
    std_threshold = max(0.3 * fontsize + 5, 5)

    b1, b2 = bboxes[edges[0][0]], bboxes[edges[0][1]]
    max_poly_distance = Polygon(b1.pts).distance(Polygon(b2.pts))
    max_centroid_alignment = min(abs(b1.centroid[0] - b2.centroid[0]), abs(b1.centroid[1] - b2.centroid[1]))

    if ((distances_sorted[0] <= distances_mean + distances_std * sigma
         or distances_sorted[0] <= fontsize * (1 + gamma))
            and (distances_std < std_threshold
                 or (max_poly_distance == 0 and max_centroid_alignment < 5))):
        return [set(connected_indices)]

    G2 = nx.Graph()
    for idx in connected_indices:
        G2.add_node(idx)
    for edge in edges[1:]:
        G2.add_edge(edge[0], edge[1])
    result = []
    for node_set in nx.algorithms.components.connected_components(G2):
        result.extend(_split_text_region(bboxes, node_set, width, height))
    return result


def _merge_bboxes(bboxes: List[TextLine], width, height):
    G = nx.Graph()
    for i, box in enumerate(bboxes):
        G.add_node(i, box=box)
    for (u, ubox), (v, vbox) in itertools.combinations(enumerate(bboxes), 2):
        if can_merge_textlines(ubox, vbox, aspect_ratio_tol=1.3, font_size_ratio_tol=2,
                               char_gap_tolerance=1, char_gap_tolerance2=3):
            G.add_edge(u, v)

    region_indices: List[Set[int]] = []
    for node_set in nx.algorithms.components.connected_components(G):
        region_indices.extend(_split_text_region(bboxes, node_set, width, height))

    for node_set in region_indices:
        nodes = list(node_set)
        txtlns = np.array(bboxes)[nodes]

        fg_r = round(np.mean([box.fg_r for box in txtlns]))
        fg_g = round(np.mean([box.fg_g for box in txtlns]))
        fg_b = round(np.mean([box.fg_b for box in txtlns]))
        bg_r = round(np.mean([box.bg_r for box in txtlns]))
        bg_g = round(np.mean([box.bg_g for box in txtlns]))
        bg_b = round(np.mean([box.bg_b for box in txtlns]))

        dirs = [box.direction for box in txtlns]
        top2 = Counter(dirs).most_common(2)
        if len(top2) == 1:
            majority_dir = top2[0][0]
        elif top2[0][1] == top2[1][1]:
            max_ar = -100
            majority_dir = top2[0][0]
            for box in txtlns:
                if box.aspect_ratio > max_ar:
                    max_ar = box.aspect_ratio
                    majority_dir = box.direction
                if 1.0 / box.aspect_ratio > max_ar:
                    max_ar = 1.0 / box.aspect_ratio
                    majority_dir = box.direction
        else:
            majority_dir = top2[0][0]

        if majority_dir == 'h':
            nodes = sorted(nodes, key=lambda x: bboxes[x].centroid[1])
        elif majority_dir == 'v':
            nodes = sorted(nodes, key=lambda x: -bboxes[x].centroid[0])
        txtlns = np.array(bboxes)[nodes]
        yield txtlns, (fg_r, fg_g, fg_b), (bg_r, bg_g, bg_b)


async def merge_to_blocks(textlines: List[TextLine], width: int, height: int,
                          verbose: bool = False) -> List[TextRegion]:
    text_regions = []
    for txtlns, fg_color, bg_color in _merge_bboxes(textlines, width, height):
        total_logprobs = 0
        for txtln in txtlns:
            total_logprobs += np.log(txtln.prob) * txtln.area
        total_logprobs /= sum([txtln.area for txtln in textlines])

        font_size = int(min([txtln.font_size for txtln in txtlns]))
        angle = np.rad2deg(np.mean([txtln.angle for txtln in txtlns])) - 90
        if abs(angle) < 3:
            angle = 0
        lines = [txtln.pts for txtln in txtlns]
        texts = [txtln.text for txtln in txtlns]
        region = TextRegion(lines, texts, font_size=font_size, angle=angle,
                            prob=np.exp(total_logprobs), fg_color=fg_color, bg_color=bg_color)
        text_regions.append(region)
    return text_regions
