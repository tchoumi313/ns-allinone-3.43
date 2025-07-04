#!/usr/bin/env python3
"""
Simple test to debug cluster member visibility
"""

import random

import matplotlib.pyplot as plt
import numpy as np
from arpmec_faithful import ARPMECProtocol, Node, NodeState


def create_test_network(N: int = 10, area_size: int = 400):
    """Create a smaller test network"""
    nodes = []
    random.seed(42)  # Fixed seed for reproducibility
    for i in range(N):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        energy = random.uniform(90, 110)
        nodes.append(Node(i, x, y, energy))
    return nodes

def debug_clustering():
    """Debug cluster formation and member assignment"""
    print("=== DEBUGGING CLUSTER MEMBER VISIBILITY ===")
    
    # Create small network
    nodes = create_test_network(10, 400)
    protocol = ARPMECProtocol(nodes, C=4, R=5, K=2)
    
    print(f"\nInitial Network:")
    for node in nodes:
        print(f"  Node-{node.id}: ({node.x:.0f}, {node.y:.0f}) energy={node.energy:.1f}")
    
    # Run clustering
    print(f"\nRunning clustering...")
    clusters = protocol.clustering_algorithm()
    
    print(f"\nAfter clustering:")
    print(f"  Clusters formed: {len(clusters)}")
    
    # Analyze results
    cluster_heads = protocol._get_cluster_heads()
    all_nodes = list(protocol.nodes.values())
    
    print(f"\n=== DETAILED ANALYSIS ===")
    print(f"Total nodes: {len(all_nodes)}")
    print(f"Cluster heads: {len(cluster_heads)}")
    
    cluster_members = []
    idle_nodes = []
    
    for node in all_nodes:
        print(f"\nNode-{node.id} at ({node.x:.0f}, {node.y:.0f}):")
        print(f"  State: {node.state}")
        print(f"  Cluster ID: {node.cluster_id}")
        print(f"  Cluster Head ID: {node.cluster_head_id}")
        print(f"  Is alive: {node.is_alive()}")
        
        if node.state == NodeState.CLUSTER_HEAD:
            print(f"  -> CLUSTER HEAD with {len(node.cluster_members)} members: {node.cluster_members}")
        elif node.state == NodeState.CLUSTER_MEMBER:
            cluster_members.append(node)
            ch = protocol.nodes.get(node.cluster_head_id)
            if ch:
                distance = node.distance_to(ch)
                print(f"  -> MEMBER of CH-{node.cluster_head_id} (distance: {distance:.1f}m)")
            else:
                print(f"  -> MEMBER but CH-{node.cluster_head_id} not found!")
        elif node.state == NodeState.IDLE:
            idle_nodes.append(node)
            print(f"  -> IDLE NODE")
    
    print(f"\n=== SUMMARY ===")
    print(f"Cluster heads: {len(cluster_heads)} - IDs: {[ch.id for ch in cluster_heads]}")
    print(f"Cluster members: {len(cluster_members)} - IDs: {[m.id for m in cluster_members]}")
    print(f"Idle nodes: {len(idle_nodes)} - IDs: {[n.id for n in idle_nodes]}")
    
    # Check specific cluster head assignments
    for ch in cluster_heads:
        members_in_ch = [m for m in cluster_members if m.cluster_head_id == ch.id]
        print(f"\nCH-{ch.id} analysis:")
        print(f"  CH.cluster_members list: {ch.cluster_members}")
        print(f"  Actual members found: {[m.id for m in members_in_ch]}")
        print(f"  Match: {sorted(ch.cluster_members) == sorted([m.id for m in members_in_ch])}")
        
        # Check distances
        for m in members_in_ch:
            distance = m.distance_to(ch)
            in_range = distance <= protocol.communication_range
            print(f"    Member-{m.id}: {distance:.1f}m (in range: {in_range})")
    
    return protocol, cluster_heads, cluster_members, idle_nodes

if __name__ == "__main__":
    debug_clustering()
