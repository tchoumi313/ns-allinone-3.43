#!/usr/bin/env python3
"""
ARPMEC Protocol Demonstration Script

This script demonstrates the implementation of the ARPMEC (Adaptive Mobile Edge 
Computing-based Routing Protocol) algorithms described in the research paper:

"ARPMEC: an adaptive mobile edge computing-based routing protocol for IoT networks"
by Miguel Landry Foko Sindjoung, Mthulisi Velempini, and Vianney Kengne Tchendji (2024)

The implementation includes:
1. Link Quality Estimation using RSSI and PDR
2. Clustering Algorithm (Algorithm 2 from paper)
3. Adaptive Routing Algorithm (Algorithm 3 from paper)
4. Energy consumption model
5. Performance evaluation framework
"""

import random
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
# Import our ARPMEC implementation
from arpmec_implementation import (ARPMECProtocol, Node, NodeState,
                                   create_random_network)


def visualize_network(nodes: List[Node], clusters: Dict[int, List[int]], 
                     title: str = "ARPMEC Network Topology"):
    """Visualize the network topology with clusters"""
    
    plt.figure(figsize=(12, 10))
    
    # Define colors for different clusters
    colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
    
    # Plot cluster members
    for i, (head_id, members) in enumerate(clusters.items()):
        color = colors[i]
        
        # Plot cluster head
        head = next(n for n in nodes if n.id == head_id)
        plt.scatter(head.x, head.y, c=[color], s=200, marker='^', 
                   label=f'CH {head_id}', edgecolors='black', linewidth=2)
        
        # Plot cluster members
        for member_id in members:
            member = next(n for n in nodes if n.id == member_id)
            plt.scatter(member.x, member.y, c=[color], s=100, marker='o', 
                       alpha=0.7, edgecolors='gray')
            
            # Draw line from member to cluster head
            plt.plot([member.x, head.x], [member.y, head.y], 
                    color=color, alpha=0.3, linewidth=1)
    
    # Plot isolated nodes (not in any cluster)
    clustered_nodes = set()
    for head_id, members in clusters.items():
        clustered_nodes.add(head_id)
        clustered_nodes.update(members)
    
    for node in nodes:
        if node.id not in clustered_nodes:
            plt.scatter(node.x, node.y, c='red', s=50, marker='x', 
                       label='Isolated' if 'Isolated' not in [t.get_text() for t in plt.gca().get_legend().get_texts() if plt.gca().get_legend()] else "")
    
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def demonstrate_link_quality_estimation():
    """Demonstrate the link quality estimation mechanism"""
    print("\n" + "="*60)
    print("LINK QUALITY ESTIMATION DEMONSTRATION")
    print("="*60)
    
    # Create two nodes for demonstration
    node1 = Node(1, 100, 100)
    node2 = Node(2, 200, 150)
    
    distance = node1.distance_to(node2)
    print(f"Distance between nodes: {distance:.2f} m")
    
    # Simulate RSSI calculation
    protocol = ARPMECProtocol([node1, node2])
    rssi = protocol.simulate_rssi(distance)
    print(f"Simulated RSSI: {rssi:.2f} dBm")
    
    # Simulate PDR (assuming some packet loss)
    packets_sent = 100
    packets_received = random.randint(80, 100)
    pdr = packets_received / packets_sent
    print(f"Packet Delivery Ratio: {pdr:.2f}")
    
    # Predict link quality
    prediction = protocol.predict_link_quality(rssi, pdr)
    print(f"Link Quality Prediction Score: {prediction:.3f}")
    
    return prediction

def demonstrate_clustering():
    """Demonstrate the clustering algorithm"""
    print("\n" + "="*60)
    print("CLUSTERING ALGORITHM DEMONSTRATION")
    print("="*60)
    
    # Create a small network for demonstration
    num_nodes = 20
    nodes = create_random_network(num_nodes, area_size=500)
    
    print(f"Created network with {num_nodes} nodes")
    print(f"Initial energy per node: {nodes[0].initial_energy:.2f} J")
    
    # Initialize ARPMEC protocol
    arpmec = ARPMECProtocol(nodes, num_channels=4, hello_messages=25)
    
    # Run clustering algorithm
    print("\nRunning clustering algorithm...")
    start_time = time.time()
    clusters = arpmec.clustering_algorithm()
    clustering_time = time.time() - start_time
    
    print(f"Clustering completed in {clustering_time:.3f} seconds")
    print(f"Number of clusters formed: {len(clusters)}")
    
    # Display cluster information
    for head_id, members in clusters.items():
        head = arpmec.nodes[head_id]
        print(f"Cluster {head_id}: Head energy = {head.energy:.2f}J, "
              f"Members = {len(members)}, Members: {members}")
    
    # Calculate clustering energy consumption
    total_energy_consumed = sum(
        node.initial_energy - node.energy for node in nodes
    )
    print(f"Total energy consumed during clustering: {total_energy_consumed:.2f} J")
    
    return arpmec, clusters

def demonstrate_adaptive_routing():
    """Demonstrate the adaptive routing algorithm"""
    print("\n" + "="*60)
    print("ADAPTIVE ROUTING DEMONSTRATION")
    print("="*60)
    
    # Use the network from clustering demonstration
    arpmec, clusters = demonstrate_clustering()
    
    initial_energy = sum(node.energy for node in arpmec.nodes.values())
    initial_alive = sum(1 for node in arpmec.nodes.values() if node.is_alive())
    
    print(f"Starting routing with {initial_alive} alive nodes")
    print(f"Total remaining energy: {initial_energy:.2f} J")
    
    # Run adaptive routing for fewer rounds for demonstration
    rounds = 10
    print(f"\nRunning adaptive routing for {rounds} rounds...")
    
    start_time = time.time()
    arpmec.adaptive_routing_algorithm(rounds)
    routing_time = time.time() - start_time
    
    # Final statistics
    final_energy = sum(node.energy for node in arpmec.nodes.values())
    final_alive = sum(1 for node in arpmec.nodes.values() if node.is_alive())
    routing_energy = initial_energy - final_energy
    
    print(f"Routing completed in {routing_time:.3f} seconds")
    print(f"Energy consumed during routing: {routing_energy:.2f} J")
    print(f"Nodes alive after routing: {final_alive}/{len(arpmec.nodes)}")
    print(f"Network lifetime: {final_alive/len(arpmec.nodes)*100:.1f}%")
    
    return arpmec

def demonstrate_energy_model():
    """Demonstrate the energy consumption model"""
    print("\n" + "="*60)
    print("ENERGY MODEL DEMONSTRATION")
    print("="*60)
    
    node = Node(1, 0, 0, initial_energy=100.0)
    
    print(f"Node initial energy: {node.energy:.2f} J")
    print(f"Energy parameters:")
    print(f"  - Transmission energy (et): {node.et} J")
    print(f"  - Reception energy (er): {node.er} J") 
    print(f"  - Amplification energy (eamp): {node.eamp} J")
    
    # Simulate different transmission scenarios
    scenarios = [
        (1, 100, "Short range communication"),
        (5, 500, "Medium range communication"),
        (10, 1000, "Long range communication")
    ]
    
    print("\nEnergy consumption for different scenarios:")
    for num_items, distance, description in scenarios:
        energy_cost = node.calculate_energy_consumption(num_items, distance)
        print(f"  {description}: {energy_cost:.3f} J "
              f"({num_items} items, {distance}m distance)")

def run_complete_demonstration():
    """Run a complete demonstration of the ARPMEC protocol"""
    print("ARPMEC PROTOCOL IMPLEMENTATION DEMONSTRATION")
    print("=" * 80)
    print(__doc__)
    
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Demonstrate each component
    demonstrate_link_quality_estimation()
    demonstrate_energy_model()
    arpmec = demonstrate_adaptive_routing()
    
    # Generate network visualization
    print("\n" + "="*60)
    print("NETWORK VISUALIZATION")
    print("="*60)
    
    try:
        fig = visualize_network(
            list(arpmec.nodes.values()), 
            arpmec.clusters,
            "ARPMEC Network After Clustering and Routing"
        )
        plt.savefig('arpmec_network_demo.png', dpi=300, bbox_inches='tight')
        print("Network topology saved as 'arpmec_network_demo.png'")
        plt.show()
    except Exception as e:
        print(f"Visualization not available: {e}")
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    stats = arpmec.get_network_statistics()
    print(f"Final network statistics:")
    print(f"  - Total nodes: {stats['total_nodes']}")
    print(f"  - Alive nodes: {stats['alive_nodes']}")
    print(f"  - Network lifetime: {stats['alive_nodes']/stats['total_nodes']*100:.1f}%")
    print(f"  - Active clusters: {stats['num_clusters']}")
    print(f"  - Average energy remaining: {stats['avg_energy_remaining']:.2f} J")
    print(f"  - Total energy consumed: {stats['total_energy_consumed']:.2f} J")
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return arpmec

def parameter_sensitivity_analysis():
    """Analyze sensitivity to key parameters"""
    print("\n" + "="*60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*60)
    
    base_params = {
        'num_nodes': 30,
        'num_channels': 8,
        'hello_messages': 50,
        'area_size': 500
    }
    
    print(f"Base parameters: {base_params}")
    
    # Test different numbers of HELLO messages
    hello_msg_options = [25, 50, 75, 100]
    print(f"\nTesting HELLO message sensitivity:")
    
    for hello_msgs in hello_msg_options:
        nodes = create_random_network(base_params['num_nodes'], base_params['area_size'])
        arpmec = ARPMECProtocol(nodes, base_params['num_channels'], hello_msgs)
        
        initial_energy = sum(node.energy for node in nodes)
        clusters = arpmec.clustering_algorithm()
        clustering_energy = initial_energy - sum(node.energy for node in nodes)
        
        print(f"  R={hello_msgs}: {len(clusters)} clusters, "
              f"{clustering_energy:.2f}J consumed")

if __name__ == "__main__":
    # Run the complete demonstration
    arpmec_instance = run_complete_demonstration()
    
    # Run parameter analysis
    parameter_sensitivity_analysis()
    
    print(f"\nFor more advanced simulations, run:")
    print(f"python arpmec_simulation.py")