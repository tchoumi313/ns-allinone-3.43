import heapq
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class NodeState(Enum):
    IDLE = "idle"
    CLUSTER_HEAD = "cluster_head"
    CLUSTER_MEMBER = "cluster_member"
    GATEWAY = "gateway"

@dataclass
class LinkQuality:
    """Link quality metrics between two nodes"""
    rssi: float  # Received Signal Strength Indicator
    pdr: float   # Packet Delivery Ratio
    prediction_score: float = 0.0

@dataclass
class Message:
    """Network message structure"""
    sender_id: int
    receiver_id: int
    msg_type: str
    content: any
    timestamp: float
    energy_cost: float = 0.0

class Node:
    """IoT Node in the ARPMEC network"""
    
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 100.0):
        self.id = node_id
        self.x = x
        self.y = y
        self.energy = initial_energy
        self.initial_energy = initial_energy
        
        # Network state
        self.state = NodeState.IDLE
        self.cluster_id = None
        self.cluster_head_id = None
        self.cluster_members: List[int] = []
        
        # Link quality tracking
        self.neighbors: Dict[int, LinkQuality] = {}
        self.hello_received: Dict[int, int] = {}  # Count of HELLO messages received from each node
        
        # Routing
        self.routing_table: Dict[int, int] = {}  # destination -> next_hop
        self.data_items: List[any] = []
        
        # Energy model parameters (from Table 3 in paper)
        self.et = 0.03  # Transmission energy (J)
        self.er = 0.02  # Reception energy (J)
        self.eamp = 0.01  # Amplification energy (J)
        
    def distance_to(self, other: 'Node') -> float:
        """Calculate Euclidean distance to another node"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def calculate_energy_consumption(self, num_items: int, distance: float) -> float:
        """Calculate energy consumption using the model from Equation 8"""
        Q = 1.0  # Energy parameter
        transmission_energy = Q * num_items * (self.et + self.eamp * distance**2)
        reception_energy = self.er * num_items
        return transmission_energy + reception_energy
    
    def update_energy(self, energy_cost: float):
        """Update node energy after transmission/reception"""
        self.energy = max(0, self.energy - energy_cost)
    
    def is_alive(self) -> bool:
        """Check if node has enough energy to operate"""
        return self.energy > 0.1  # Minimum energy threshold

class ARPMECProtocol:
    """Implementation of ARPMEC Protocol Algorithms"""
    
    def __init__(self, nodes: List[Node], num_channels: int = 16, hello_messages: int = 100):
        self.nodes = {node.id: node for node in nodes}
        self.num_channels = num_channels
        self.R = hello_messages  # Number of HELLO messages for LQE
        self.communication_range = 1000.0  # 1km as specified in paper
        self.current_time = 0.0
        self.clusters: Dict[int, List[int]] = {}  # cluster_head_id -> [member_ids]
        
    def get_neighbors(self, node_id: int) -> List[int]:
        """Get list of neighbors within communication range"""
        node = self.nodes[node_id]
        neighbors = []
        
        for other_id, other_node in self.nodes.items():
            if other_id != node_id:
                distance = node.distance_to(other_node)
                if distance <= self.communication_range:
                    neighbors.append(other_id)
        
        return neighbors
    
    def simulate_rssi(self, distance: float) -> float:
        """Simulate RSSI based on distance with some noise"""
        # Simple path loss model with noise
        if distance == 0:
            return -30.0  # Very strong signal
        
        # Free space path loss: RSSI = -20*log10(d) - 20*log10(f) + 27.55
        # Simplified for simulation
        rssi = -40 - 20 * math.log10(distance / 100.0) + random.gauss(0, 5)
        return max(-100, min(-30, rssi))  # Clamp between -100 and -30 dBm
    
    def predict_link_quality(self, rssi: float, pdr: float) -> float:
        """
        Predict link quality using machine learning approach
        Simplified version of the RF algorithm mentioned in paper
        """
        # Normalize RSSI (-100 to -30 dBm) to (0 to 1)
        rssi_normalized = (rssi + 100) / 70.0
        
        # Weighted combination (simplified ML prediction)
        prediction_score = 0.6 * pdr + 0.4 * rssi_normalized
        
        # Add some randomness to simulate ML prediction variance
        prediction_score += random.gauss(0, 0.05)
        
        return max(0, min(1, prediction_score))
    
    def clustering_algorithm(self):
        """
        Algorithm 2: Clustering of objects using link quality prediction
        """
        print("Starting clustering algorithm...")
        
        # Step 1: Send HELLO messages for neighbor discovery
        for channel in range(self.num_channels):
            for node_id in sorted(self.nodes.keys()):  # Ordered by ID as specified
                node = self.nodes[node_id]
                if not node.is_alive():
                    continue
                
                neighbors = self.get_neighbors(node_id)
                
                for _ in range(self.R):  # Send R HELLO messages
                    for neighbor_id in neighbors:
                        neighbor = self.nodes[neighbor_id]
                        if not neighbor.is_alive():
                            continue
                        
                        # Simulate message transmission
                        distance = node.distance_to(neighbor)
                        rssi = self.simulate_rssi(distance)
                        
                        # Update neighbor's received message count
                        if node_id not in neighbor.hello_received:
                            neighbor.hello_received[node_id] = 0
                        neighbor.hello_received[node_id] += 1
                        
                        # Calculate energy cost
                        energy_cost = node.calculate_energy_consumption(1, distance)
                        node.update_energy(energy_cost)
                        
                        # Update link quality at receiver
                        pdr = neighbor.hello_received[node_id] / (self.R * self.num_channels)
                        prediction_score = self.predict_link_quality(rssi, pdr)
                        
                        neighbor.neighbors[node_id] = LinkQuality(
                            rssi=rssi,
                            pdr=pdr,
                            prediction_score=prediction_score
                        )
        
        # Step 2: JOIN phase - each node decides which cluster to join
        join_decisions: Dict[int, int] = {}  # node_id -> chosen_cluster_head_id
        
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            if not node.is_alive():
                continue
            
            if not node.neighbors:
                continue
            
            # Find neighbor with best link quality prediction
            best_neighbor_id = max(
                node.neighbors.keys(),
                key=lambda n_id: node.neighbors[n_id].prediction_score
            )
            
            join_decisions[node_id] = best_neighbor_id
        
        # Step 3: Cluster head election
        cluster_heads = set()
        cluster_assignments: Dict[int, int] = {}  # node_id -> cluster_head_id
        
        for node_id, chosen_head in join_decisions.items():
            if chosen_head not in cluster_heads:
                cluster_heads.add(chosen_head)
                self.nodes[chosen_head].state = NodeState.CLUSTER_HEAD
                self.nodes[chosen_head].cluster_id = chosen_head
                
        # Assign cluster members
        for node_id, chosen_head in join_decisions.items():
            if node_id != chosen_head:
                self.nodes[node_id].state = NodeState.CLUSTER_MEMBER
                self.nodes[node_id].cluster_head_id = chosen_head
                self.nodes[node_id].cluster_id = chosen_head
                
                if chosen_head not in self.clusters:
                    self.clusters[chosen_head] = []
                self.clusters[chosen_head].append(node_id)
                self.nodes[chosen_head].cluster_members.append(node_id)
        
        print(f"Clustering complete. Created {len(cluster_heads)} clusters.")
        return self.clusters
    
    def adaptive_routing_algorithm(self, rounds: int = 200):
        """
        Algorithm 3: Adaptive data routing between network's objects
        """
        print(f"Starting adaptive routing for {rounds} rounds...")
        
        for round_num in range(rounds):
            print(f"Round {round_num + 1}/{rounds}")
            
            # Process each cluster
            for cluster_head_id, members in self.clusters.items():
                cluster_head = self.nodes[cluster_head_id]
                
                if not cluster_head.is_alive():
                    # Cluster head is dead, need re-election
                    self._elect_new_cluster_head(cluster_head_id, members)
                    continue
                
                # Check if cluster head has enough energy for the round
                energy_threshold = 5.0  # Minimum energy to act as CH
                
                if cluster_head.energy < energy_threshold:
                    # CH gives up role
                    print(f"Cluster head {cluster_head_id} giving up role due to low energy")
                    self._elect_new_cluster_head(cluster_head_id, members)
                    continue
                
                # Cluster head operations
                self._cluster_head_operations(cluster_head_id, members)
                
                # Cluster member operations
                for member_id in members:
                    if self.nodes[member_id].is_alive():
                        self._cluster_member_operations(member_id)
        
        print("Adaptive routing completed.")
    
    def _elect_new_cluster_head(self, old_head_id: int, members: List[int]):
        """Elect new cluster head based on residual energy"""
        candidates = []
        
        for member_id in members:
            member = self.nodes[member_id]
            if member.is_alive():
                candidates.append((member.energy, member.id))
        
        if not candidates:
            # No viable candidates, cluster dissolves
            del self.clusters[old_head_id]
            return
        
        # Select member with highest energy (and highest ID as tiebreaker)
        candidates.sort(reverse=True)
        new_head_energy, new_head_id = candidates[0]
        
        # Update roles
        old_head = self.nodes[old_head_id]
        old_head.state = NodeState.CLUSTER_MEMBER
        old_head.cluster_head_id = new_head_id
        
        new_head = self.nodes[new_head_id]
        new_head.state = NodeState.CLUSTER_HEAD
        new_head.cluster_head_id = None
        new_head.cluster_members = [m for m in members if m != new_head_id]
        new_head.cluster_members.append(old_head_id)
        
        # Update cluster mapping
        self.clusters[new_head_id] = self.clusters[old_head_id]
        del self.clusters[old_head_id]
        
        print(f"New cluster head elected: {new_head_id} (energy: {new_head_energy:.2f})")
    
    def _cluster_head_operations(self, cluster_head_id: int, members: List[int]):
        """Operations performed by cluster head during a round"""
        cluster_head = self.nodes[cluster_head_id]
        
        # Listen for incoming data items
        # Simulate receiving data from members and gateways
        
        # Aggregate and forward data as needed
        # This is simplified - in real implementation, would handle actual data routing
        
        # Calculate energy consumption for cluster head operations
        num_operations = len(members) + 1  # Communication with all members + gateway
        avg_distance = 100.0  # Average distance for cluster operations
        energy_cost = cluster_head.calculate_energy_consumption(num_operations, avg_distance)
        cluster_head.update_energy(energy_cost)
    
    def _cluster_member_operations(self, member_id: int):
        """Operations performed by cluster member during a round"""
        member = self.nodes[member_id]
        
        # Send data to cluster head
        cluster_head_id = member.cluster_head_id
        if cluster_head_id and cluster_head_id in self.nodes:
            cluster_head = self.nodes[cluster_head_id]
            distance = member.distance_to(cluster_head)
            
            # Simulate sending data items
            num_items = random.randint(1, 5)  # Random number of data items
            energy_cost = member.calculate_energy_consumption(num_items, distance)
            member.update_energy(energy_cost)
    
    def get_network_statistics(self) -> Dict:
        """Get current network statistics"""
        alive_nodes = sum(1 for node in self.nodes.values() if node.is_alive())
        total_energy_consumed = sum(
            node.initial_energy - node.energy for node in self.nodes.values()
        )
        avg_energy_remaining = sum(
            node.energy for node in self.nodes.values() if node.is_alive()
        ) / max(1, alive_nodes)
        
        return {
            "alive_nodes": alive_nodes,
            "total_nodes": len(self.nodes),
            "total_energy_consumed": total_energy_consumed,
            "avg_energy_remaining": avg_energy_remaining,
            "num_clusters": len(self.clusters),
            "cluster_heads": list(self.clusters.keys())
        }

def create_random_network(num_nodes: int, area_size: float = 1000.0) -> List[Node]:
    """Create a random network of IoT nodes"""
    nodes = []
    for i in range(num_nodes):
        x = random.uniform(0, area_size)
        y = random.uniform(0, area_size)
        energy = random.uniform(80, 120)  # Random initial energy
        nodes.append(Node(i, x, y, energy))
    
    return nodes

# Example usage and simulation
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create network
    print("Creating network with 50 nodes...")
    nodes = create_random_network(50)
    
    # Initialize ARPMEC protocol
    arpmec = ARPMECProtocol(nodes, num_channels=16, hello_messages=100)
    
    # Run clustering algorithm
    clusters = arpmec.clustering_algorithm()
    
    print("\nCluster formation results:")
    for head_id, members in clusters.items():
        print(f"Cluster {head_id}: Head energy = {arpmec.nodes[head_id].energy:.2f}J, "
              f"Members = {len(members)}")
    
    # Get initial statistics
    initial_stats = arpmec.get_network_statistics()
    print(f"\nInitial network statistics:")
    print(f"Total nodes: {initial_stats['total_nodes']}")
    print(f"Alive nodes: {initial_stats['alive_nodes']}")
    print(f"Number of clusters: {initial_stats['num_clusters']}")
    print(f"Average energy remaining: {initial_stats['avg_energy_remaining']:.2f}J")
    
    # Run adaptive routing
    arpmec.adaptive_routing_algorithm(rounds=50)
    
    # Get final statistics
    final_stats = arpmec.get_network_statistics()
    print(f"\nFinal network statistics:")
    print(f"Alive nodes: {final_stats['alive_nodes']}")
    print(f"Total energy consumed: {final_stats['total_energy_consumed']:.2f}J")
    print(f"Average energy remaining: {final_stats['avg_energy_remaining']:.2f}J")
    print(f"Number of active clusters: {final_stats['num_clusters']}")
    
    # Calculate network lifetime and efficiency
    network_lifetime = final_stats['alive_nodes'] / initial_stats['total_nodes']
    energy_efficiency = final_stats['avg_energy_remaining'] / 100.0  # Assuming 100J initial
    
    print(f"\nPerformance Metrics:")
    print(f"Network lifetime: {network_lifetime:.2%}")
    print(f"Energy efficiency: {energy_efficiency:.2%}")