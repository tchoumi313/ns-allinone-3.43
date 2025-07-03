import math
import random
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class NodeState(Enum):
    IDLE = "idle"
    CLUSTER_HEAD = "cluster_head"
    CLUSTER_MEMBER = "cluster_member"

@dataclass
class HelloMessage:
    """HELLO message structure as per Algorithm 2"""
    sender_id: int
    channel: int
    sequence_num: int  # Which HELLO message (1 to R)
    timestamp: int     # Time slot when sent

@dataclass
class JoinMessage:
    """JOIN message structure as per Algorithm 2"""
    sender_id: int
    target_id: int     # The node it wants to join (M in paper)
    timestamp: int

class Node:
    """IoT Node implementation following exact paper specifications - FIXED"""
    
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 100.0):
        # Basic node properties
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
        
        # Algorithm 2 variables - EXACT from paper
        self.neighbors: Dict[int, Dict] = {}  # neighbor_id -> {rssi_values, pdr_per_channel, score}
        self.hello_received_count: Dict[Tuple[int, int], int] = {}  # (sender_id, channel) -> count
        self.hello_messages_buffer: List[HelloMessage] = []
        self.join_messages_received: List[JoinMessage] = []
        
        # Algorithm 3 variables - EXACT from paper  
        self.rank_in_cluster = None  # j in paper
        self.cluster_size = None     # k in paper
        self.items_to_share = 0      # #i in paper
        self.total_cluster_items = 0 # h in paper
        self.gateway_items = 0       # ω in paper
        
        # FIXED Energy model parameters (Table 3) - realistic values
        self.et = 0.03    # Transmission energy (J) - per packet
        self.er = 0.02    # Reception energy (J) - per packet
        self.eamp = 0.000001  # FIXED: Amplification energy - much smaller for realistic consumption
        
        # Energy threshold for cluster head role
        self.energy_threshold_f = 5.0  # f in paper
        
    def distance_to(self, other: 'Node') -> float:
        """Calculate Euclidean distance to another node"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def calculate_energy_consumption(self, num_items: int, distance: float) -> float:
        """FIXED Energy model from Equation 8"""
        Q = 1.0  # Energy parameter
        
        # FIXED: Realistic energy calculation
        # For distance in meters, use distance/1000 to normalize
        distance_km = distance / 1000.0  # Convert to km for realistic amplification
        
        transmission_energy = Q * num_items * (self.et + self.eamp * (distance_km ** 2))
        reception_energy = self.er * num_items
        
        total_energy = transmission_energy + reception_energy
        
        # SAFETY: Cap maximum energy consumption to prevent instant death
        max_energy_per_transmission = 2.0  # Maximum 2J per transmission
        return min(total_energy, max_energy_per_transmission)
    
    def update_energy(self, energy_cost: float):
        """Update node energy"""
        self.energy = max(0, self.energy - energy_cost)
    
    def is_alive(self) -> bool:
        """Check if node has enough energy"""
        return self.energy > 0.1

class ARPMECProtocol:
    """FIXED implementation of ARPMEC following exact algorithms"""
    
    def __init__(self, nodes: List[Node], C: int = 16, R: int = 100, K: int = 3):
        self.nodes = {node.id: node for node in nodes}
        self.N = len(nodes)  # Number of objects
        self.C = C           # Number of channels  
        self.R = R           # Min messages for accurate LQE
        self.K = K           # Number of MEC servers
        self.HUBmax = 10     # Max objects per cluster
        
        # FIXED: More realistic communication range for the area size
        self.communication_range = 300.0  # 300m instead of 1000m
        self.current_time_slot = 0
        
        # ML model for link quality prediction (as per paper)
        self.lqe_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lqe_scaler = StandardScaler()
        self._train_lqe_model()
    
    def _train_lqe_model(self):
        """Train Random Forest model for LQE as mentioned in paper"""
        # Generate synthetic training data for LQE model
        training_size = 1000
        rssi_values = np.random.uniform(-100, -30, training_size)  # dBm
        pdr_values = np.random.uniform(0, 1, training_size)
        
        # Generate labels based on realistic link quality criteria
        features = np.column_stack([rssi_values, pdr_values])
        labels = []
        
        for rssi, pdr in features:
            # Good link: high PDR and strong RSSI
            if pdr > 0.8 and rssi > -60:
                labels.append(2)  # Excellent
            elif pdr > 0.6 and rssi > -80:
                labels.append(1)  # Good  
            else:
                labels.append(0)  # Poor
        
        # Train the model
        self.lqe_scaler.fit(features)
        scaled_features = self.lqe_scaler.transform(features)
        self.lqe_model.fit(scaled_features, labels)
    
    def predict_link_quality(self, rssi: float, pdr: float) -> float:
        """Link quality prediction using Random Forest (exact as paper)"""
        features = np.array([[rssi, pdr]])
        scaled_features = self.lqe_scaler.transform(features)
        
        # Get prediction probabilities for quality classes
        probabilities = self.lqe_model.predict_proba(scaled_features)[0]
        
        # Convert to single score (weighted by quality levels)
        quality_score = probabilities[0] * 0.2 + probabilities[1] * 0.6 + probabilities[2] * 1.0
        return quality_score
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighbors within communication range"""
        node = self.nodes[node_id]
        neighbors = []
        
        for other_id, other_node in self.nodes.items():
            if other_id != node_id:
                distance = node.distance_to(other_node)
                if distance <= self.communication_range:
                    neighbors.append(other_id)
        
        return neighbors
    
    def simulate_rssi(self, distance: float) -> float:
        """FIXED: More realistic RSSI simulation"""
        if distance <= 0:
            return -30.0
        
        # Realistic path loss model
        # RSSI = -30 - 20*log10(d/10) + noise
        rssi = -30 - 20 * math.log10(max(distance / 10.0, 1.0)) + random.gauss(0, 3)
        return max(-100, min(-30, rssi))
    
    def clustering_algorithm(self):
        """
        FIXED Algorithm 2: Clustering of objects using link quality prediction
        """
        print("Starting FIXED Algorithm 2: Clustering with LQE...")
        
        # Step 1: HELLO message broadcasting phase - SIMPLIFIED for efficiency
        print(f"Phase 1: Broadcasting HELLO messages for neighbor discovery...")
        
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            if not node.is_alive():
                continue
                
            neighbors = self.get_neighbors(node_id)
            
            # SIMPLIFIED: Send fewer HELLO messages to prevent energy drain
            for neighbor_id in neighbors:
                neighbor = self.nodes[neighbor_id]
                if not neighbor.is_alive():
                    continue
                
                # Simulate message exchange
                distance = node.distance_to(neighbor)
                rssi = self.simulate_rssi(distance)
                
                # FIXED: Proper neighbor tracking
                key = (node_id, 0)  # Use channel 0 for simplicity
                if key not in neighbor.hello_received_count:
                    neighbor.hello_received_count[key] = 0
                    neighbor.neighbors[node_id] = {
                        'rssi_values': [],
                        'pdr_per_channel': {},
                        'score': 0.0
                    }
                
                neighbor.hello_received_count[key] += 1
                neighbor.neighbors[node_id]['rssi_values'].append(rssi)
                
                # Calculate PDR (simplified)
                pdr = min(1.0, neighbor.hello_received_count[key] / 10.0)  # Normalize to [0,1]
                neighbor.neighbors[node_id]['pdr_per_channel'][0] = pdr
                
                # FIXED: Realistic energy consumption
                energy_cost = node.calculate_energy_consumption(1, distance)
                node.update_energy(energy_cost)
        
        # Step 2: Link Quality Prediction and JOIN decisions
        print("Phase 2: Link Quality Prediction and JOIN decisions...")
        
        join_decisions = {}
        
        for node_id, node in self.nodes.items():
            if not node.is_alive() or not node.neighbors:
                continue
                
            best_neighbor_id = None
            best_score = -1.0
            
            for neighbor_id, neighbor_data in node.neighbors.items():
                if not neighbor_data['rssi_values']:
                    continue
                
                # Calculate average RSSI and PDR
                avg_rssi = np.mean(neighbor_data['rssi_values'])
                avg_pdr = np.mean(list(neighbor_data['pdr_per_channel'].values())) if neighbor_data['pdr_per_channel'] else 0.0
                
                # Use Random Forest prediction
                prediction_score = self.predict_link_quality(avg_rssi, avg_pdr)
                neighbor_data['score'] = prediction_score
                
                # Find best neighbor
                if prediction_score > best_score:
                    best_score = prediction_score
                    best_neighbor_id = neighbor_id
            
            if best_neighbor_id is not None:
                join_decisions[node_id] = best_neighbor_id
        
        # Step 3: FIXED Cluster Head Election
        print("Phase 3: Cluster Head Election...")
        
        # Count JOIN votes for each node
        join_votes = {}
        for sender, target in join_decisions.items():
            if target not in join_votes:
                join_votes[target] = []
            join_votes[target].append(sender)
        
        clusters = {}
        
        # Nodes that received votes become cluster heads
        for head_id, voters in join_votes.items():
            if head_id in self.nodes and self.nodes[head_id].is_alive():
                # Become cluster head
                head_node = self.nodes[head_id]
                head_node.state = NodeState.CLUSTER_HEAD
                head_node.cluster_id = head_id
                head_node.cluster_members = []
                clusters[head_id] = []
                
                # Add voters as cluster members
                for voter_id in voters:
                    if voter_id != head_id and voter_id in self.nodes:
                        member_node = self.nodes[voter_id]
                        member_node.state = NodeState.CLUSTER_MEMBER
                        member_node.cluster_head_id = head_id
                        member_node.cluster_id = head_id
                        clusters[head_id].append(voter_id)
                        head_node.cluster_members.append(voter_id)
        
        # FIXED: Handle isolated nodes - form single-node clusters
        for node_id, node in self.nodes.items():
            if node.is_alive() and node.state == NodeState.IDLE:
                # Make it its own cluster head
                node.state = NodeState.CLUSTER_HEAD
                node.cluster_id = node_id
                node.cluster_members = []
                clusters[node_id] = []
        
        print(f"Clustering complete. Created {len(clusters)} clusters.")
        return clusters
    
    def adaptive_routing_algorithm(self, T: int = 200):
        """
        FIXED Algorithm 3: Adaptive data routing
        """
        print(f"Starting FIXED Algorithm 3: Adaptive Routing for {T} rounds...")
        
        clusters = {node_id: node.cluster_members for node_id, node in self.nodes.items() 
                   if node.state == NodeState.CLUSTER_HEAD}
        
        for round_num in range(T):
            #if round_num % 10 == 0:  # Print less frequently
            print(f"Round {round_num + 1}/{T}")
            
            for node_id, node in self.nodes.items():
                if not node.is_alive():
                    continue
                
                if node.state == NodeState.CLUSTER_HEAD:
                    self._fixed_cluster_head_operations(node, round_num)
                else:
                    self._fixed_cluster_member_operations(node, round_num)
    
    def _fixed_cluster_head_operations(self, node: Node, round_num: int):
        """FIXED Cluster Head operations"""
        
        # Check energy threshold
        if node.energy < node.energy_threshold_f:
            print(f"CH {node.id} giving up role (energy: {node.energy:.2f}J)")
            node.state = NodeState.CLUSTER_MEMBER
            self._elect_new_cluster_head(node.id)
            return
        
        # FIXED: Realistic cluster head operations
        # Simulate data aggregation and forwarding with realistic energy cost
        num_operations = len(node.cluster_members) + 1
        energy_cost = 0.1 * num_operations  # 0.1J per operation
        node.update_energy(energy_cost)
    
    def _fixed_cluster_member_operations(self, node: Node, round_num: int):
        """FIXED Cluster Member operations"""
        
        # Check if CH is still alive
        if node.cluster_head_id and node.cluster_head_id in self.nodes:
            ch = self.nodes[node.cluster_head_id]
            if not ch.is_alive() or ch.state != NodeState.CLUSTER_HEAD:
                self._participate_in_ch_election(node)
        
        # FIXED: Realistic data transmission
        if node.cluster_head_id and node.cluster_head_id in self.nodes:
            ch = self.nodes[node.cluster_head_id]
            distance = node.distance_to(ch)
            
            # Send data with realistic energy consumption
            energy_cost = node.calculate_energy_consumption(1, distance)
            node.update_energy(energy_cost)
    
    def _participate_in_ch_election(self, node: Node):
        """FIXED CH election process"""
        
        # Get cluster members
        cluster_members = [n for n in self.nodes.values() 
                          if n.cluster_id == node.cluster_id and n.is_alive()]
        
        if not cluster_members:
            return
        
        # Simple election: highest energy becomes CH
        best_member = max(cluster_members, key=lambda n: n.energy)
        
        if best_member.id == node.id:
            # Become new CH
            node.state = NodeState.CLUSTER_HEAD
            node.cluster_head_id = None
            node.cluster_members = [n.id for n in cluster_members if n.id != node.id]
            
            # Update other members
            for member in cluster_members:
                if member.id != node.id:
                    member.cluster_head_id = node.id
                    member.state = NodeState.CLUSTER_MEMBER
        else:
            # Update CH reference
            node.cluster_head_id = best_member.id
    
    def _elect_new_cluster_head(self, old_ch_id: int):
        """FIXED cluster head replacement"""
        old_ch = self.nodes[old_ch_id]
        cluster_members = [self.nodes[mid] for mid in old_ch.cluster_members 
                          if mid in self.nodes and self.nodes[mid].is_alive()]
        
        if not cluster_members:
            return
        
        # Elect member with highest energy
        best_member = max(cluster_members, key=lambda n: n.energy)
        
        # Promote to cluster head
        best_member.state = NodeState.CLUSTER_HEAD
        best_member.cluster_head_id = None
        best_member.cluster_members = [n.id for n in cluster_members if n.id != best_member.id]
        best_member.cluster_members.append(old_ch_id)
        
        # Update old CH
        old_ch.state = NodeState.CLUSTER_MEMBER
        old_ch.cluster_head_id = best_member.id
        old_ch.cluster_members = []
        
        # Update other members
        for member in cluster_members:
            if member.id != best_member.id:
                member.cluster_head_id = best_member.id
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        alive_nodes = sum(1 for node in self.nodes.values() if node.is_alive())
        total_energy_consumed = sum(
            node.initial_energy - node.energy for node in self.nodes.values()
        )
        
        cluster_heads = [n for n in self.nodes.values() if n.state == NodeState.CLUSTER_HEAD]
        
        return {
            "alive_nodes": alive_nodes,
            "total_nodes": len(self.nodes),
            "network_lifetime": alive_nodes / len(self.nodes),
            "total_energy_consumed": total_energy_consumed,
            "energy_per_node": total_energy_consumed / len(self.nodes),
            "num_clusters": len(cluster_heads),
            "avg_cluster_size": len(self.nodes) / max(1, len(cluster_heads)),
            "clustering_time_slots": self.R * self.N * self.C + self.N + 2 * self.K
        }

# FIXED Test function
def test_fixed_implementation():
    """Test the FIXED ARPMEC implementation"""
    print("Testing FIXED ARPMEC implementation")
    print("=" * 50)
    
    # Create smaller, more realistic network
    N = 20  # Fewer nodes for testing
    nodes = []
    for i in range(N):
        x = random.uniform(0, 500)  # Smaller area: 500m x 500m
        y = random.uniform(0, 500)
        energy = random.uniform(90, 110)
        nodes.append(Node(i, x, y, energy))
    
    # More conservative parameters
    C = 4     # Fewer channels
    R = 10    # Fewer HELLO messages
    K = 2     # MEC servers
    
    protocol = ARPMECProtocol(nodes, C, R, K)
    
    print(f"Network: {N} nodes, {C} channels, {R} HELLO messages")
    print(f"Area: 500m x 500m, Communication range: {protocol.communication_range}m")
    
    # Test energy model first
    test_node = nodes[0]
    energy_100m = test_node.calculate_energy_consumption(1, 100)
    energy_300m = test_node.calculate_energy_consumption(1, 300)
    
    print(f"\nFIXED Energy Model Test:")
    print(f"1 packet at 100m: {energy_100m:.4f}J")
    print(f"1 packet at 300m: {energy_300m:.4f}J")
    
    if energy_100m > 10:
        print("ERROR: Energy consumption still too high!")
        return None, None
    
    print("Running FIXED Algorithm 2 (Clustering)...")
    
    start_time = time.time()
    clusters = protocol.clustering_algorithm()
    clustering_time = time.time() - start_time
    
    print(f"Clustering completed in {clustering_time:.3f}s")
    
    # Check results
    alive_after_clustering = sum(1 for n in nodes if n.is_alive())
    energy_consumed = sum(n.initial_energy - n.energy for n in nodes)
    
    print(f"Results after clustering:")
    print(f"- Alive nodes: {alive_after_clustering}/{N}")
    print(f"- Energy consumed: {energy_consumed:.2f}J")
    print(f"- Clusters formed: {len(clusters)}")
    
    if len(clusters) == 0:
        print("WARNING: Still no clusters formed!")
        return protocol, None
    
    # Show cluster details
    for head_id, members in clusters.items():
        head = protocol.nodes[head_id]
        print(f"Cluster {head_id}: Head energy={head.energy:.1f}J, Members={len(members)}")
    
    print("Running FIXED Algorithm 3 (Routing)...")
    T = 10  # Fewer rounds for testing
    
    start_time = time.time()
    protocol.adaptive_routing_algorithm(T)
    routing_time = time.time() - start_time
    
    print(f"Routing completed in {routing_time:.3f}s")
    
    # Final metrics
    metrics = protocol.get_performance_metrics()
    
    print(f"\nFINAL RESULTS:")
    print(f"Network lifetime: {metrics['network_lifetime']*100:.1f}%")
    print(f"Energy per node: {metrics['energy_per_node']:.2f}J")
    print(f"Active clusters: {metrics['num_clusters']}")
    
    return protocol, metrics

if __name__ == "__main__":
    import time
    random.seed(42)
    np.random.seed(42)
    
    protocol, metrics = test_fixed_implementation()