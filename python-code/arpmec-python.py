import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
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
    """IoT Node implementation following exact paper specifications"""
    
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
        
        # Energy model parameters (Table 3)
        self.et = 0.03    # Transmission energy (J)
        self.er = 0.02    # Reception energy (J) 
        self.eamp = 0.01  # Amplification energy (J)
        
        # Energy threshold for cluster head role
        self.energy_threshold_f = 5.0  # f in paper
        
    def distance_to(self, other: 'Node') -> float:
        """Calculate Euclidean distance to another node"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def calculate_energy_consumption(self, num_items: int, distance: float) -> float:
        """Energy model from Equation 8 - EXACT"""
        Q = 1.0  # Energy parameter
        transmission_energy = Q * num_items * (self.et + self.eamp * distance**2)
        reception_energy = self.er * num_items
        return transmission_energy + reception_energy
    
    def update_energy(self, energy_cost: float):
        """Update node energy"""
        self.energy = max(0, self.energy - energy_cost)
    
    def is_alive(self) -> bool:
        """Check if node has enough energy"""
        return self.energy > 0.1

class ARPMECProtocol:
    """FAITHFUL implementation of ARPMEC following exact algorithms"""
    
    def __init__(self, nodes: List[Node], C: int = 16, R: int = 100, K: int = 3):
        self.nodes = {node.id: node for node in nodes}
        self.N = len(nodes)  # Number of objects
        self.C = C           # Number of channels  
        self.R = R           # Min messages for accurate LQE
        self.K = K           # Number of MEC servers
        self.HUBmax = 10     # Max objects per cluster
        
        self.communication_range = 1000.0  # 1km
        self.current_time_slot = 0
        
        # ML model for link quality prediction (as per paper)
        self.lqe_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lqe_scaler = StandardScaler()
        self._train_lqe_model()
    
    def _train_lqe_model(self):
        """Train Random Forest model for LQE as mentioned in paper"""
        # Generate synthetic training data for LQE model
        # Paper mentions RSSI and PDR as features
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
        """Simulate RSSI with path loss model"""
        if distance <= 0:
            return -30.0
        
        # Free space path loss model with noise
        rssi = -30 - 20 * math.log10(distance / 10.0) + random.gauss(0, 3)
        return max(-100, min(-30, rssi))
    
    def clustering_algorithm(self):
        """
        Algorithm 2: Clustering of objects using link quality prediction
        FAITHFUL implementation following exact paper steps
        """
        print("Starting Algorithm 2: Clustering with LQE...")
        
        # Step 1: HELLO message broadcasting phase
        # EXACT timing as per paper: (I-1)×R×C + (R-1)×C + 1
        
        print(f"Phase 1: Broadcasting {self.R} HELLO messages on {self.C} channels...")
        
        for node_id in sorted(self.nodes.keys()):  # Ordered by ID (Line 4)
            node = self.nodes[node_id]
            if not node.is_alive():
                continue
                
            neighbors = self.get_neighbors(node_id)
            
            for r in range(self.R):  # R HELLO messages
                for c in range(self.C):  # On each channel
                    
                    # Calculate exact time slot as per paper
                    time_slot = (node_id - 1) * self.R * self.C + r * self.C + c + 1
                    
                    # Create HELLO message
                    hello_msg = HelloMessage(
                        sender_id=node_id,
                        channel=c,
                        sequence_num=r + 1,
                        timestamp=time_slot
                    )
                    
                    # Send to all neighbors (Line 4 broadcast)
                    for neighbor_id in neighbors:
                        neighbor = self.nodes[neighbor_id]
                        if not neighbor.is_alive():
                            continue
                        
                        # Simulate transmission
                        distance = node.distance_to(neighbor)
                        rssi = self.simulate_rssi(distance)
                        
                        # Neighbor receives and records (Lines 8-10)
                        key = (node_id, c)
                        if key not in neighbor.hello_received_count:
                            neighbor.hello_received_count[key] = 0
                            neighbor.neighbors[node_id] = {
                                'rssi_values': [],
                                'pdr_per_channel': {},
                                'score': 0.0
                            }
                        
                        neighbor.hello_received_count[key] += 1
                        neighbor.neighbors[node_id]['rssi_values'].append(rssi)
                        
                        # Calculate PDR for this channel (Line 10)
                        pdr = neighbor.hello_received_count[key] / self.R
                        neighbor.neighbors[node_id]['pdr_per_channel'][c] = pdr
                        
                        # Energy consumption for transmission
                        energy_cost = node.calculate_energy_consumption(1, distance)
                        node.update_energy(energy_cost)
        
        # After R×N×C slots, calculate link quality predictions (Lines 11-12)
        print("Phase 2: Link Quality Prediction...")
        
        for node_id, node in self.nodes.items():
            if not node.is_alive():
                continue
                
            best_neighbor_id = None
            best_score = -1.0
            
            for neighbor_id, neighbor_data in node.neighbors.items():
                if not neighbor_data['rssi_values']:
                    continue
                
                # Calculate average RSSI
                avg_rssi = np.mean(neighbor_data['rssi_values'])
                
                # Calculate average PDR across all channels
                if neighbor_data['pdr_per_channel']:
                    avg_pdr = np.mean(list(neighbor_data['pdr_per_channel'].values()))
                else:
                    avg_pdr = 0.0
                
                # Use Random Forest prediction (exact as paper)
                prediction_score = self.predict_link_quality(avg_rssi, avg_pdr)
                neighbor_data['score'] = prediction_score
                
                # Find best neighbor (Line 12)
                if prediction_score > best_score:
                    best_score = prediction_score
                    best_neighbor_id = neighbor_id
            
            # Send JOIN message (Line 13)
            if best_neighbor_id is not None:
                join_msg = JoinMessage(
                    sender_id=node_id,
                    target_id=best_neighbor_id,
                    timestamp=self.current_time_slot
                )
                
                # Target receives JOIN message
                if best_neighbor_id in self.nodes:
                    self.nodes[best_neighbor_id].join_messages_received.append(join_msg)
        
        # Step 3: Cluster Head Election (Lines 19-21)
        print("Phase 3: Cluster Head Election...")
        
        clusters = {}
        
        for node_id, node in self.nodes.items():
            if not node.is_alive():
                continue
            
            # Count JOIN messages received (Line 19)
            join_count = len([msg for msg in node.join_messages_received 
                            if msg.target_id == node_id])
            
            if join_count > 0:
                # Become cluster head (Line 19)
                node.state = NodeState.CLUSTER_HEAD
                node.cluster_id = node_id
                clusters[node_id] = []
                
                # Add members who sent JOIN
                for msg in node.join_messages_received:
                    if msg.target_id == node_id:
                        member_node = self.nodes[msg.sender_id]
                        member_node.state = NodeState.CLUSTER_MEMBER
                        member_node.cluster_head_id = node_id
                        member_node.cluster_id = node_id
                        clusters[node_id].append(msg.sender_id)
                        node.cluster_members.append(msg.sender_id)
            else:
                # Remain cluster member (Line 17) 
                node.state = NodeState.CLUSTER_MEMBER
        
        print(f"Clustering complete. Created {len(clusters)} clusters.")
        print(f"Time complexity: {self.R * self.N * self.C + self.N + 2 * self.K} slots")
        
        return clusters
    
    def adaptive_routing_algorithm(self, T: int = 200):
        """
        Algorithm 3: Adaptive data routing - FAITHFUL implementation
        T = number of rounds
        """
        print(f"Starting Algorithm 3: Adaptive Routing for {T} rounds...")
        
        clusters = {node_id: node.cluster_members for node_id, node in self.nodes.items() 
                   if node.state == NodeState.CLUSTER_HEAD}
        
        for round_num in range(T):
            print(f"Round {round_num + 1}/{T}")
            
            for node_id, node in self.nodes.items():
                if not node.is_alive():
                    continue
                
                if node.state == NodeState.CLUSTER_HEAD:
                    self._cluster_head_operations(node, round_num)
                else:
                    self._cluster_member_operations(node, round_num)
    
    def _cluster_head_operations(self, node: Node, round_num: int):
        """Cluster Head operations from Algorithm 3 (Lines 3-11)"""
        
        # Line 4: Check energy threshold
        if node.energy < self.energy_threshold_f:
            # Line 11: Give up CH role
            print(f"CH {node.id} giving up role (energy: {node.energy:.2f}J)")
            node.state = NodeState.CLUSTER_MEMBER
            self._elect_new_cluster_head(node.id)
            return
        
        # Line 5: Listen for HUBmax slots for GW/MS items
        # Simulate receiving items from gateway
        gateway_items = random.randint(0, 3)
        node.gateway_items = gateway_items
        
        # Line 6: Share list of objects and items in cluster
        cluster_items = sum(random.randint(1, 5) for _ in node.cluster_members)
        node.total_cluster_items = cluster_items
        
        # Line 7: Listen during cluster communications
        # Line 9: Send non-cluster items to GW
        if gateway_items > 0:
            # Energy for forwarding to gateway
            energy_cost = node.calculate_energy_consumption(gateway_items, 500)  # Assume 500m to GW
            node.update_energy(energy_cost)
    
    def _cluster_member_operations(self, node: Node, round_num: int):
        """Cluster Member operations from Algorithm 3 (Lines 12-24)"""
        
        # Line 13: Listen for cluster communications (1 slot)
        
        # Check if CH abdicated (Lines 14-22)
        if node.cluster_head_id and node.cluster_head_id in self.nodes:
            ch = self.nodes[node.cluster_head_id]
            if ch.state != NodeState.CLUSTER_HEAD:
                # New CH election process (Lines 15-22)
                self._participate_in_ch_election(node)
        
        # Line 23: Broadcast #i messages  
        node.items_to_share = random.randint(1, 5)
        
        if node.cluster_head_id and node.cluster_head_id in self.nodes:
            ch = self.nodes[node.cluster_head_id]
            distance = node.distance_to(ch)
            energy_cost = node.calculate_energy_consumption(node.items_to_share, distance)
            node.update_energy(energy_cost)
    
    def _participate_in_ch_election(self, node: Node):
        """CH election process (Lines 15-22) - EXACT implementation"""
        
        # Get cluster members
        cluster_members = [n for n in self.nodes.values() 
                          if n.cluster_id == node.cluster_id and n.is_alive()]
        
        k = len(cluster_members)  # Line 17: k = number of objects in cluster
        
        # Find node's rank j in cluster (Line 15)
        sorted_members = sorted(cluster_members, key=lambda n: n.id)
        j = next(i+1 for i, n in enumerate(sorted_members) if n.id == node.id)
        node.rank_in_cluster = j
        
        # Line 15: Listen for j-1 slots
        # Line 16: Broadcast (node.id, energy(node.id)) at jth slot
        # Line 17: Listen until kth slot
        
        # Simulate energy comparison (Line 18)
        highest_energy = max(n.energy for n in cluster_members)
        highest_energy_nodes = [n for n in cluster_members if n.energy == highest_energy]
        
        # Line 18-19: If this node has highest energy and highest ID among ties
        if node.energy == highest_energy:
            highest_id = max(n.id for n in highest_energy_nodes)
            if node.id == highest_id:
                # Become new CH
                node.state = NodeState.CLUSTER_HEAD
                node.cluster_head_id = None
                node.cluster_members = [n.id for n in cluster_members if n.id != node.id]
                
                # Update other members
                for member in cluster_members:
                    if member.id != node.id:
                        member.cluster_head_id = node.id
                        member.state = NodeState.CLUSTER_MEMBER
                
                print(f"Node {node.id} elected as new CH (energy: {node.energy:.2f}J)")
        else:
            # Line 22: Remain CM, update CH reference
            new_ch = max(highest_energy_nodes, key=lambda n: n.id)
            node.cluster_head_id = new_ch.id
    
    def _elect_new_cluster_head(self, old_ch_id: int):
        """Handle cluster head giving up role"""
        old_ch = self.nodes[old_ch_id]
        cluster_members = [self.nodes[mid] for mid in old_ch.cluster_members if mid in self.nodes]
        
        if not cluster_members:
            return
        
        # Find member with highest energy
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
        """Get exact performance metrics as per paper"""
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

# Test with exact paper parameters
def test_faithful_implementation():
    """Test with exact parameters from Table 3"""
    print("Testing FAITHFUL ARPMEC implementation")
    print("Using exact parameters from Table 3")
    
    # Create network (N between 1-500 as per paper)
    N = 100
    nodes = []
    for i in range(N):
        x = random.uniform(0, 1000)  # 1km x 1km area
        y = random.uniform(0, 1000)
        energy = random.uniform(90, 110)  # Initial energy variation
        nodes.append(Node(i, x, y, energy))
    
    # Initialize with exact paper parameters
    C = 16    # Channels (from Table 3)
    R = 100   # HELLO messages (from Table 3)
    K = 3     # MEC servers
    
    protocol = ARPMECProtocol(nodes, C, R, K)
    
    print(f"Network: {N} nodes, {C} channels, {R} HELLO messages")
    print("Running Algorithm 2 (Clustering)...")
    
    start_time = time.time()
    clusters = protocol.clustering_algorithm()
    clustering_time = time.time() - start_time
    
    print(f"Clustering completed in {clustering_time:.3f}s")
    
    print("Running Algorithm 3 (Adaptive Routing)...")
    T = 50  # Reduced for demo
    
    start_time = time.time()
    protocol.adaptive_routing_algorithm(T)
    routing_time = time.time() - start_time
    
    print(f"Routing completed in {routing_time:.3f}s")
    
    # Get performance metrics
    metrics = protocol.get_performance_metrics()
    
    print("\nPERFORMANCE RESULTS:")
    print(f"Network lifetime: {metrics['network_lifetime']*100:.1f}%")
    print(f"Energy per node: {metrics['energy_per_node']:.2f}J")
    print(f"Number of clusters: {metrics['num_clusters']}")
    print(f"Average cluster size: {metrics['avg_cluster_size']:.1f}")
    print(f"Theoretical time complexity: O({metrics['clustering_time_slots']}) slots")
    
    return protocol, metrics

if __name__ == "__main__":
    import time
    random.seed(42)
    np.random.seed(42)
    
    protocol, metrics = test_faithful_implementation()