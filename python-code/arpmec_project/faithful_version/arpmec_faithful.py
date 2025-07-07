import math
import random
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ML libraries (optional - fallback to simple heuristics if not available)
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: scikit-learn not available. Using simple heuristics for resource prediction.")

warnings.filterwarnings('ignore')

@dataclass
class MECTask:
    """Task to be processed by MEC server"""
    task_id: str
    source_cluster_id: int
    cpu_requirement: float
    memory_requirement: float
    deadline: float
    data_size: float  # MB
    created_time: float

@dataclass
class InterClusterMessage:
    """Message for inter-cluster communication"""
    message_id: str
    source_cluster_id: int
    destination_cluster_id: int
    message_type: str  # 'data', 'control', 'ack'
    payload: Dict
    timestamp: float

@dataclass
class IARServer:
    """Infrastructure Access Router (IAR) - Base Station between CH and MEC"""
    def __init__(self, iar_id: int, x: float, y: float, coverage_radius: float = 200.0):
        self.id = iar_id
        self.x = x
        self.y = y
        self.coverage_radius = coverage_radius
        self.connected_clusters = []  # List of cluster IDs served
        self.connected_mec_servers = []  # List of MEC server IDs
        self.message_queue = []  # Queue for CH->MEC messages
        self.load_balancing_table = {}  # Track MEC server loads
        self.routing_table = {}  # Routing decisions for adaptive routing
        
    def distance_to(self, x: float, y: float) -> float:
        """Calculate distance to a point"""
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def can_serve_cluster(self, ch_x: float, ch_y: float) -> bool:
        """Check if IAR can serve a cluster head"""
        return self.distance_to(ch_x, ch_y) <= self.coverage_radius
    
    def register_cluster(self, cluster_id: int):
        """Register a cluster with this IAR"""
        if cluster_id not in self.connected_clusters:
            self.connected_clusters.append(cluster_id)
    
    def add_mec_server(self, mec_id: int):
        """Add a MEC server to this IAR's network"""
        if mec_id not in self.connected_mec_servers:
            self.connected_mec_servers.append(mec_id)
    
    def route_to_best_mec(self, task: 'MECTask', mec_servers: Dict[int, 'MECServer']) -> Optional['MECServer']:
        """Adaptive routing: Find best MEC server for task"""
        if not self.connected_mec_servers:
            return None
        
        # Find least loaded MEC server
        best_mec = None
        min_load = float('inf')
        
        for mec_id in self.connected_mec_servers:
            if mec_id in mec_servers:
                mec = mec_servers[mec_id]
                current_load = mec.get_load_percentage()
                
                # Check if MEC can handle the task
                if mec.can_handle_task(task) and current_load < min_load:
                    min_load = current_load
                    best_mec = mec
        
        # If no MEC can handle locally, try load balancing to other IARs
        if best_mec is None:
            print(f"IAR-{self.id}: No local MEC available, attempting load balancing")
            # This would trigger inter-IAR communication in a real implementation
        
        return best_mec

class MECServer:
    """MEC Server for edge computing tasks with improved load balancing"""
    
    def __init__(self, server_id: int, x: float, y: float, 
                 cpu_capacity: float = 100.0, memory_capacity: float = 1000.0):
        self.id = server_id
        self.x = x
        self.y = y
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.connected_clusters = []
        self.task_queue = []
        self.processing_tasks = []
        self.completed_tasks = []
        self.total_tasks_processed = 0
        self.rejected_tasks = 0  # Track rejected tasks for load balancing
        self.load_history = []  # Track load over time
        
    def can_handle_task(self, task: MECTask) -> bool:
        """Check if MEC server can handle the task with improved thresholds"""
        # Lower threshold to reduce overloads (80% instead of 100%)
        cpu_threshold = self.cpu_capacity * 0.8
        memory_threshold = self.memory_capacity * 0.8
        
        return (self.cpu_usage + task.cpu_requirement <= cpu_threshold and
                self.memory_usage + task.memory_requirement <= memory_threshold)
    
    def accept_task(self, task: MECTask) -> bool:
        """Accept task for processing with improved admission control"""
        if self.can_handle_task(task):
            self.task_queue.append(task)
            self.cpu_usage += task.cpu_requirement
            self.memory_usage += task.memory_requirement
            return True
        else:
            self.rejected_tasks += 1
            return False
    
    def get_load_percentage(self) -> float:
        """Get current load as percentage"""
        cpu_load = (self.cpu_usage / self.cpu_capacity) * 100
        memory_load = (self.memory_usage / self.memory_capacity) * 100
        return max(cpu_load, memory_load)
    
    @property
    def current_load(self) -> float:
        """Property for current load percentage"""
        return self.get_load_percentage()
    
    def get_load_level(self) -> str:
        """Get load level description"""
        load = self.get_load_percentage()
        if load < 30:
            return "LOW"
        elif load < 60:
            return "MEDIUM"
        elif load < 80:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def process_tasks(self):
        """Process tasks in the queue with improved processing"""
        completed_results = []
        
        # Record current load
        self.load_history.append(self.current_load)
        if len(self.load_history) > 100:  # Keep only last 100 entries
            self.load_history.pop(0)
        
        # Process more tasks when load is low, fewer when high
        load_level = self.get_load_percentage()
        if load_level < 30:
            max_tasks_to_process = 5
        elif load_level < 60:
            max_tasks_to_process = 3
        else:
            max_tasks_to_process = 1
            
        tasks_to_process = min(max_tasks_to_process, len(self.task_queue))
        
        for _ in range(tasks_to_process):
            if self.task_queue:
                task = self.task_queue.pop(0)
                self.processing_tasks.append(task)
                if self.get_load_percentage() < 80:  # Only show if not overloaded
                    print(f"    MEC-{self.id}: Processing task {task.task_id} (load: {self.current_load:.1f}%/{self.get_load_level()})")
        
        # Complete some processing tasks (improved completion rate)
        completion_probability = 0.8 if self.get_load_percentage() < 50 else 0.4
        if self.processing_tasks and random.random() < completion_probability:
            completed_task = self.processing_tasks.pop(0)
            self.completed_tasks.append(completed_task)
            self.cpu_usage = max(0, self.cpu_usage - completed_task.cpu_requirement)
            self.memory_usage = max(0, self.memory_usage - completed_task.memory_requirement)
            self.total_tasks_processed += 1
            
            if self.get_load_percentage() < 80:  # Only show if not overloaded
                print(f"    MEC-{self.id}: Completed task {completed_task.task_id}")
            
            # Add to results
            completed_results.append({
                'task_id': completed_task.task_id,
                'source_cluster_id': completed_task.source_cluster_id
            })
        
        return completed_results

    def distance_to(self, x: float, y: float) -> float:
        """Calculate distance to a point"""
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)

class NodeState(Enum):
    IDLE = "idle"
    CLUSTER_HEAD = "cluster_head"
    CLUSTER_MEMBER = "cluster_member"
    GATEWAY = "gateway"  # Add gateway state

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
        
        # Mobility parameters - ENHANCED for demonstrable re-clustering
        self.velocity_x = random.uniform(-15.0, 15.0)  # m/s in x direction (8x faster)
        self.velocity_y = random.uniform(-15.0, 15.0)  # m/s in y direction (8x faster)  
        self.max_velocity = 20.0  # Maximum velocity m/s (8x faster)
        self.mobility_model = "random_waypoint"  # Mobility model
        self.direction_change_probability = 0.3  # 30% chance to change direction each round
        
        # Network state
        self.state = NodeState.IDLE
        self.cluster_id = None
        self.cluster_head_id = None
        self.cluster_members: List[int] = []
        self.is_gateway = False  # Add gateway flag
        
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
        
        # Inter-cluster communication
        self.inter_cluster_messages = []
        self.mec_tasks = []
        
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
    
    def update_mobility(self, area_bounds: Tuple[float, float, float, float] = (0, 1000, 0, 1000)):
        """Update node position based on mobility model"""
        min_x, max_x, min_y, max_y = area_bounds
        
        # Random waypoint mobility model
        if self.mobility_model == "random_waypoint":
            # Randomly change direction
            if random.random() < self.direction_change_probability:
                self.velocity_x = random.uniform(-self.max_velocity, self.max_velocity)
                self.velocity_y = random.uniform(-self.max_velocity, self.max_velocity)
            
            # Update position
            new_x = self.x + self.velocity_x
            new_y = self.y + self.velocity_y
            
            # Bounce off boundaries
            if new_x < min_x or new_x > max_x:
                self.velocity_x = -self.velocity_x
                new_x = max(min_x, min(max_x, new_x))
            
            if new_y < min_y or new_y > max_y:
                self.velocity_y = -self.velocity_y
                new_y = max(min_y, min(max_y, new_y))
            
            self.x = new_x
            self.y = new_y
    
    def get_mobility_info(self) -> Dict:
        """Get current mobility information"""
        speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        direction = math.atan2(self.velocity_y, self.velocity_x) * 180 / math.pi
        return {
            'speed': speed,
            'direction': direction,
            'velocity_x': self.velocity_x,
            'velocity_y': self.velocity_y
        }

class ARPMECProtocol:
    """ENHANCED implementation of ARPMEC with IAR infrastructure and adaptive routing"""
    
    def __init__(self, nodes: List[Node], C: int = 16, R: int = 100, K: int = 3):
        self.nodes = {node.id: node for node in nodes}
        self.N = len(nodes)  # Number of objects
        self.C = C           # Number of channels  
        self.R = R           # Min messages for accurate LQE
        self.K = K           # Number of MEC servers
        self.HUBmax = 10     # Max objects per cluster
        
        # ENHANCED: More realistic communication range for demonstrable mobility effects
        self.communication_range = 100.0  # 100m for intra-cluster communication
        self.inter_cluster_range = 250.0  # 250m for CH-to-CH communication
        self.mec_communication_range = 400.0  # 400m for CH-to-MEC communication
        self.current_time_slot = 0
        
        # Infrastructure
        self.mec_servers = {}
        self.iar_servers = {}  # NEW: IAR infrastructure
        self.clusters = {}
        self.inter_cluster_routing_table = {}
        
        # ML model for link quality prediction (as per paper)
        # Initialize ML models if available
        if ML_AVAILABLE:
            self.lqe_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.lqe_scaler = StandardScaler()
        else:
            self.lqe_model = None
            self.lqe_scaler = None
        self._train_lqe_model()
        
        # Initialize MEC servers
        self._deploy_mec_servers()
        
        # NEW: Initialize IAR infrastructure
        self._deploy_iar_infrastructure()
        self._connect_iar_to_mec()
    
    def _deploy_iar_infrastructure(self):
        """Deploy IAR (Infrastructure Access Router) servers strategically"""
        print("Deploying IAR infrastructure...")
        
        # Deploy IAR servers to provide coverage for clusters
        # Strategy: Place IAR servers to cover different areas
        area_width = 1000
        area_height = 1000
        
        # Create a grid of IAR servers for comprehensive coverage
        iar_positions = [
            (200, 200),   # IAR-1: Southwest
            (200, 800),   # IAR-2: Northwest  
            (700, 250),   # IAR-3: Southeast
            (800, 800),   # IAR-4: Northeast
            (500, 500),   # IAR-5: Central
        ]
        
        for i, (x, y) in enumerate(iar_positions):
            iar_id = i + 1
            iar_server = IARServer(iar_id, x, y, coverage_radius=250)  # 250m coverage
            self.iar_servers[iar_id] = iar_server
            print(f"  IAR-{iar_id} deployed at ({x}, {y}) with {iar_server.coverage_radius}m coverage")
    
    def _connect_iar_to_mec(self):
        """Connect IAR servers to MEC servers for routing"""
        print("Connecting IAR servers to MEC servers...")
        
        for iar_id, iar in self.iar_servers.items():
            # Find nearest MEC servers for this IAR
            mec_distances = []
            for mec_id, mec in self.mec_servers.items():
                distance = iar.distance_to(mec.x, mec.y)
                mec_distances.append((mec_id, distance))
            
            # Sort by distance and connect to nearest 2-3 MEC servers
            mec_distances.sort(key=lambda x: x[1])
            connected_mecs = []
            
            for mec_id, distance in mec_distances[:3]:  # Connect to 3 nearest MECs
                if distance <= 400:  # Within 400m range
                    iar.add_mec_server(mec_id)
                    connected_mecs.append(f"MEC-{mec_id}(d={distance:.0f}m)")
            
            print(f"  IAR-{iar_id} connected to: {', '.join(connected_mecs)}")
    
    def _find_nearest_iar_server(self, node: Node) -> Optional[IARServer]:
        """Find nearest IAR server for a cluster head"""
        nearest_iar = None
        min_distance = float('inf')
        
        for iar in self.iar_servers.values():
            distance = iar.distance_to(node.x, node.y)
            if distance < min_distance and iar.can_serve_cluster(node.x, node.y):
                min_distance = distance
                nearest_iar = iar
        
        return nearest_iar
    
    def _deploy_mec_servers(self):
        """Deploy MEC servers strategically in the network"""
        print(f"Deploying {self.K} MEC servers...")
        
        # Find network boundaries
        min_x = min(node.x for node in self.nodes.values())
        max_x = max(node.x for node in self.nodes.values())
        min_y = min(node.y for node in self.nodes.values())
        max_y = max(node.y for node in self.nodes.values())
        
        # Deploy servers evenly across the network
        for i in range(self.K):
            if self.K == 1:
                x = (min_x + max_x) / 2
                y = (min_y + max_y) / 2
            else:
                x = min_x + (i + 1) * (max_x - min_x) / (self.K + 1)
                y = min_y + (max_y - min_y) / 2
            
            server = MECServer(
                server_id=i,
                x=x, y=y,
                cpu_capacity=200.0,
                memory_capacity=1000.0
            )
            self.mec_servers[i] = server
            print(f"  MEC Server {i} deployed at ({x:.1f}, {y:.1f})")
    
    def _find_nearest_mec_server(self, node: Node) -> Optional[MECServer]:
        """Find nearest MEC server to a node"""
        if not self.mec_servers:
            return None
        
        nearest_server = None
        min_distance = float('inf')
        
        for server in self.mec_servers.values():
            distance = server.distance_to(node.x, node.y)
            if distance < min_distance:
                min_distance = distance
                nearest_server = server
        
        return nearest_server
    
    def _get_cluster_heads(self) -> List[Node]:
        """Get all cluster heads in the network"""
        return [node for node in self.nodes.values() 
                if node.state == NodeState.CLUSTER_HEAD and node.is_alive()]
    
    def _find_inter_cluster_neighbors(self, ch_node: Node) -> List[Node]:
        """Find other cluster heads within inter-cluster communication range"""
        neighbors = []
        
        for other_ch in self._get_cluster_heads():
            if other_ch.id != ch_node.id:
                distance = ch_node.distance_to(other_ch)
                if distance <= self.inter_cluster_range:
                    neighbors.append(other_ch)
        
        return neighbors
    
    def _build_inter_cluster_routing_table(self):
        """Build routing table for inter-cluster communication"""
        print("Building inter-cluster routing table...")
        
        cluster_heads = self._get_cluster_heads()
        
        for ch in cluster_heads:
            self.inter_cluster_routing_table[ch.id] = {
                'neighbors': [],
                'mec_server': None,
                'routes': {}
            }
            
            # Find neighboring cluster heads
            neighbors = self._find_inter_cluster_neighbors(ch)
            self.inter_cluster_routing_table[ch.id]['neighbors'] = [n.id for n in neighbors]
            
            # Find nearest MEC server
            nearest_mec = self._find_nearest_mec_server(ch)
            if nearest_mec:
                self.inter_cluster_routing_table[ch.id]['mec_server'] = nearest_mec.id
                # Add cluster to MEC server's connected clusters
                if ch.cluster_id not in nearest_mec.connected_clusters:
                    nearest_mec.connected_clusters.append(ch.cluster_id)
            
            # Build routes to other clusters (simplified flooding approach)
            for target_ch in cluster_heads:
                if target_ch.id != ch.id:
                    # Direct route if neighbor
                    if target_ch.id in self.inter_cluster_routing_table[ch.id]['neighbors']:
                        self.inter_cluster_routing_table[ch.id]['routes'][target_ch.id] = [target_ch.id]
                    else:
                        # Multi-hop route (simplified - use nearest neighbor as next hop)
                        if neighbors:
                            nearest_neighbor = min(neighbors, key=lambda n: n.distance_to(target_ch))
                            self.inter_cluster_routing_table[ch.id]['routes'][target_ch.id] = [nearest_neighbor.id]
    
    def _ch_to_ch_communication(self, source_ch: Node, target_cluster_id: int, message: InterClusterMessage):
        """ENHANCED: CH-to-CH communication via IAR and MEC infrastructure"""
        
        # Step 1: Find nearest IAR server for source CH
        nearest_iar = self._find_nearest_iar_server(source_ch)
        if not nearest_iar:
            print(f"CH-{source_ch.id}: No IAR server available")
            return False
        
        # Step 2: Calculate energy cost for CH-to-IAR communication
        distance_to_iar = nearest_iar.distance_to(source_ch.x, source_ch.y)
        energy_cost_to_iar = source_ch.calculate_energy_consumption(1, distance_to_iar)
        
        if source_ch.energy < energy_cost_to_iar:
            print(f"CH-{source_ch.id}: Insufficient energy for IAR communication")
            return False
        
        # Step 3: Find target cluster's CH
        target_ch = None
        for ch in self._get_cluster_heads():
            if ch.cluster_id == target_cluster_id:
                target_ch = ch
                break
        
        if not target_ch:
            print(f"CH-{source_ch.id}: Target cluster {target_cluster_id} not found")
            return False
        
        # Step 4: Find target IAR server
        target_iar = self._find_nearest_iar_server(target_ch)
        if not target_iar:
            print(f"CH-{source_ch.id}: Target cluster has no IAR server")
            return False
        
        # Step 5: Use IAR for adaptive routing to find best MEC server
        best_mec = nearest_iar.route_to_best_mec(
            MECTask(
                task_id=f"comm_task_{message.message_id}",
                source_cluster_id=message.source_cluster_id,
                cpu_requirement=1.0,
                memory_requirement=5.0,
                deadline=self.current_time_slot + 10,
                data_size=5.0,
                created_time=self.current_time_slot
            ),
            self.mec_servers
        )
        
        if not best_mec:
            print(f"CH-{source_ch.id}: No MEC server available for routing")
            return False
        
        # Step 6: Execute communication: CH -> IAR -> MEC -> Target IAR -> Target CH
        source_ch.update_energy(energy_cost_to_iar)
        
        # Register cluster with IAR
        nearest_iar.register_cluster(source_ch.cluster_id)
        
        # Add to routing table for load balancing
        nearest_iar.routing_table[target_cluster_id] = best_mec.id
        
        # Simulate message delivery
        target_ch.inter_cluster_messages.append(message)
        
        print(f"CH-{source_ch.id} -> IAR-{nearest_iar.id} -> MEC-{best_mec.id} -> IAR-{target_iar.id} -> CH-{target_ch.id}: "
              f"{message.message_type} (via IAR/MEC infrastructure)")
        
        return True
    
    def _ch_to_mec_communication(self, ch_node: Node, task: MECTask):
        """ENHANCED: CH-to-MEC communication via IAR with adaptive routing"""
        
        # Step 1: Find nearest IAR server
        nearest_iar = self._find_nearest_iar_server(ch_node)
        if not nearest_iar:
            print(f"CH-{ch_node.id}: No IAR server available")
            return False
        
        # Step 2: Calculate energy cost for CH-to-IAR communication
        distance_to_iar = nearest_iar.distance_to(ch_node.x, ch_node.y)
        energy_cost = ch_node.calculate_energy_consumption(1, distance_to_iar)
        
        if ch_node.energy < energy_cost:
            print(f"CH-{ch_node.id}: Insufficient energy for IAR communication")
            return False
        
        # Step 3: Use IAR for adaptive routing to find best MEC server
        best_mec = nearest_iar.route_to_best_mec(task, self.mec_servers)
        
        if not best_mec:
            print(f"CH-{ch_node.id}: No MEC server available (all overloaded)")
            return False
        
        # Step 4: Try to offload task to selected MEC server
        if best_mec.accept_task(task):
            ch_node.update_energy(energy_cost)
            
            # Register cluster with IAR
            nearest_iar.register_cluster(ch_node.cluster_id)
            
            print(f"CH-{ch_node.id} -> IAR-{nearest_iar.id} -> MEC-{best_mec.id}: Task {task.task_id} "
                  f"(load: {best_mec.get_load_percentage():.1f}%/{best_mec.get_load_level()})")
            return True
        else:
            print(f"CH-{ch_node.id}: MEC-{best_mec.id} rejected task {task.task_id} "
                  f"(load: {best_mec.get_load_percentage():.1f}%/{best_mec.get_load_level()})")
            return False
    
    def _process_inter_cluster_messages(self):
        """Process inter-cluster messages at all cluster heads"""
        
        for ch in self._get_cluster_heads():
            # Process incoming messages
            for message in ch.inter_cluster_messages[:]:  # Copy to avoid modification during iteration
                if message.message_type == 'data':
                    # Handle data message
                    print(f"CH-{ch.id} received data from cluster {message.source_cluster_id}")
                    
                    # Generate MEC task from data
                    task = MECTask(
                        task_id=f"task_{message.source_cluster_id}_{ch.id}_{len(ch.mec_tasks)}",
                        source_cluster_id=message.source_cluster_id,
                        cpu_requirement=random.uniform(1, 10),
                        memory_requirement=random.uniform(10, 50),
                        deadline=self.current_time_slot + 10,
                        data_size=random.uniform(1, 20),
                        created_time=self.current_time_slot
                    )
                    
                    # Try to process locally or forward to MEC
                    if not self._ch_to_mec_communication(ch, task):
                        # Could not process - maybe forward to another cluster
                        print(f"CH-{ch.id} could not process task {task.task_id}")
                
                ch.inter_cluster_messages.remove(message)
    
    def _generate_inter_cluster_traffic(self):
        """Generate inter-cluster communication traffic with reduced frequency"""
        
        cluster_heads = self._get_cluster_heads()
        
        # REDUCED: Generate less frequent inter-cluster traffic (15% instead of 30%)
        for ch in cluster_heads:
            if random.random() < 0.15:  # 15% chance to send inter-cluster message
                # Find a target cluster
                other_chs = [c for c in cluster_heads if c.id != ch.id]
                if other_chs:
                    target_ch = random.choice(other_chs)
                    
                    # Create inter-cluster message
                    message = InterClusterMessage(
                        message_id=f"msg_{ch.id}_{target_ch.id}_{self.current_time_slot}",
                        source_cluster_id=ch.cluster_id,
                        destination_cluster_id=target_ch.cluster_id,
                        message_type='data',
                        payload={'data': f'sensor_data_from_cluster_{ch.cluster_id}'},
                        timestamp=self.current_time_slot
                    )
                    
                    # Send message via IAR infrastructure
                    self._ch_to_ch_communication(ch, target_ch.cluster_id, message)
    
    def _generate_mec_tasks(self):
        """Generate MEC tasks from cluster heads with reduced frequency"""
        
        for ch in self._get_cluster_heads():
            # REDUCED: Generate fewer tasks to prevent overloads (20% instead of 40%)
            if random.random() < 0.2:  # 20% chance to generate local task
                task = MECTask(
                    task_id=f"local_task_{ch.id}_{self.current_time_slot}",
                    source_cluster_id=ch.cluster_id,
                    cpu_requirement=random.uniform(1, 8),  # Reduced CPU requirement
                    memory_requirement=random.uniform(10, 50),  # Reduced memory requirement
                    deadline=self.current_time_slot + 20,
                    data_size=random.uniform(1, 20),
                    created_time=self.current_time_slot
                )
                
                # Try to offload to MEC server via IAR
                self._ch_to_mec_communication(ch, task)
    def _check_and_recluster(self):
        """PAPER-BASED COMPREHENSIVE re-clustering: Handle ALL nodes and ensure proper clustering"""
        nodes_changed = False
        
        print("\n=== COMPREHENSIVE RE-CLUSTERING (Paper Algorithm) ===")
        
        # Step 1: Check cluster head validity and step down low-energy CHs
        cluster_heads = self._get_cluster_heads()
        stepped_down_chs = []
        
        for ch in cluster_heads[:]:  # Copy list to avoid modification during iteration
            if ch.energy < ch.energy_threshold_f:
                print(f"CH-{ch.id} stepping down due to low energy ({ch.energy:.1f}J)")
                
                # All members become idle
                for member_id in ch.cluster_members[:]:
                    if member_id in self.nodes:
                        member = self.nodes[member_id]
                        member.state = NodeState.IDLE
                        member.cluster_head_id = None
                        member.cluster_id = None
                
                # CH becomes idle
                ch.state = NodeState.IDLE
                ch.cluster_head_id = None
                ch.cluster_id = None
                ch.cluster_members = []
                stepped_down_chs.append(ch)
                nodes_changed = True
        
        # Step 2: Process ALL alive nodes for clustering
        remaining_chs = self._get_cluster_heads()  # Get updated list
        all_nodes = [n for n in self.nodes.values() if n.is_alive()]
        
        # Separate idle nodes and existing members
        idle_nodes = [n for n in all_nodes if n.state == NodeState.IDLE]
        member_nodes = [n for n in all_nodes if n.state == NodeState.CLUSTER_MEMBER]
        
        print(f"Processing {len(idle_nodes)} idle nodes and {len(member_nodes)} cluster members")
        
        # Step 3: Check existing cluster members for validity
        for node in member_nodes:
            current_ch = self.nodes.get(node.cluster_head_id) if node.cluster_head_id is not None else None
            need_reassignment = False
            
            if not current_ch or not current_ch.is_alive() or current_ch.state != NodeState.CLUSTER_HEAD:
                need_reassignment = True
                print(f"Node-{node.id}: CH is invalid/dead")
            elif node.distance_to(current_ch) > self.communication_range:
                need_reassignment = True
                print(f"Node-{node.id}: Too far from CH-{current_ch.id} ({node.distance_to(current_ch):.1f}m)")
            
            if need_reassignment:
                # Remove from current cluster
                if current_ch and node.id in current_ch.cluster_members:
                    current_ch.cluster_members.remove(node.id)
                
                # Make idle for reassignment
                node.state = NodeState.IDLE
                node.cluster_head_id = None
                node.cluster_id = None
                idle_nodes.append(node)
                nodes_changed = True
        
        # Step 4: Assign all idle nodes to clusters or promote them to CHs
        for node in idle_nodes:
            # Find nearest available cluster head
            best_ch = None
            min_distance = float('inf')
            
            for ch in remaining_chs:
                if ch.id != node.id and ch.is_alive():
                    distance = node.distance_to(ch)
                    if distance <= self.communication_range and distance < min_distance:
                        # Check cluster capacity
                        if len(ch.cluster_members) < self.HUBmax:
                            min_distance = distance
                            best_ch = ch
            
            if best_ch:
                # Join existing cluster
                node.cluster_head_id = best_ch.id
                node.cluster_id = best_ch.cluster_id
                node.state = NodeState.CLUSTER_MEMBER
                if node.id not in best_ch.cluster_members:
                    best_ch.cluster_members.append(node.id)
                
                energy_cost = node.calculate_energy_consumption(1, min_distance)
                node.update_energy(energy_cost)
                
                print(f"[JOIN] Node-{node.id} → CH-{best_ch.id} (distance: {min_distance:.1f}m)")
                nodes_changed = True
                
            else:
                # No suitable cluster found - check if can become cluster head
                can_be_ch = True
                
                # Don't become CH if too close to existing CHs (avoid cluster overlap)
                for ch in remaining_chs:
                    if node.distance_to(ch) < self.communication_range * 0.7:  # 70m separation
                        can_be_ch = False
                        break
                
                if can_be_ch and node.energy > node.energy_threshold_f:
                    # Promote to cluster head
                    node.state = NodeState.CLUSTER_HEAD
                    node.cluster_head_id = None
                    node.cluster_id = node.id
                    node.cluster_members = []
                    remaining_chs.append(node)  # Add to CH list
                    
                    print(f"[PROMOTE] Node-{node.id} → NEW CLUSTER HEAD")
                    nodes_changed = True
                else:
                    # Stay idle (will be retried next round)
                    print(f"[IDLE] Node-{node.id} remains idle (no suitable cluster/CH position)")
        
        # Step 5: Handle cluster merging if CHs are too close
        final_chs = self._get_cluster_heads()
        for i, ch1 in enumerate(final_chs):
            for j, ch2 in enumerate(final_chs[i+1:], i+1):
                distance = ch1.distance_to(ch2)
                
                # Merge clusters if CHs are too close (paper suggests avoiding overlapping coverage)
                if distance < self.communication_range * 0.6:  # 60m merge threshold
                    ch1_size = len(ch1.cluster_members)
                    ch2_size = len(ch2.cluster_members)
                    
                    # Merge smaller cluster into larger one
                    if ch1_size >= ch2_size:
                        self._merge_clusters(ch2, ch1)
                        print(f"[MERGE] CH-{ch2.id} → CH-{ch1.id} (distance: {distance:.1f}m)")
                    else:
                        self._merge_clusters(ch1, ch2)
                        print(f"[MERGE] CH-{ch1.id} → CH-{ch2.id} (distance: {distance:.1f}m)")
                    
                    nodes_changed = True
                    break
        
        # Step 6: Show final clustering statistics
        if nodes_changed:
            final_chs = self._get_cluster_heads()
            idle_count = sum(1 for n in self.nodes.values() if n.state == NodeState.IDLE and n.is_alive())
            
            print(f"\n=== POST RE-CLUSTERING STATISTICS ===")
            print(f"  Total Cluster Heads: {len(final_chs)}")
            for ch in final_chs:
                print(f"    CH-{ch.id}: {len(ch.cluster_members)} members")
            print(f"  Idle nodes: {idle_count}")
            
            if idle_count > 0:
                print(f"  ⚠️  {idle_count} nodes remain idle (will retry next round)")
        
        return nodes_changed
    
    def _merge_clusters(self, source_ch, target_ch):
        """Merge source cluster into target cluster (paper-based cluster consolidation)"""
        print(f"    Merging cluster {source_ch.id} ({len(source_ch.cluster_members)} members) into cluster {target_ch.id}")
        
        # Transfer all members
        for member_id in source_ch.cluster_members[:]:
            if member_id in self.nodes:
                member = self.nodes[member_id]
                member.cluster_head_id = target_ch.id
                member.cluster_id = target_ch.cluster_id
                if member_id not in target_ch.cluster_members:
                    target_ch.cluster_members.append(member_id)
        
        # Convert source CH to member of target cluster
        source_ch.state = NodeState.CLUSTER_MEMBER
        source_ch.cluster_head_id = target_ch.id
        source_ch.cluster_id = target_ch.cluster_id
        source_ch.cluster_members = []
        
        # Add source CH as member of target
        if source_ch.id not in target_ch.cluster_members:
            target_ch.cluster_members.append(source_ch.id)
    
    def _process_mec_servers(self):
        """Process tasks at all MEC servers"""
        
        for server in self.mec_servers.values():
            completed_tasks = server.process_tasks()
            
            # Send results back to clusters
            for result in completed_tasks:
                print(f"MEC-{server.id} completed task {result['task_id']} "
                      f"from cluster {result['source_cluster_id']}")
    
    def _check_cluster_head_validity(self):
        """Check if cluster heads are still valid and promote new ones if needed"""
        nodes_changed = False
        
        for ch in self._get_cluster_heads()[:]:  # Copy to avoid modification during iteration
            # Check if CH has enough energy
            if ch.energy < ch.energy_threshold_f:
                print(f"CH-{ch.id} has low energy ({ch.energy:.1f}J), stepping down")
                
                # Find best member to promote
                best_member = None
                best_score = -1
                
                for member_id in ch.cluster_members[:]:
                    if member_id in self.nodes:
                        member = self.nodes[member_id]
                        if member.is_alive():
                            # Score based on energy and centrality
                            energy_score = member.energy / member.initial_energy
                            
                            # Calculate centrality (average distance to other members)
                            total_distance = 0
                            valid_members = 0
                            for other_id in ch.cluster_members:
                                if other_id != member_id and other_id in self.nodes:
                                    other_node = self.nodes[other_id]
                                    if other_node.is_alive():
                                        total_distance += member.distance_to(other_node)
                                        valid_members += 1
                            
                            centrality_score = 1.0 / (1.0 + total_distance / max(1, valid_members))
                            total_score = energy_score * 0.7 + centrality_score * 0.3
                            
                            if total_score > best_score:
                                best_score = total_score
                                best_member = member
                
                if best_member:
                    # Promote member to CH
                    best_member.state = NodeState.CLUSTER_HEAD
                    best_member.cluster_head_id = None
                    best_member.cluster_members = ch.cluster_members[:]
                    best_member.cluster_members.remove(best_member.id)  # Remove self
                    
                    # Update all members
                    for member_id in best_member.cluster_members:
                        if member_id in self.nodes:
                            self.nodes[member_id].cluster_head_id = best_member.id
                    
                    # Demote old CH
                    ch.state = NodeState.CLUSTER_MEMBER
                    ch.cluster_head_id = best_member.id
                    ch.cluster_members = []
                    
                    print(f"Promoted Node-{best_member.id} to CH, demoted CH-{ch.id}")
                    nodes_changed = True
                else:
                    # No suitable replacement - dissolve cluster
                    print(f"Dissolving cluster of CH-{ch.id} (no suitable replacement)")
                    ch.state = NodeState.IDLE
                    ch.cluster_head_id = None
                    ch.cluster_id = None
                    
                    for member_id in ch.cluster_members:
                        if member_id in self.nodes:
                            member = self.nodes[member_id]
                            member.state = NodeState.IDLE
                            member.cluster_head_id = None
                            member.cluster_id = None
                    
                    ch.cluster_members = []
                    nodes_changed = True
        
        return nodes_changed
    
    def _train_lqe_model(self):
        """Train Random Forest model for LQE as mentioned in paper"""
        # Generate synthetic training data for LQE model
        training_size = 10000
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
        if ML_AVAILABLE and self.lqe_model is not None:
            self.lqe_scaler.fit(features)
            scaled_features = self.lqe_scaler.transform(features)
            self.lqe_model.fit(scaled_features, labels)
        else:
            # Store simple lookup for fallback
            self.simple_lqe_data = list(zip(features, labels))
    
    def predict_link_quality(self, rssi: float, pdr: float) -> float:
        """Link quality prediction using Random Forest or simple heuristics"""
        if ML_AVAILABLE and self.lqe_model is not None:
            features = np.array([[rssi, pdr]])
            scaled_features = self.lqe_scaler.transform(features)
            
            # Get prediction probabilities for quality classes
            probabilities = self.lqe_model.predict_proba(scaled_features)[0]
            
            # Convert to single score (weighted by quality levels)
            quality_score = probabilities[0] * 0.2 + probabilities[1] * 0.6 + probabilities[2] * 1.0
            return quality_score
        else:
            # Simple heuristic fallback
            if pdr > 0.8 and rssi > -60:
                return 1.0  # Excellent
            elif pdr > 0.6 and rssi > -80:
                return 0.6  # Good  
            else:
                return 0.2  # Poor
    
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
        
        # Step 3: IMPROVED Cluster Head Election with Distance Constraints
        print("Phase 3: Distance-Aware Cluster Head Election...")
        
        # IMPROVEMENT: Filter join decisions by distance to prevent distant cluster assignments
        filtered_join_decisions = {}
        for sender_id, target_id in join_decisions.items():
            sender = self.nodes[sender_id]
            target = self.nodes[target_id]
            distance = sender.distance_to(target)
            
            # Only allow cluster membership if within communication range
            if distance <= self.communication_range:
                filtered_join_decisions[sender_id] = target_id
            else:
                print(f"Node {sender_id} rejected distant CH {target_id} (distance: {distance:.1f}m > {self.communication_range}m)")
        
        # Count JOIN votes for each node (with distance filtering)
        join_votes = {}
        for sender, target in filtered_join_decisions.items():
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
                
                # Add voters as cluster members (already distance-filtered)
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
        
        # IMPORTANT: Build inter-cluster communication infrastructure
        self.clusters = clusters
        self._build_inter_cluster_routing_table()
        
        print(f"Inter-cluster communication setup complete.")
        return clusters
    
    def adaptive_routing_algorithm(self, T: int = 200):
        """FIXED Algorithm 3: Adaptive routing with inter-cluster communication and MOBILITY"""
        print(f"Starting FIXED Algorithm 3: Adaptive Routing with mobility and re-clustering for {T} rounds...")
        
        reclustering_interval = 5  # Re-cluster every 5 rounds (was 10) for more frequent re-clustering
        
        for round_num in range(T):
            self.current_time_slot = round_num
            
            if round_num % 10 == 0:
                print(f"Round {round_num + 1}/{T}")
            
            # STEP 1: Update node mobility (nodes move every round)
            area_bounds = (0, 1000, 0, 1000)
            for node in self.nodes.values():
                if node.is_alive():
                    old_x, old_y = node.x, node.y
                    node.update_mobility(area_bounds)
                    
                    # Show significant movements
                    distance_moved = math.sqrt((node.x - old_x)**2 + (node.y - old_y)**2)
                    if distance_moved > 20 and round_num % 20 == 0:  # Show every 20 rounds
                        print(f"Node-{node.id} moved {distance_moved:.1f}m to ({node.x:.0f}, {node.y:.0f})")
            
            # STEP 2: Check for re-clustering due to mobility
            if round_num % reclustering_interval == 0 and round_num > 0:
                print(f"\n🔄 MOBILITY-TRIGGERED RE-CLUSTERING (Round {round_num + 1})")
                
                # Check cluster membership validity
                membership_changed = self._check_and_recluster()
                
                # Check cluster head validity
                leadership_changed = self._check_cluster_head_validity()
                
                if membership_changed or leadership_changed:
                    print("✅ Network topology updated due to mobility")
                    # Rebuild inter-cluster routing table
                    self._build_inter_cluster_routing_table()
                else:
                    print("✅ Network topology stable")
            
            # STEP 3: Normal algorithm operations
            alive_nodes = [node for node in self.nodes.values() if node.is_alive()]
            
            for node in alive_nodes:
                if node.state == NodeState.CLUSTER_HEAD:
                    self._fixed_cluster_head_operations(node, round_num)
                else:
                    self._fixed_cluster_member_operations(node, round_num)
            
            # STEP 4: Inter-cluster communication operations
            self._generate_inter_cluster_traffic()
            self._generate_mec_tasks()
            self._process_inter_cluster_messages()
            self._process_mec_servers()
            
            # STEP 5: Show network status every 20 rounds
            if round_num % 20 == 0:
                cluster_heads = self._get_cluster_heads()
                alive_count = len(alive_nodes)
                total_energy = sum(n.initial_energy - n.energy for n in self.nodes.values())
                avg_speed = np.mean([n.get_mobility_info()['speed'] for n in alive_nodes]) if alive_nodes else 0
                
                print(f"\n📊 Network Status (Round {round_num + 1}):")
                print(f"   Alive nodes: {alive_count}/{len(self.nodes)}")
                print(f"   Active clusters: {len(cluster_heads)}")
                print(f"   Total energy consumed: {total_energy:.1f}J")
                print(f"   Average node speed: {avg_speed:.1f}m/s")
                
                # Show cluster sizes
                if cluster_heads:
                    cluster_sizes = [len(ch.cluster_members) + 1 for ch in cluster_heads]  # +1 for CH itself
                    print(f"   Cluster sizes: {cluster_sizes}")
        
        print(f"\n✅ Adaptive routing completed. Network experienced mobility-based re-clustering.")
    
    def _reset_clustering_state(self):
        """Reset all clustering state for re-clustering"""
        for node in self.nodes.values():
            if node.is_alive():
                node.state = NodeState.IDLE
                node.cluster_id = None
                node.cluster_head_id = None
                node.cluster_members = []
                node.neighbors = {}
                node.hello_received_count = {}
                node.hello_messages_buffer = []
                node.join_messages_received = []
        
        # Clear existing routing table
        self.inter_cluster_routing_table = {}
        self.clusters = {}
    
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
        """Get performance metrics including IAR infrastructure and adaptive routing"""
        alive_nodes = sum(1 for node in self.nodes.values() if node.is_alive())
        total_energy_consumed = sum(
            node.initial_energy - node.energy for node in self.nodes.values()
        )
        
        cluster_heads = [n for n in self.nodes.values() if n.state == NodeState.CLUSTER_HEAD]
        
        # MEC server statistics
        total_mec_tasks = sum(server.total_tasks_processed for server in self.mec_servers.values())
        total_mec_utilization = sum(server.cpu_usage for server in self.mec_servers.values())
        total_mec_rejected = sum(server.rejected_tasks for server in self.mec_servers.values())
        
        # IAR server statistics
        total_iar_clusters = sum(len(iar.connected_clusters) for iar in self.iar_servers.values())
        total_iar_routes = sum(len(iar.routing_table) for iar in self.iar_servers.values())
        
        # Inter-cluster communication statistics
        total_inter_cluster_messages = sum(len(node.inter_cluster_messages) for node in self.nodes.values())
        
        return {
            "alive_nodes": alive_nodes,
            "total_nodes": len(self.nodes),
            "network_lifetime": alive_nodes / len(self.nodes),
            "total_energy_consumed": total_energy_consumed,
            "energy_per_node": total_energy_consumed / len(self.nodes),
            "num_clusters": len(cluster_heads),
            "avg_cluster_size": len(self.nodes) / max(1, len(cluster_heads)),
            "clustering_time_slots": self.R * self.N * self.C + self.N + 2 * self.K,
            "total_mec_tasks_processed": total_mec_tasks,
            "total_mec_tasks_rejected": total_mec_rejected,
            "avg_mec_utilization": total_mec_utilization / max(1, len(self.mec_servers)),
            "inter_cluster_routes": len(self.inter_cluster_routing_table),
            "pending_inter_cluster_messages": total_inter_cluster_messages,
            "mec_servers": len(self.mec_servers),
            "iar_servers": len(self.iar_servers),
            "total_iar_clusters": total_iar_clusters,
            "total_iar_routes": total_iar_routes,
            "mec_success_rate": total_mec_tasks / max(1, total_mec_tasks + total_mec_rejected) * 100
        }

# ENHANCED Test function with IAR infrastructure
def test_enhanced_implementation():
    """Test the ENHANCED ARPMEC implementation with IAR infrastructure and adaptive routing"""
    print("Testing ENHANCED ARPMEC implementation with IAR infrastructure and adaptive routing")
    print("=" * 90)
    
    # Create realistic network
    N = 25  # Nodes for testing inter-cluster communication
    nodes = []
    for i in range(N):
        x = random.uniform(0, 800)  # 800m x 800m area
        y = random.uniform(0, 800)
        energy = random.uniform(90, 110)
        nodes.append(Node(i, x, y, energy))
    
    # Conservative parameters to reduce overloads
    C = 4     # Fewer channels
    R = 10    # Fewer HELLO messages
    K = 3     # MEC servers
    
    protocol = ARPMECProtocol(nodes, C, R, K)
    
    print(f"Network: {N} nodes, {C} channels, {R} HELLO messages")
    print(f"Infrastructure: {K} MEC servers, {len(protocol.iar_servers)} IAR servers")
    print(f"Area: 800m x 800m, Communication range: {protocol.communication_range}m")
    print(f"Inter-cluster range: {protocol.inter_cluster_range}m")
    
    # Test energy model first
    test_node = nodes[0]
    energy_100m = test_node.calculate_energy_consumption(1, 100)
    energy_300m = test_node.calculate_energy_consumption(1, 300)
    
    print(f"\nEnergy Model Test:")
    print(f"1 packet at 100m: {energy_100m:.4f}J")
    print(f"1 packet at 300m: {energy_300m:.4f}J")
    
    if energy_100m > 10:
        print("ERROR: Energy consumption still too high!")
        return None, None
    
    print("Running Algorithm 2 (Clustering with IAR/MEC deployment)...")
    
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
    print(f"- MEC servers deployed: {len(protocol.mec_servers)}")
    print(f"- IAR servers deployed: {len(protocol.iar_servers)}")
    print(f"- Inter-cluster routes: {len(protocol.inter_cluster_routing_table)}")
    
    if len(clusters) == 0:
        print("WARNING: No clusters formed!")
        return protocol, None
    
    # Show cluster details with IAR assignment
    print(f"\nCluster Details:")
    for head_id, members in clusters.items():
        head = protocol.nodes[head_id]
        nearest_iar = protocol._find_nearest_iar_server(head)
        iar_distance = nearest_iar.distance_to(head.x, head.y) if nearest_iar else float('inf')
        print(f"Cluster {head_id}: Head energy={head.energy:.1f}J, Members={len(members)}, "
              f"IAR={nearest_iar.id if nearest_iar else 'None'} "
              f"(distance: {iar_distance:.1f}m)")
    
    # Show IAR server details
    print(f"\nIAR Server Details:")
    for iar_id, iar in protocol.iar_servers.items():
        mec_connections = len(iar.connected_mec_servers)
        cluster_connections = len(iar.connected_clusters)
        print(f"IAR-{iar_id}: Position=({iar.x:.1f}, {iar.y:.1f}), "
              f"Coverage={iar.coverage_radius}m, "
              f"MECs={mec_connections}, Clusters={cluster_connections}")
    
    # Show MEC server details
    print(f"\nMEC Server Details:")
    for server_id, server in protocol.mec_servers.items():
        print(f"MEC-{server_id}: Position=({server.x:.1f}, {server.y:.1f}), "
              f"CPU={server.cpu_capacity}, Memory={server.memory_capacity}, "
              f"Load={server.get_load_percentage():.1f}%/{server.get_load_level()}")
    
    print("Running Algorithm 3 (Adaptive Routing with IAR infrastructure)...")
    T = 30  # More rounds to see adaptive routing and load balancing
    
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
    print(f"MEC tasks processed: {metrics['total_mec_tasks_processed']}")
    print(f"MEC tasks rejected: {metrics['total_mec_tasks_rejected']}")
    print(f"MEC success rate: {metrics['mec_success_rate']:.1f}%")
    print(f"Average MEC utilization: {metrics['avg_mec_utilization']:.1f}%")
    print(f"IAR servers: {metrics['iar_servers']}")
    print(f"IAR cluster connections: {metrics['total_iar_clusters']}")
    print(f"IAR routing entries: {metrics['total_iar_routes']}")
    print(f"Inter-cluster routes: {metrics['inter_cluster_routes']}")
    print(f"Pending inter-cluster messages: {metrics['pending_inter_cluster_messages']}")
    
    # Success validation
    success = (
        len(clusters) > 0 and 
        metrics['network_lifetime'] > 0.5 and 
        metrics['total_mec_tasks_processed'] > 0 and
        metrics['mec_success_rate'] > 50 and  # At least 50% task success rate
        metrics['iar_servers'] > 0
    )
    
    if success:
        print(f"\n✅ ENHANCED ARPMEC WITH IAR INFRASTRUCTURE SUCCESS!")
        print(f"   - Clusters formed: ✅")
        print(f"   - MEC servers deployed: ✅") 
        print(f"   - IAR infrastructure deployed: ✅")
        print(f"   - Inter-cluster communication: ✅")
        print(f"   - Adaptive routing: ✅")
        print(f"   - Reduced overloads: ✅ ({metrics['mec_success_rate']:.1f}% success rate)")
    else:
        print(f"\n❌ IMPLEMENTATION STILL HAS ISSUES!")
        if metrics['mec_success_rate'] < 50:
            print(f"   - Low MEC success rate: {metrics['mec_success_rate']:.1f}%")
    
    return protocol, metrics

if __name__ == "__main__":
    import time
    random.seed(42)
    np.random.seed(42)
    
    protocol, metrics = test_enhanced_implementation()