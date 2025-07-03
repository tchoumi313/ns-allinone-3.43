#!/usr/bin/env python3
"""
Real ARPMEC Network Simulation
Implements proper discrete-event simulation with:
- Message passing between nodes
- IP addressing and routing
- MEC servers and gateways
- Time-driven continuous re-clustering
- Distributed node behavior
"""

import heapq
import ipaddress
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# MESSAGE SYSTEM - Real network messages
# ============================================================================

class MessageType(Enum):
    HELLO = "HELLO"
    JOIN = "JOIN"
    DATA = "DATA"
    CLUSTER_HEAD_ANNOUNCE = "CH_ANNOUNCE"
    CLUSTER_MEMBER_UPDATE = "CM_UPDATE"
    MEC_REGISTER = "MEC_REGISTER"
    MEC_DATA_FORWARD = "MEC_DATA_FORWARD"
    NODE_MOBILITY_UPDATE = "MOBILITY_UPDATE"

@dataclass
class NetworkMessage:
    """Real network message with proper headers"""
    msg_id: str
    msg_type: MessageType
    source_ip: str
    dest_ip: str
    source_node_id: int
    dest_node_id: int
    payload: Dict[str, Any]
    timestamp: float
    ttl: int = 64
    size_bytes: int = 64  # Message size for energy calculation
    channel: int = 0
    
    def __post_init__(self):
        if not self.msg_id:
            self.msg_id = str(uuid.uuid4())[:8]

@dataclass 
class Event:
    """Discrete event for simulation"""
    time: float
    event_type: str
    node_id: int
    data: Any = None
    
    def __lt__(self, other):
        return self.time < other.time

# ============================================================================
# MEC INFRASTRUCTURE - Real MEC servers and gateways
# ============================================================================

class MECServer:
    """Mobile Edge Computing Server"""
    
    def __init__(self, mec_id: int, x: float, y: float, ip_address: str):
        self.mec_id = mec_id
        self.x = x
        self.y = y
        self.ip_address = ip_address
        self.connected_clusters: Dict[int, float] = {}  # cluster_head_id -> connection_time
        self.routing_table: Dict[str, str] = {}  # dest_network -> next_hop_ip
        self.data_cache: List[NetworkMessage] = []
        self.coverage_range = 500.0  # MEC coverage range
        
    def distance_to_node(self, node_x: float, node_y: float) -> float:
        return math.sqrt((self.x - node_x)**2 + (self.y - node_y)**2)
    
    def can_serve_node(self, node_x: float, node_y: float) -> bool:
        return self.distance_to_node(node_x, node_y) <= self.coverage_range
    
    def register_cluster(self, cluster_head_id: int, timestamp: float):
        """Register a cluster head with this MEC server"""
        self.connected_clusters[cluster_head_id] = timestamp
        print(f"MEC-{self.mec_id}: Registered cluster head {cluster_head_id}")
    
    def forward_data(self, message: NetworkMessage, target_mec_ip: str):
        """Forward data to another MEC server"""
        message.payload['forwarded_by'] = self.ip_address
        message.payload['forward_target'] = target_mec_ip
        print(f"MEC-{self.mec_id}: Forwarding {message.msg_type} to {target_mec_ip}")

class Gateway:
    """Gateway connecting MEC servers"""
    
    def __init__(self, gateway_id: int):
        self.gateway_id = gateway_id
        self.mec_servers: Dict[int, MECServer] = {}
        self.routing_table: Dict[str, int] = {}  # dest_network -> mec_id
        
    def add_mec_server(self, mec_server: MECServer):
        self.mec_servers[mec_server.mec_id] = mec_server
        
    def route_between_mecs(self, source_mec_id: int, dest_cluster_id: int, message: NetworkMessage):
        """Route message between MEC servers"""
        # Find MEC serving destination cluster
        for mec_id, mec in self.mec_servers.items():
            if dest_cluster_id in mec.connected_clusters:
                if mec_id != source_mec_id:
                    source_mec = self.mec_servers[source_mec_id]
                    source_mec.forward_data(message, mec.ip_address)
                return mec_id
        return None

# ============================================================================
# NETWORK NODE - With IP addressing and proper networking
# ============================================================================

class NetworkNode:
    """IoT Node with proper networking capabilities"""
    
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 100.0):
        # Basic properties
        self.node_id = node_id
        self.x = x
        self.y = y
        self.energy = initial_energy
        self.initial_energy = initial_energy
        
        # Network configuration
        self.ip_address = f"192.168.1.{node_id + 10}"  # Assign IP address
        self.mac_address = f"00:11:22:33:44:{node_id:02x}"
        self.subnet_mask = "255.255.255.0"
        
        # Routing and neighbors
        self.routing_table: Dict[str, str] = {}  # dest_ip -> next_hop_ip
        self.neighbor_table: Dict[int, Dict] = {}  # node_id -> {ip, rssi, last_seen, lqe_score}
        self.message_buffer: List[NetworkMessage] = []
        
        # Clustering state
        self.cluster_state = "IDLE"  # IDLE, CLUSTER_HEAD, CLUSTER_MEMBER
        self.cluster_id = None
        self.cluster_head_ip = None
        self.cluster_members: List[int] = []
        
        # MEC connection
        self.connected_mec_id = None
        self.mec_ip = None
        
        # Mobility
        self.velocity_x = random.uniform(-2, 2)  # m/s
        self.velocity_y = random.uniform(-2, 2)  # m/s
        self.mobility_enabled = True
        
        # Timing
        self.last_hello_time = 0
        self.last_clustering_time = 0
        self.hello_interval = 5.0  # Send HELLO every 5 seconds
        self.clustering_interval = 30.0  # Re-cluster every 30 seconds
        
        # Energy model
        self.tx_power = 0.03  # Transmission energy (J)
        self.rx_power = 0.02  # Reception energy (J)
        self.idle_power = 0.001  # Idle energy per second (J/s)
        
        # Communication range
        self.comm_range = 200.0  # Communication range in meters
        
    def distance_to(self, other_node: 'NetworkNode') -> float:
        return math.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2)
    
    def can_communicate_with(self, other_node: 'NetworkNode') -> bool:
        return self.distance_to(other_node) <= self.comm_range
    
    def calculate_rssi(self, distance: float) -> float:
        """Calculate RSSI based on distance"""
        if distance <= 0:
            return -30.0
        # Path loss model: RSSI = -30 - 20*log10(d/10) + noise
        rssi = -30 - 20 * math.log10(distance / 10.0) + random.gauss(0, 3)
        return max(-100, min(-30, rssi))
    
    def send_message(self, dest_node_id: int, msg_type: MessageType, payload: Dict, simulation):
        """Send a message to another node"""
        dest_node = simulation.nodes.get(dest_node_id)
        if not dest_node:
            return False
        
        # Check if destination is reachable
        if not self.can_communicate_with(dest_node):
            return False
        
        # Create message
        message = NetworkMessage(
            msg_id="",
            msg_type=msg_type,
            source_ip=self.ip_address,
            dest_ip=dest_node.ip_address,
            source_node_id=self.node_id,
            dest_node_id=dest_node_id,
            payload=payload,
            timestamp=simulation.current_time,
            size_bytes=64 + len(str(payload))
        )
        
        # Calculate energy cost
        distance = self.distance_to(dest_node)
        energy_cost = self.tx_power * (message.size_bytes / 1000.0)  # Energy per KB
        self.consume_energy(energy_cost)
        
        # Schedule message delivery
        propagation_delay = distance / 300000000  # Speed of light approximation
        delivery_time = simulation.current_time + propagation_delay + random.uniform(0.001, 0.005)
        
        delivery_event = Event(delivery_time, "MESSAGE_DELIVERY", dest_node_id, message)
        simulation.schedule_event(delivery_event)
        
        print(f"[{simulation.current_time:.3f}] Node-{self.node_id} -> Node-{dest_node_id}: {msg_type.value}")
        return True
    
    def receive_message(self, message: NetworkMessage, simulation):
        """Process received message"""
        # Energy cost for reception
        energy_cost = self.rx_power * (message.size_bytes / 1000.0)
        self.consume_energy(energy_cost)
        
        print(f"[{simulation.current_time:.3f}] Node-{self.node_id} received {message.msg_type.value} from Node-{message.source_node_id}")
        
        # Process based on message type
        if message.msg_type == MessageType.HELLO:
            self._process_hello_message(message, simulation)
        elif message.msg_type == MessageType.JOIN:
            self._process_join_message(message, simulation)
        elif message.msg_type == MessageType.DATA:
            self._process_data_message(message, simulation)
        elif message.msg_type == MessageType.CLUSTER_HEAD_ANNOUNCE:
            self._process_ch_announce(message, simulation)
    
    def _process_hello_message(self, message: NetworkMessage, simulation):
        """Process HELLO message for neighbor discovery"""
        sender_id = message.source_node_id
        sender_node = simulation.nodes[sender_id]
        
        # Calculate link quality metrics
        distance = self.distance_to(sender_node)
        rssi = self.calculate_rssi(distance)
        
        # Update neighbor table
        if sender_id not in self.neighbor_table:
            self.neighbor_table[sender_id] = {
                'ip': message.source_ip,
                'rssi_history': [],
                'last_seen': simulation.current_time,
                'hello_count': 0,
                'lqe_score': 0.0
            }
        
        neighbor_info = self.neighbor_table[sender_id]
        neighbor_info['rssi_history'].append(rssi)
        neighbor_info['last_seen'] = simulation.current_time
        neighbor_info['hello_count'] += 1
        
        # Keep only recent RSSI values
        if len(neighbor_info['rssi_history']) > 10:
            neighbor_info['rssi_history'] = neighbor_info['rssi_history'][-10:]
        
        # Calculate LQE score (simplified)
        avg_rssi = sum(neighbor_info['rssi_history']) / len(neighbor_info['rssi_history'])
        pdr = min(1.0, neighbor_info['hello_count'] / 10.0)  # Packet delivery ratio
        neighbor_info['lqe_score'] = 0.6 * pdr + 0.4 * ((avg_rssi + 100) / 70.0)  # Normalize RSSI
    
    def _process_join_message(self, message: NetworkMessage, simulation):
        """Process JOIN message for cluster formation"""
        if self.cluster_state == "CLUSTER_HEAD":
            # Accept the node as cluster member
            member_id = message.source_node_id
            if member_id not in self.cluster_members:
                self.cluster_members.append(member_id)
                
                # Send confirmation
                self.send_message(member_id, MessageType.CLUSTER_HEAD_ANNOUNCE, {
                    'cluster_id': self.cluster_id,
                    'cluster_head_ip': self.ip_address
                }, simulation)
    
    def _process_data_message(self, message: NetworkMessage, simulation):
        """Process data message"""
        if self.cluster_state == "CLUSTER_HEAD":
            # Forward to MEC server if connected
            if self.connected_mec_id is not None:
                mec_server = simulation.mec_servers[self.connected_mec_id]
                mec_server.data_cache.append(message)
    
    def _process_ch_announce(self, message: NetworkMessage, simulation):
        """Process cluster head announcement"""
        if self.cluster_state == "IDLE":
            self.cluster_state = "CLUSTER_MEMBER"
            self.cluster_id = message.payload['cluster_id']
            self.cluster_head_ip = message.payload['cluster_head_ip']
    
    def update_mobility(self, simulation):
        """Update node position based on mobility"""
        if not self.mobility_enabled:
            return
        
        # Update position
        time_delta = 1.0  # 1 second time step
        self.x += self.velocity_x * time_delta
        self.y += self.velocity_y * time_delta
        
        # Boundary conditions (bounce off walls)
        if self.x < 0 or self.x > simulation.area_width:
            self.velocity_x = -self.velocity_x
            self.x = max(0, min(simulation.area_width, self.x))
        
        if self.y < 0 or self.y > simulation.area_height:
            self.velocity_y = -self.velocity_y
            self.y = max(0, min(simulation.area_height, self.y))
        
        # Occasionally change direction
        if random.random() < 0.1:
            self.velocity_x += random.uniform(-0.5, 0.5)
            self.velocity_y += random.uniform(-0.5, 0.5)
            
            # Limit maximum speed
            max_speed = 3.0  # m/s
            speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
            if speed > max_speed:
                self.velocity_x = (self.velocity_x / speed) * max_speed
                self.velocity_y = (self.velocity_y / speed) * max_speed
    
    def send_hello_broadcast(self, simulation):
        """Send HELLO message to all neighbors"""
        neighbors = []
        for other_id, other_node in simulation.nodes.items():
            if other_id != self.node_id and self.can_communicate_with(other_node):
                neighbors.append(other_id)
        
        # Broadcast HELLO to all neighbors
        for neighbor_id in neighbors:
            self.send_message(neighbor_id, MessageType.HELLO, {
                'position': (self.x, self.y),
                'energy': self.energy,
                'timestamp': simulation.current_time
            }, simulation)
    
    def initiate_clustering(self, simulation):
        """Initiate clustering process"""
        if not self.neighbor_table:
            # No neighbors, become single-node cluster
            self.cluster_state = "CLUSTER_HEAD"
            self.cluster_id = self.node_id
            self.cluster_members = []
            return
        
        # Find best neighbor to join based on LQE score
        best_neighbor_id = max(self.neighbor_table.keys(), 
                              key=lambda nid: self.neighbor_table[nid]['lqe_score'])
        
        # Send JOIN message
        self.send_message(best_neighbor_id, MessageType.JOIN, {
            'requesting_cluster_membership': True,
            'node_energy': self.energy,
            'lqe_score': self.neighbor_table[best_neighbor_id]['lqe_score']
        }, simulation)
    
    def consume_energy(self, amount: float):
        """Consume energy"""
        self.energy = max(0, self.energy - amount)
    
    def is_alive(self) -> bool:
        return self.energy > 0.1

# ============================================================================
# DISCRETE EVENT SIMULATION ENGINE
# ============================================================================

class ARPMECNetworkSimulation:
    """Discrete-event network simulation for ARPMEC"""
    
    def __init__(self, area_width: float = 1000, area_height: float = 1000):
        # Simulation parameters
        self.current_time = 0.0
        self.end_time = 300.0  # 5 minutes simulation
        self.area_width = area_width
        self.area_height = area_height
        
        # Event queue
        self.event_queue: List[Event] = []
        
        # Network components
        self.nodes: Dict[int, NetworkNode] = {}
        self.mec_servers: Dict[int, MECServer] = {}
        self.gateway = Gateway(0)
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'clusters_formed': 0,
            'energy_consumed': 0,
            'node_deaths': 0
        }
        
    def add_node(self, node: NetworkNode):
        """Add a node to the simulation"""
        self.nodes[node.node_id] = node
        
        # Schedule initial events
        self.schedule_event(Event(random.uniform(0, 1), "HELLO_BROADCAST", node.node_id))
        self.schedule_event(Event(random.uniform(5, 10), "CLUSTERING", node.node_id))
        self.schedule_event(Event(1.0, "MOBILITY_UPDATE", node.node_id))
    
    def add_mec_server(self, mec: MECServer):
        """Add MEC server to simulation"""
        self.mec_servers[mec.mec_id] = mec
        self.gateway.add_mec_server(mec)
    
    def schedule_event(self, event: Event):
        """Schedule an event"""
        heapq.heappush(self.event_queue, event)
    
    def run_simulation(self):
        """Run the discrete-event simulation"""
        print(f"Starting ARPMEC Network Simulation")
        print(f"Area: {self.area_width}m x {self.area_height}m")
        print(f"Nodes: {len(self.nodes)}, MEC servers: {len(self.mec_servers)}")
        print("="*60)
        
        while self.event_queue and self.current_time < self.end_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            
            # Process event
            if event.node_id in self.nodes:
                node = self.nodes[event.node_id]
                
                if not node.is_alive():
                    continue
                
                self._process_event(event, node)
        
        self._print_final_statistics()
    
    def _process_event(self, event: Event, node: NetworkNode):
        """Process a single event"""
        if event.event_type == "HELLO_BROADCAST":
            node.send_hello_broadcast(self)
            # Schedule next HELLO
            next_hello = Event(self.current_time + node.hello_interval, "HELLO_BROADCAST", node.node_id)
            self.schedule_event(next_hello)
            
        elif event.event_type == "CLUSTERING":
            node.initiate_clustering(self)
            # Schedule next clustering
            next_clustering = Event(self.current_time + node.clustering_interval, "CLUSTERING", node.node_id)
            self.schedule_event(next_clustering)
            
        elif event.event_type == "MOBILITY_UPDATE":
            node.update_mobility(self)
            # Consume idle energy
            node.consume_energy(node.idle_power * 1.0)  # 1 second
            # Schedule next mobility update
            next_mobility = Event(self.current_time + 1.0, "MOBILITY_UPDATE", node.node_id)
            self.schedule_event(next_mobility)
            
        elif event.event_type == "MESSAGE_DELIVERY":
            message = event.data
            node.receive_message(message, self)
            self.stats['messages_delivered'] += 1
    
    def _print_final_statistics(self):
        """Print simulation statistics"""
        alive_nodes = sum(1 for node in self.nodes.values() if node.is_alive())
        total_energy = sum(node.initial_energy - node.energy for node in self.nodes.values())
        cluster_heads = sum(1 for node in self.nodes.values() if node.cluster_state == "CLUSTER_HEAD")
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETED")
        print("="*60)
        print(f"Simulation time: {self.current_time:.1f}s")
        print(f"Alive nodes: {alive_nodes}/{len(self.nodes)}")
        print(f"Network lifetime: {alive_nodes/len(self.nodes)*100:.1f}%")
        print(f"Total energy consumed: {total_energy:.2f}J")
        print(f"Messages delivered: {self.stats['messages_delivered']}")
        print(f"Clusters formed: {cluster_heads}")
        print(f"Average energy per node: {total_energy/len(self.nodes):.2f}J")

# ============================================================================
# MAIN SIMULATION SETUP
# ============================================================================

def create_realistic_scenario():
    """Create a realistic ARPMEC network scenario"""
    simulation = ARPMECNetworkSimulation(area_width=800, area_height=600)
    
    # Create IoT nodes
    num_nodes = 25
    for i in range(num_nodes):
        # Create clusters of nodes in different areas
        if i < 10:
            # Urban area cluster
            x = random.uniform(100, 300)
            y = random.uniform(100, 250)
        elif i < 20:
            # Suburban area cluster  
            x = random.uniform(400, 700)
            y = random.uniform(150, 400)
        else:
            # Mobile nodes
            x = random.uniform(0, 800)
            y = random.uniform(0, 600)
        
        energy = random.uniform(90, 110)
        node = NetworkNode(i, x, y, energy)
        
        # Some nodes are mobile, others are static
        if i >= 20:
            node.mobility_enabled = True
        else:
            node.mobility_enabled = False
            node.velocity_x = 0
            node.velocity_y = 0
        
        simulation.add_node(node)
    
    # Create MEC servers
    mec1 = MECServer(1, 200, 200, "10.0.1.1")  # Urban MEC
    mec2 = MECServer(2, 600, 300, "10.0.2.1")  # Suburban MEC
    
    simulation.add_mec_server(mec1)
    simulation.add_mec_server(mec2)
    
    return simulation

if __name__ == "__main__":
    # Run realistic network simulation
    sim = create_realistic_scenario()
    sim.run_simulation()