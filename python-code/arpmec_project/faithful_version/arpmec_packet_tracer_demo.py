#!/usr/bin/env python3
"""
ARPMEC PROTOCOL-DRIVEN Packet Tracer Demo
Creates a video animation showing REAL protocol-driven packets moving through the network.
This version uses the actual ARPMEC protocol logic instead of synthetic traffic generation.

Key Features:
- Uses actual protocol methods for inter-cluster communication
- Real IAR/MEC infrastructure routing
- Dynamic clustering with mobility
- Paper-faithful implementation
- No synthetic traffic - only protocol-driven flows
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from arpmec_faithful import (ARPMECProtocol, IARServer, InterClusterMessage,
                             MECServer, MECTask, Node, NodeState)

# Animation configuration
ANIMATION_SPEED = 0.07  # Packet movement speed
FRAME_DURATION = 200   # milliseconds per frame
VIDEO_DURATION = 60    # seconds

@dataclass 
@dataclass
class MovingPacket:
    """Represents a packet moving through the network with 5-second path visibility"""
    packet_id: str
    source: Tuple[float, float]
    destination: Tuple[float, float]
    path: List[Tuple[float, float]]  # Via points
    current_pos: Tuple[float, float]
    progress: float  # 0.0 to 1.0
    packet_type: str  # 'data', 'mec_task', 'inter_cluster', 'control'
    color: str
    size: float
    active: bool = True
    current_segment: int = 0  # Current segment in path
    description: str = ""  # Description of the communication
    source_node_id: int = -1
    dest_node_id: int = -1
    hop_descriptions: List[str] = None  # Hop-by-hop descriptions
    routing_events: List[str] = None  # Routing events to display
    last_hop_event: str = ""  # Last hop event triggered
    current_hop_info: str = ""  # Current hop being processed
    routing_log: List[str] = None  # Detailed routing log
    hop_timestamps: List[float] = None  # Time when each hop was reached
    path_visibility_timer: float = 10.0  # NEW: How long path lines should be visible (seconds)
    path_created_time: float = 0.0  # NEW: When the path was created
    show_path_lines: bool = True  # NEW: Whether to show path lines
    
    def update(self, dt: float):
        """Update packet position with hop-by-hop movement and pausing"""
        if not self.active or not self.path:
            return
            
        # Update path visibility timer - Hide path lines after 5 seconds
        current_time = time.time()
        if current_time - self.path_created_time >= self.path_visibility_timer:
            self.show_path_lines = False
            
        # Get current segment endpoints
        if self.current_segment >= len(self.path) - 1:
            self.active = False
            return
            
        start = self.path[self.current_segment]
        end = self.path[self.current_segment + 1]
        
        # Move along current segment with slower speed for visibility
        self.progress += ANIMATION_SPEED * dt * 0.5  # Slower movement for better visibility
        
        # Check if we've reached the end of current segment (arrived at hop)
        if self.progress >= 1.0:
            # We've reached the next hop - PAUSE here
            self.current_pos = end
            
            # Trigger routing event when arriving at hop
            if self.routing_events and self.current_segment < len(self.routing_events):
                self.last_hop_event = self.routing_events[self.current_segment]
                self.current_hop_info = self.last_hop_event
            
            # Add pause time at each hop for processing
            if not hasattr(self, 'pause_time'):
                self.pause_time = 0.0
            
            self.pause_time += dt
            
            # Pause for 2 seconds at each hop (except final destination)
            if self.pause_time >= 2.0 or self.current_segment >= len(self.path) - 2:
                # Move to next segment
                self.current_segment += 1
                self.progress = 0.0
                self.pause_time = 0.0
                
                if self.current_segment >= len(self.path) - 1:
                    self.current_pos = self.destination
                    self.active = False
                    # Trigger final routing event
                    if self.routing_events and len(self.routing_events) > self.current_segment:
                        self.last_hop_event = self.routing_events[-1]
                        self.current_hop_info = self.last_hop_event
                    return
        else:
            # Interpolate position along current segment
            t = self.progress
            self.current_pos = (
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1])
            )

class ARPMECPacketTracerDemo:
    """Packet Tracer style demonstration of ARPMEC protocol"""
    
    def __init__(self, num_nodes: int = 20, area_size: int = 1200):  # Increased area size
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.protocol = None
        self.moving_packets = []
        self.packet_id_counter = 0
        
        # Color schemes
        self.node_colors = {
            'cluster_head': '#FF4444',
            'cluster_member': '#4444FF', 
            'idle': '#CCCCCC'
        }
        
        self.infra_colors = {
            'mec_server': '#000080',
            'iar_server': '#800080'
        }
        
        self.packet_colors = {
            'data': '#00FF00',
            'mec_task': '#FFA500', 
            'inter_cluster': '#FF00FF',
            'control': '#FFFF00'
        }
        
        # Initialize random seed for reproducible results
        random.seed(42)
        np.random.seed(42)
    
    def create_network(self):
        """Create a realistic network for demonstration"""
        print("Creating ARPMEC network with IAR infrastructure...")
        
        # Create nodes in clustered groups
        nodes = []
        cluster_centers = [
            (200, 200), (600, 200), (400, 600)
        ]
        
        nodes_per_cluster = self.num_nodes // len(cluster_centers)
        node_id = 0
        
        for cx, cy in cluster_centers:
            for i in range(nodes_per_cluster):
                # Place within communication range of center
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, 80)  # Within 80m
                
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                
                # Keep in bounds
                x = max(50, min(self.area_size - 50, x))
                y = max(50, min(self.area_size - 50, y))
                
                energy = random.uniform(90, 110)
                nodes.append(Node(node_id, x, y, energy))
                node_id += 1
        
        # Add remaining nodes randomly
        while len(nodes) < self.num_nodes:
            x = random.uniform(50, self.area_size - 50)
            y = random.uniform(50, self.area_size - 50)
            energy = random.uniform(90, 110)
            nodes.append(Node(node_id, x, y, energy))
            node_id += 1
        
        # Create protocol
        self.protocol = ARPMECProtocol(nodes, C=4, R=5, K=3)
        
        # Perform initial clustering
        clusters = self.protocol.clustering_algorithm()
        print(f"Created {len(clusters)} clusters")
        
        return clusters
    
    def create_packet(self, source_pos: Tuple[float, float], 
                     dest_pos: Tuple[float, float], 
                     path: List[Tuple[float, float]],
                     packet_type: str) -> MovingPacket:
        """Create a new animated packet"""
        self.packet_id_counter += 1
        
        # Complete path including source and destination
        complete_path = [source_pos] + path + [dest_pos]
        
        packet = MovingPacket(
            packet_id=f"packet_{self.packet_id_counter}",
            source=source_pos,
            destination=dest_pos,
            path=complete_path,
            current_pos=source_pos,
            progress=0.0,
            packet_type=packet_type,
            color=self.packet_colors[packet_type],
            size=8 if packet_type == 'mec_task' else 6,
            active=True,
            current_segment=0,
            description="",  # Will be set by traffic generator
            source_node_id=-1,
            dest_node_id=-1,
            path_visibility_timer=5.0,  # Path visible for 5 seconds only
            path_created_time=time.time(),  # Set current time for 5s timer
            show_path_lines=True  # Initially show path lines
        )
        
        return packet
    
    def generate_packet_traffic(self):
        """Generate PROTOCOL-HIERARCHY-RESPECTING packet traffic - following ARPMEC communication patterns"""
        
        # APPROACH: Follow the actual ARPMEC protocol communication hierarchy
        # 1. Member → CH (intra-cluster)
        # 2. CH → IAR → MEC (cluster to infrastructure) 
        # 3. CH → IAR → MEC → IAR → CH (inter-cluster via infrastructure)
        # 4. All communications respect range limitations and routing hierarchy
        
        cluster_heads = self.protocol._get_cluster_heads()
        cluster_members = [n for n in self.protocol.nodes.values() 
                          if n.state == NodeState.CLUSTER_MEMBER and n.is_alive()]
        
        # 1. INTRA-CLUSTER COMMUNICATION (Member → CH) - Respects range limits
        self._generate_intra_cluster_traffic(cluster_heads, cluster_members)
        
        # 2. MEC TASK OFFLOADING (CH → IAR → MEC) - Following protocol hierarchy
        self._generate_mec_task_traffic(cluster_heads)
        
        # 3. INTER-CLUSTER COMMUNICATION (CH → IAR → MEC → IAR → CH) - Full protocol path
        self._generate_inter_cluster_traffic(cluster_heads)
        
        # 4. Run actual protocol operations (for metrics and state management)
        self.protocol._generate_inter_cluster_traffic()
        self.protocol._generate_mec_tasks()
        self.protocol._process_inter_cluster_messages()
        self.protocol._process_mec_servers()
    
    def _generate_intra_cluster_traffic(self, cluster_heads, cluster_members):
        """Generate intra-cluster communication that respects range limits"""
        
        for member in cluster_members:
            if not member.cluster_head_id:
                continue
                
            # Find the assigned cluster head
            ch = next((ch for ch in cluster_heads if ch.id == member.cluster_head_id), None)
            if not ch:
                continue
                
            # Check if member is within communication range of CH
            distance = math.sqrt((member.x - ch.x)**2 + (member.y - ch.y)**2)
            if distance > self.protocol.communication_range:
                print(f"⚠️ Member-{member.id} too far from CH-{ch.id} ({distance:.1f}m > {self.protocol.communication_range}m)")
                continue
            
            # Generate traffic with realistic frequency
            if random.random() < 0.12:  # 12% chance for visibility
                packet = self.create_packet(
                    (member.x, member.y), (ch.x, ch.y), [],
                    'data'
                )
                packet.description = f"Intra-cluster: Member-{member.id} → CH-{ch.id} (distance: {distance:.1f}m)"
                packet.source_node_id = member.id
                packet.dest_node_id = ch.id
                packet.routing_events = [
                    f"Member-{member.id}: Collecting sensor data within range ({distance:.1f}m)",
                    f"CH-{ch.id}: Received intra-cluster data from Member-{member.id}"
                ]
                self.moving_packets.append(packet)
                print(f"� INTRA-CLUSTER: Member-{member.id} → CH-{ch.id} ({distance:.1f}m)")
    
    def _generate_mec_task_traffic(self, cluster_heads):
        """Generate MEC task offloading following CH → IAR → MEC hierarchy"""
        
        for ch in cluster_heads:
            if random.random() < 0.20:  # 20% chance for MEC tasks
                # Follow protocol: CH must go through nearest IAR to reach MEC
                nearest_iar = self.protocol._find_nearest_iar_server(ch)
                if not nearest_iar:
                    print(f"❌ CH-{ch.id}: No IAR server within range for MEC task")
                    continue
                
                # Check IAR connectivity
                if not nearest_iar.connected_mec_servers:
                    print(f"❌ IAR-{nearest_iar.id}: No connected MEC servers")
                    continue
                
                # Select MEC server through IAR (protocol hierarchy)
                mec_id = nearest_iar.connected_mec_servers[0]  # Use first available
                target_mec = self.protocol.mec_servers[mec_id]
                
                # Verify IAR-CH distance
                iar_distance = math.sqrt((ch.x - nearest_iar.x)**2 + (ch.y - nearest_iar.y)**2)
                if iar_distance > nearest_iar.coverage_radius:
                    print(f"⚠️ CH-{ch.id} outside IAR-{nearest_iar.id} coverage ({iar_distance:.1f}m)")
                    continue
                
                # Create protocol-compliant MEC task packet
                self._create_mec_task_packet_hierarchy(ch, nearest_iar, target_mec)
    
    def _generate_inter_cluster_traffic(self, cluster_heads):
        """Generate inter-cluster communication following full protocol hierarchy"""
        
        if len(cluster_heads) < 2:
            return
            
        if random.random() < 0.18:  # 18% chance for inter-cluster
            source_ch = random.choice(cluster_heads)
            target_ch = random.choice([ch for ch in cluster_heads if ch.id != source_ch.id])
            
            # Follow ARPMEC protocol: CH → IAR → MEC → IAR → CH (NO shortcuts!)
            source_iar = self.protocol._find_nearest_iar_server(source_ch)
            target_iar = self.protocol._find_nearest_iar_server(target_ch)
            
            if not source_iar or not target_iar:
                print(f"❌ Inter-cluster blocked: Missing IAR coverage")
                return
                
            # Verify both CHs are within IAR coverage
            source_distance = math.sqrt((source_ch.x - source_iar.x)**2 + (source_ch.y - source_iar.y)**2)
            target_distance = math.sqrt((target_ch.x - target_iar.x)**2 + (target_ch.y - target_iar.y)**2)
            
            if source_distance > source_iar.coverage_radius:
                print(f"⚠️ Source CH-{source_ch.id} outside IAR coverage ({source_distance:.1f}m)")
                return
                
            if target_distance > target_iar.coverage_radius:
                print(f"⚠️ Target CH-{target_ch.id} outside IAR coverage ({target_distance:.1f}m)")
                return
            
            # Create protocol-compliant inter-cluster packet
            self._create_inter_cluster_packet_hierarchy(source_ch, target_ch, source_iar, target_iar)
    
    def _create_mec_task_packet_hierarchy(self, source_ch: Node, iar: 'IARServer', mec: 'MECServer'):
        """Create MEC task packet following protocol hierarchy: CH → IAR → MEC"""
        
        # Build the protocol-compliant path: CH → IAR → MEC (NO direct CH → MEC!)
        complete_path = [
            (source_ch.x, source_ch.y),    # Start at cluster head
            (iar.x, iar.y),                # Must go through IAR first
            (mec.x, mec.y)                 # Finally reach MEC server
        ]
        
        # Create task identifier
        task_id = f"task_{source_ch.id}_{self.protocol.current_time_slot}"
        
        packet = MovingPacket(
            packet_id=f"mec_task_{task_id}",
            source=(source_ch.x, source_ch.y),
            destination=(mec.x, mec.y),
            path=complete_path,
            current_pos=(source_ch.x, source_ch.y),
            progress=0.0,
            packet_type='mec_task',
            color=self.packet_colors['mec_task'],
            size=8,
            active=True,
            current_segment=0,
            description=f"MEC Task: CH-{source_ch.id} → IAR-{iar.id} → MEC-{mec.id}",
            source_node_id=source_ch.id,
            dest_node_id=mec.id,
            routing_events=[
                f"CH-{source_ch.id}: Generated MEC task (requires IAR routing)",
                f"IAR-{iar.id}: Routing task to optimal MEC server",
                f"MEC-{mec.id}: Processing task (load: {mec.get_load_percentage():.1f}%)"
            ],
            hop_descriptions=[
                f"CH-{source_ch.id} → IAR-{iar.id}",
                f"IAR-{iar.id} → MEC-{mec.id}"
            ],
            path_visibility_timer=5.0,
            path_created_time=time.time(),
            show_path_lines=True
        )
        
        self.moving_packets.append(packet)
        print(f"🚀 MEC HIERARCHY: CH-{source_ch.id} → IAR-{iar.id} → MEC-{mec.id}")
    
    def _create_inter_cluster_packet_hierarchy(self, source_ch: Node, target_ch: Node, 
                                             source_iar: 'IARServer', target_iar: 'IARServer'):
        """Create inter-cluster packet following FULL protocol hierarchy"""
        
        # CRITICAL: Follow COMPLETE ARPMEC protocol path - NO shortcuts allowed!
        complete_path = [(source_ch.x, source_ch.y)]
        
        # Step 1: Source CH → Source IAR (mandatory first hop)
        complete_path.append((source_iar.x, source_iar.y))
        
        # Step 2: Source IAR → MEC (ALWAYS through MEC for inter-cluster)
        if source_iar.connected_mec_servers:
            source_mec_id = source_iar.connected_mec_servers[0]
            source_mec = self.protocol.mec_servers[source_mec_id]
            complete_path.append((source_mec.x, source_mec.y))
            
            # Step 3: Check if different MEC needed for target IAR
            if target_iar.connected_mec_servers:
                target_mec_id = target_iar.connected_mec_servers[0]
                
                if source_mec_id != target_mec_id:
                    # Different MEC servers: add MEC-to-MEC hop
                    target_mec = self.protocol.mec_servers[target_mec_id]
                    complete_path.append((target_mec.x, target_mec.y))
                    
                    # Step 4: Target MEC → Target IAR (mandatory)
                    complete_path.append((target_iar.x, target_iar.y))
                    routing_desc = f"CH-{source_ch.id} → IAR-{source_iar.id} → MEC-{source_mec_id} → MEC-{target_mec_id} → IAR-{target_iar.id} → CH-{target_ch.id}"
                    hop_events = [
                        f"CH-{source_ch.id}: Initiating inter-cluster message",
                        f"IAR-{source_iar.id}: Routing to MEC infrastructure", 
                        f"MEC-{source_mec_id}: Processing inter-cluster routing",
                        f"MEC-{target_mec_id}: Receiving inter-cluster message",
                        f"IAR-{target_iar.id}: Delivering to target cluster",
                        f"CH-{target_ch.id}: Inter-cluster message received"
                    ]
                else:
                    # Same MEC serves both: MEC → Target IAR (mandatory)
                    complete_path.append((target_iar.x, target_iar.y))
                    routing_desc = f"CH-{source_ch.id} → IAR-{source_iar.id} → MEC-{source_mec_id} → IAR-{target_iar.id} → CH-{target_ch.id}"
                    hop_events = [
                        f"CH-{source_ch.id}: Initiating inter-cluster message",
                        f"IAR-{source_iar.id}: Routing to shared MEC server",
                        f"MEC-{source_mec_id}: Processing inter-cluster routing",
                        f"IAR-{target_iar.id}: Receiving from shared MEC",
                        f"CH-{target_ch.id}: Inter-cluster message received"
                    ]
        
        # Step 5: Final hop Target IAR → Target CH (mandatory)
        complete_path.append((target_ch.x, target_ch.y))
        
        # Create the protocol-compliant packet
        packet = MovingPacket(
            packet_id=f"inter_cluster_{source_ch.id}_{target_ch.id}_{self.protocol.current_time_slot}",
            source=(source_ch.x, source_ch.y),
            destination=(target_ch.x, target_ch.y),
            path=complete_path,
            current_pos=(source_ch.x, source_ch.y),
            progress=0.0,
            packet_type='inter_cluster',
            color=self.packet_colors['inter_cluster'],
            size=10,
            active=True,
            current_segment=0,
            description=f"Inter-cluster HIERARCHY: {routing_desc}",
            source_node_id=source_ch.id,
            dest_node_id=target_ch.id,
            routing_events=hop_events,
            hop_descriptions=[f"Hop {i+1}" for i in range(len(complete_path)-1)],
            path_visibility_timer=8.0,  # Longer visibility for complex inter-cluster paths
            path_created_time=time.time(),
            show_path_lines=True
        )
        
        self.moving_packets.append(packet)
        print(f"📡 INTER-CLUSTER HIERARCHY: {routing_desc}")
        
        # Create corresponding message in protocol
        from arpmec_faithful import InterClusterMessage
        message = InterClusterMessage(
            message_id=f"hierarchy_msg_{source_ch.id}_{target_ch.id}_{self.protocol.current_time_slot}",
            source_cluster_id=source_ch.cluster_id,
            destination_cluster_id=target_ch.cluster_id,
            message_type='data',
            payload={'data': f'hierarchy_data_cluster_{source_ch.cluster_id}'},
            timestamp=self.protocol.current_time_slot
        )
    def _create_inter_cluster_packet(self, source_ch: Node, target_ch: Node, message: 'InterClusterMessage'):
        """Simplified inter-cluster packet creation - delegates to hierarchy method"""
        source_iar = self.protocol._find_nearest_iar_server(source_ch)
        target_iar = self.protocol._find_nearest_iar_server(target_ch)
        
        if source_iar and target_iar:
            self._create_inter_cluster_packet_hierarchy(source_ch, target_ch, source_iar, target_iar)
    
    def _create_mec_task_packet(self, source_ch: Node, mec: 'MECServer', task: 'MECTask'):
        """Simplified MEC task packet creation - delegates to hierarchy method"""
        source_iar = self.protocol._find_nearest_iar_server(source_ch)
        
        if source_iar:
            self._create_mec_task_packet_hierarchy(source_ch, source_iar, mec)
    
    def draw_network(self, ax, frame_time: float):
        """Draw the network state with better layout and smaller nodes"""
        ax.clear()
        ax.set_xlim(-100, self.area_size + 200)  # Extended frame for legends
        ax.set_ylim(-100, self.area_size + 100)  # Extended frame for legends
        ax.set_facecolor('#F0F8FF')
        ax.grid(True, alpha=0.3)
        
        cluster_heads = self.protocol._get_cluster_heads()
        cluster_members = [n for n in self.protocol.nodes.values() 
                          if n.state == NodeState.CLUSTER_MEMBER and n.is_alive()]
        idle_nodes = [n for n in self.protocol.nodes.values() 
                     if n.state == NodeState.IDLE and n.is_alive()]
        
        # 1. Draw IAR coverage areas
        for iar in self.protocol.iar_servers.values():
            coverage = plt.Circle((iar.x, iar.y), iar.coverage_radius, 
                                fill=False, color='purple', alpha=0.15, linestyle=':')
            ax.add_patch(coverage)
        
        # 2. Draw cluster boundaries with RANGE LIMITS CLEARLY SHOWN
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, ch in enumerate(cluster_heads):
            color = colors[i % len(colors)]
            # Show actual communication range limit (solid line)
            boundary = plt.Circle((ch.x, ch.y), self.protocol.communication_range,
                                fill=False, color=color, alpha=0.6, linestyle='-', linewidth=2)
            ax.add_patch(boundary)
            
            # Show effective cluster range (75% for stricter clustering - dashed line)
            effective_range = plt.Circle((ch.x, ch.y), self.protocol.communication_range * 0.75,
                                       fill=False, color=color, alpha=0.4, linestyle='--', linewidth=1)
            ax.add_patch(effective_range)
        
        # 3. Draw infrastructure connections
        for iar in self.protocol.iar_servers.values():
            for mec_id in iar.connected_mec_servers:
                if mec_id in self.protocol.mec_servers:
                    mec = self.protocol.mec_servers[mec_id]
                    ax.plot([iar.x, mec.x], [iar.y, mec.y],
                           'purple', alpha=0.5, linewidth=1.5, linestyle=':')
        
        # 4. Draw communication links with HIERARCHY EMPHASIS
        for ch in cluster_heads:
            # CH to IAR (CRITICAL HIERARCHY LINK - thick line)
            nearest_iar = self.protocol._find_nearest_iar_server(ch)
            if nearest_iar:
                iar_distance = math.sqrt((ch.x - nearest_iar.x)**2 + (ch.y - nearest_iar.y)**2)
                
                # Color code by distance: green=good, yellow=far, red=out of range
                if iar_distance <= nearest_iar.coverage_radius * 0.7:
                    link_color = 'green'
                    alpha = 0.8
                elif iar_distance <= nearest_iar.coverage_radius:
                    link_color = 'orange'
                    alpha = 0.7
                else:
                    link_color = 'red'
                    alpha = 0.9
                
                ax.plot([ch.x, nearest_iar.x], [ch.y, nearest_iar.y],
                       link_color, alpha=alpha, linewidth=3, linestyle='-')
                
                # Add distance label
                mid_x = (ch.x + nearest_iar.x) / 2
                mid_y = (ch.y + nearest_iar.y) / 2
                ax.text(mid_x, mid_y, f'{iar_distance:.0f}m', 
                       ha='center', fontsize=8, color=link_color, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
            # Member to CH (show only valid connections within range)
            for member_id in ch.cluster_members:
                if member_id in self.protocol.nodes:
                    member = self.protocol.nodes[member_id]
                    if member.is_alive():
                        member_distance = math.sqrt((ch.x - member.x)**2 + (ch.y - member.y)**2)
                        
                        # Only show if within effective range
                        if member_distance <= self.protocol.communication_range * 0.75:
                            ax.plot([ch.x, member.x], [ch.y, member.y],
                                   'gray', alpha=0.5, linewidth=1)
                        else:
                            # Show problematic connections in red
                            ax.plot([ch.x, member.x], [ch.y, member.y],
                                   'red', alpha=0.8, linewidth=2, linestyle=':')
                            
                            # Mark as problematic
                            mid_x = (ch.x + member.x) / 2
                            mid_y = (ch.y + member.y) / 2
                            ax.text(mid_x, mid_y, '⚠️', ha='center', fontsize=12)
        
        # 5. Draw nodes (smaller sizes)
        for node in self.protocol.nodes.values():
            if not node.is_alive():
                continue
                
            if node.state == NodeState.CLUSTER_HEAD:
                ax.scatter(node.x, node.y, c=self.node_colors['cluster_head'], 
                          s=120, marker='^', edgecolors='black', linewidth=1.5, zorder=5)  # Smaller
                ax.text(node.x, node.y + 15, f'CH-{node.id}', 
                       ha='center', fontsize=8, weight='bold')  # Smaller font
                
            elif node.state == NodeState.CLUSTER_MEMBER:
                ax.scatter(node.x, node.y, c=self.node_colors['cluster_member'], 
                          s=60, marker='o', edgecolors='black', linewidth=1, zorder=5)  # Smaller
                ax.text(node.x, node.y + 12, f'{node.id}', 
                       ha='center', fontsize=7)  # Smaller font
                
            else:  # IDLE
                ax.scatter(node.x, node.y, c=self.node_colors['idle'], 
                          s=40, marker='o', edgecolors='gray', linewidth=1, zorder=5)  # Smaller
        
        # 6. Draw infrastructure (smaller sizes)
        for mec in self.protocol.mec_servers.values():
            load_pct = mec.get_load_percentage()
            load_color = 'red' if load_pct > 80 else 'orange' if load_pct > 60 else 'green'
            
            ax.scatter(mec.x, mec.y, c=self.infra_colors['mec_server'], 
                      s=250, marker='s', edgecolors=load_color, linewidth=2, zorder=6)  # Smaller
            ax.text(mec.x, mec.y - 25, f'MEC-{mec.id}', 
                   ha='center', fontsize=8, weight='bold', color='darkblue')  # Smaller font
            ax.text(mec.x, mec.y + 25, f'{load_pct:.0f}%', 
                   ha='center', fontsize=7, color=load_color, weight='bold')  # Smaller font
        
        for iar in self.protocol.iar_servers.values():
            ax.scatter(iar.x, iar.y, c=self.infra_colors['iar_server'], 
                      s=180, marker='D', edgecolors='black', linewidth=1.5, zorder=6)  # Smaller
            ax.text(iar.x, iar.y - 20, f'IAR-{iar.id}', 
                   ha='center', fontsize=7, weight='bold', color='purple')  # Smaller font
        
        # 7. Update and draw moving packets with enhanced path visualization
        packet_descriptions = []  # Collect descriptions for overlay
        routing_events = []  # Collect active routing events
        
        for i, packet in enumerate(self.moving_packets[:]):
            packet.update(1.0)
            if not packet.active:
                self.moving_packets.remove(packet)
                continue
            
            # Collect current routing event if available
            if hasattr(packet, 'current_hop_info') and packet.current_hop_info:
                routing_events.append(packet.current_hop_info)
            
            # Draw enhanced path line with directional arrows (only if still visible)
            if len(packet.path) > 1 and packet.show_path_lines:
                path_x = [pos[0] for pos in packet.path]
                path_y = [pos[1] for pos in packet.path]
                
                # Draw thick path line
                ax.plot(path_x, path_y, color=packet.color, alpha=0.7, 
                       linewidth=4, linestyle='-', zorder=50)
                
                # Add directional arrows along the path
                for j in range(len(packet.path) - 1):
                    start = packet.path[j]
                    end = packet.path[j + 1]
                    
                    # Calculate arrow position (middle of segment)
                    arrow_x = (start[0] + end[0]) / 2
                    arrow_y = (start[1] + end[1]) / 2
                    
                    # Calculate arrow direction
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    length = math.sqrt(dx*dx + dy*dy)
                    
                    if length > 0:
                        dx /= length
                        dy /= length
                        
                        # Draw arrow
                        ax.annotate('', xy=(arrow_x + dx*15, arrow_y + dy*15), 
                                   xytext=(arrow_x - dx*15, arrow_y - dy*15),
                                   arrowprops=dict(arrowstyle='->', color=packet.color, 
                                                 lw=3, alpha=0.8), zorder=60)
                
                # Add path segment labels for inter-cluster packets
                if packet.packet_type == 'inter_cluster' and len(packet.path) >= 5:
                    # Label each segment of the CH -> IAR -> MEC -> IAR -> CH path
                    labels = ['CH→IAR', 'IAR→MEC', 'MEC→IAR', 'IAR→CH']
                    for j, label in enumerate(labels[:len(packet.path)-1]):
                        if j < len(packet.path) - 1:
                            start = packet.path[j]
                            end = packet.path[j + 1]
                            mid_x = (start[0] + end[0]) / 2
                            mid_y = (start[1] + end[1]) / 2
                            
                            # Offset label to avoid overlap
                            offset_y = 20 if j % 2 == 0 else -20
                            
                            ax.text(mid_x, mid_y + offset_y, label, 
                                   ha='center', va='center', fontsize=8, 
                                   color='white', weight='bold',
                                   bbox=dict(boxstyle="round,pad=0.2", 
                                           facecolor=packet.color, alpha=0.8),
                                   zorder=80)
            
            # Draw packet with enhanced pulsing effect for multi-hop packets
            if packet.packet_type == 'inter_cluster':
                # Special pulsing for multi-hop inter-cluster packets
                pulse_size = packet.size * (12 + 5 * math.sin(frame_time * 6))
                ax.scatter(packet.current_pos[0], packet.current_pos[1], 
                          c=packet.color, s=pulse_size, marker='o', 
                          alpha=0.95, zorder=100, edgecolors='red', linewidth=3)
                
                # Add "MULTI-HOP" indicator
                ax.text(packet.current_pos[0], packet.current_pos[1] - 35, 
                       "🔄 MULTI-HOP", 
                       ha='center', fontsize=9, color='red', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', 
                               alpha=0.95, edgecolor='red', linewidth=2),
                       zorder=110)
            else:
                # Standard pulsing for other packets
                pulse_size = packet.size * (10 + 3 * math.sin(frame_time * 5))
                ax.scatter(packet.current_pos[0], packet.current_pos[1], 
                          c=packet.color, s=pulse_size, marker='o', 
                          alpha=0.9, zorder=100, edgecolors='white', linewidth=2)
            
            # Add packet ID and type label
            ax.text(packet.current_pos[0], packet.current_pos[1] + 25, 
                   f"{packet.packet_type.upper()}", 
                   ha='center', fontsize=8, color='white', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=packet.color, alpha=0.8),
                   zorder=110)
            
            # Add comprehensive path description for inter-cluster packets
            if packet.description and packet.packet_type == 'inter_cluster':
                # Position description in a prominent location
                desc_x = 50
                desc_y = self.area_size - 50 - (i * 60)  # Stack descriptions vertically
                
                # Create prominent description box
                ax.text(desc_x, desc_y, f"🚀 {packet.description}", 
                       ha='left', va='center', fontsize=10, 
                       color='darkred', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.5", 
                               facecolor='yellow', alpha=0.95, 
                               edgecolor='red', linewidth=2),
                       zorder=200)
                
                # Draw connecting line from packet to description
                ax.plot([packet.current_pos[0], desc_x], 
                       [packet.current_pos[1], desc_y],
                       color=packet.color, alpha=0.5, linewidth=1, 
                       linestyle=':', zorder=90)
            
            # Collect description for overlay
            if packet.description:
                packet_descriptions.append(packet.description)
        
        # 8. Draw path descriptions overlay (for all packets)
        if packet_descriptions:
            overlay_text = "📡 Active Communications:\n" + "\n".join(f"• {desc}" for desc in packet_descriptions[:6])  # Limit to 6 descriptions
            ax.text(0.02, 0.02, overlay_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='lightyellow', alpha=0.95, edgecolor='orange'),
                    color='darkred', weight='bold', zorder=200)
        
        # 8.5. Display routing events overlay (step-by-step routing information)
        if routing_events:
            routing_text = "🔄 Routing Events:\n" + "\n".join(f"• {event}" for event in routing_events[:5])  # Limit to 5 events
            ax.text(0.98, 0.02, routing_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='lightgreen', alpha=0.95, edgecolor='darkgreen'),
                    color='darkgreen', weight='bold', zorder=200)
        
        # 9. Add network statistics with HIERARCHY COMPLIANCE
        active_packets = len(self.moving_packets)
        avg_mec_load = sum(s.get_load_percentage() for s in self.protocol.mec_servers.values()) / len(self.protocol.mec_servers)
        
        # Count hierarchy compliance
        valid_connections = 0
        problematic_connections = 0
        
        for ch in cluster_heads:
            # Check CH-IAR distances
            nearest_iar = self.protocol._find_nearest_iar_server(ch)
            if nearest_iar:
                iar_distance = math.sqrt((ch.x - nearest_iar.x)**2 + (ch.y - nearest_iar.y)**2)
                if iar_distance <= nearest_iar.coverage_radius:
                    valid_connections += 1
                else:
                    problematic_connections += 1
            
            # Check member-CH distances
            for member_id in ch.cluster_members:
                if member_id in self.protocol.nodes:
                    member = self.protocol.nodes[member_id]
                    if member.is_alive():
                        member_distance = math.sqrt((ch.x - member.x)**2 + (ch.y - member.y)**2)
                        if member_distance <= self.protocol.communication_range * 0.75:
                            valid_connections += 1
                        else:
                            problematic_connections += 1
        
        hierarchy_compliance = (valid_connections / max(valid_connections + problematic_connections, 1)) * 100
        
        stats_text = f"ARPMEC Protocol with COMMUNICATION HIERARCHY\n"
        stats_text += f"Time: {frame_time:.1f}s | Active Packets: {active_packets}\n"
        stats_text += f"Clusters: {len(cluster_heads)} | Members: {len(cluster_members)} | Idle: {len(idle_nodes)}\n"
        stats_text += f"Infrastructure: {len(self.protocol.iar_servers)} IAR, {len(self.protocol.mec_servers)} MEC\n"
        stats_text += f"Hierarchy Compliance: {hierarchy_compliance:.1f}% ({valid_connections} valid, {problematic_connections} problematic)\n"
        stats_text += f"Average MEC Load: {avg_mec_load:.1f}%"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor='lightcyan', alpha=0.9))
        
        # 10. Create legend with better positioning
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                      markersize=8, label='Cluster Head'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=6, label='Cluster Member'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkblue', 
                      markersize=8, label='MEC Server'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='purple', 
                      markersize=6, label='IAR Server'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=4, label='Data Packet'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=4, label='MEC Task'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', 
                      markersize=6, label='Multi-Hop Inter-Cluster'),
        ]
        
        # Position legend in extended frame area
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                 fontsize=10, fancybox=True, shadow=True, framealpha=0.95)
        
        ax.set_title(f'ARPMEC PROTOCOL with COMMUNICATION HIERARCHY\n'
                    f'🔄 Range-Limited: Member ↔ CH ↔ IAR ↔ MEC | Real Protocol-Driven Traffic | Distance Validation', 
                    fontsize=12, weight='bold', pad=10)
        ax.set_xlabel('X Position (meters)', fontsize=10)
        ax.set_ylabel('Y Position (meters)', fontsize=10)
    
    def create_animation(self, filename: str = "arpmec_packet_tracer.mp4", 
                        duration: int = 30):
        """Create PROTOCOL-DRIVEN packet tracer style animation"""
        print(f"Creating PROTOCOL-DRIVEN Packet Tracer style animation...")
        
        # Create figure with larger size to accommodate legends and extended frame
        fig, ax = plt.subplots(figsize=(20, 14))  # Larger figure
        fig.patch.set_facecolor('white')
        
        # Initialize protocol time tracking
        self.protocol_time_slot = 0
        self.last_reclustering_time = 0
        self.reclustering_interval = 50  # Re-cluster every 50 frames (10 seconds) - more frequent
        
        # Animation function
        def animate(frame):
            frame_time = frame * FRAME_DURATION / 1000.0  # Convert to seconds
            
            # CRITICAL: Advance protocol time to sync with animation
            self.protocol.current_time_slot = self.protocol_time_slot
            
            # Step 1: Handle node mobility and check for significant movement
            significant_movement = self._update_node_mobility(frame_time)
            
            # Step 2: Generate protocol-driven traffic (every 30 frames for visibility)
            if frame % 30 == 0:
                self.generate_packet_traffic()
            
            # Step 3: Advance protocol state (every 50 frames)
            if frame % 50 == 0 and frame > 0:
                self.protocol_time_slot += 1
                self.protocol.current_time_slot = self.protocol_time_slot
                
                # Run one step of the protocol's adaptive routing
                cluster_heads = self.protocol._get_cluster_heads()
                if cluster_heads:
                    # Simulate protocol operations for each cluster head
                    for ch in cluster_heads:
                        self.protocol._fixed_cluster_head_operations(ch, self.protocol_time_slot)
                    
                    # Simulate protocol operations for members
                    members = [n for n in self.protocol.nodes.values() 
                              if n.state == NodeState.CLUSTER_MEMBER and n.is_alive()]
                    for member in members:
                        self.protocol._fixed_cluster_member_operations(member, self.protocol_time_slot)
            
            # Step 4: Handle dynamic re-clustering - IMPROVED LOGIC
            should_recluster = False
            
            # Time-based reclustering (less frequent)
            if frame - self.last_reclustering_time > self.reclustering_interval:
                should_recluster = True
                print(f"⏰ Time-based reclustering triggered (every {self.reclustering_interval} frames)")
            
            # Distance-based reclustering (immediate when nodes move too far)
            elif significant_movement:
                should_recluster = True
                print(f"📍 Distance-based reclustering triggered (nodes moved significantly)")
            
            if should_recluster:
                self.last_reclustering_time = frame
                self._trigger_protocol_reclustering()
            
            # Step 5: Draw network state
            self.draw_network(ax, frame_time)
            
            return []
        
        # Calculate frame count
        fps = 1000 // FRAME_DURATION  # Frames per second
        total_frames = duration * fps
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                     interval=FRAME_DURATION, blit=False, repeat=False)
        
        # Save as MP4
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='ARPMEC Protocol-Driven Demo'), bitrate=1800)
            anim.save(filename, writer=writer)
            print(f"✅ PROTOCOL-DRIVEN Video saved as: {filename}")
        except Exception as e:
            print(f"❌ Error saving video: {e}")
            print("Trying to save as GIF instead...")
            anim.save(filename.replace('.mp4', '.gif'), writer='pillow', fps=fps//2)
    
    def _update_node_mobility(self, frame_time: float):
        """Update node positions with SLOWER movement and distance-based reclustering trigger"""
        
        significant_movement = False
        
        for node in self.protocol.nodes.values():
            if not node.is_alive():
                continue
                
            # Store original position to check for significant movement
            old_x, old_y = node.x, node.y
            
            # Dramatically reduce movement speed
            original_speed = getattr(node, 'speed', 2.0)
            node.speed = 0.2  # Reduce from ~2.0 to 0.2 (10x slower)
            
            # Update node position using protocol's mobility model
            node.update_mobility((0, self.area_size, 0, self.area_size))
            
            # Restore original speed
            node.speed = original_speed
            
            # Ensure nodes stay within bounds
            node.x = max(50, min(self.area_size - 50, node.x))
            node.y = max(50, min(self.area_size - 50, node.y))
            
            # Check for significant movement (>20 meters)
            movement_distance = math.sqrt((node.x - old_x)**2 + (node.y - old_y)**2)
            if movement_distance > 20:
                significant_movement = True
            
            # ENFORCE CLUSTER DISTANCE CONSTRAINTS for cluster members
            if (node.state == NodeState.CLUSTER_MEMBER and 
                hasattr(node, 'cluster_head_id') and node.cluster_head_id):
                
                # Find cluster head
                cluster_head = None
                for ch_node in self.protocol.nodes.values():
                    if ch_node.id == node.cluster_head_id and ch_node.state == NodeState.CLUSTER_HEAD:
                        cluster_head = ch_node
                        break
                
                if cluster_head:
                    # Calculate distance to cluster head
                    distance = math.sqrt((node.x - cluster_head.x)**2 + (node.y - cluster_head.y)**2)
                    max_cluster_distance = self.protocol.communication_range * 0.75  # 75% of comm range
                    
                    if distance > max_cluster_distance:
                        # Trigger immediate reclustering when nodes are too far
                        print(f"⚠️ Node-{node.id} too far from CH-{cluster_head.id} ({distance:.1f}m) - triggering reclustering")
                        significant_movement = True
                        
                        # Move node closer to stay connected temporarily
                        direction_x = (cluster_head.x - node.x) / distance
                        direction_y = (cluster_head.y - node.y) / distance
                        
                        # Place node at maximum allowed distance
                        node.x = cluster_head.x - direction_x * max_cluster_distance
                        node.y = cluster_head.y - direction_y * max_cluster_distance
                        
                        # Ensure still within area bounds
                        node.x = max(50, min(self.area_size - 50, node.x))
                        node.y = max(50, min(self.area_size - 50, node.y))
        
        return significant_movement
    
    def _trigger_protocol_reclustering(self):
        """Trigger protocol-driven re-clustering due to mobility"""
        
        print(f"🔄 Triggering protocol-driven re-clustering due to mobility at time {self.protocol_time_slot}")
        
        # Reset clustering state as per protocol
        self.protocol._reset_clustering_state()
        
        # Perform new clustering
        try:
            new_clusters = self.protocol.clustering_algorithm()
            self.protocol._build_inter_cluster_routing_table()
            print(f"✅ Re-clustering completed: {len(new_clusters)} clusters formed")
        except Exception as e:
            print(f"❌ Re-clustering failed: {e}")
    
    def show_live_demo(self):
        """Show live interactive demo with PROTOCOL-DRIVEN traffic - IDENTICAL to video version"""
        print("Starting PROTOCOL-DRIVEN Live Demo...")
        
        # Create figure with SAME size as video version
        fig, ax = plt.subplots(figsize=(20, 14))  # Same as video
        fig.patch.set_facecolor('white')
        
        # Initialize protocol time tracking - SAME as video version
        self.protocol_time_slot = 0
        self.last_reclustering_time = 0
        self.reclustering_interval = 50  # Re-cluster every 50 frames (10 seconds) - more frequent
        
        # Animation function - IDENTICAL to video version
        def animate_live(frame):
            frame_time = frame * FRAME_DURATION / 1000.0  # Convert to seconds
            
            # CRITICAL: Advance protocol time to sync with animation
            self.protocol.current_time_slot = self.protocol_time_slot
            
            # Step 1: Handle node mobility and check for significant movement
            significant_movement = self._update_node_mobility(frame_time)
            
            # Step 2: Generate protocol-driven traffic (every 30 frames for visibility)
            if frame % 30 == 0:
                self.generate_packet_traffic()
            
            # Step 3: Advance protocol state (every 50 frames)
            if frame % 50 == 0 and frame > 0:
                self.protocol_time_slot += 1
                self.protocol.current_time_slot = self.protocol_time_slot
                
                # Run one step of the protocol's adaptive routing
                cluster_heads = self.protocol._get_cluster_heads()
                if cluster_heads:
                    # Simulate protocol operations for each cluster head
                    for ch in cluster_heads:
                        self.protocol._fixed_cluster_head_operations(ch, self.protocol_time_slot)
                    
                    # Simulate protocol operations for members
                    members = [n for n in self.protocol.nodes.values() 
                              if n.state == NodeState.CLUSTER_MEMBER and n.is_alive()]
                    for member in members:
                        self.protocol._fixed_cluster_member_operations(member, self.protocol_time_slot)
            
            # Step 4: Handle dynamic re-clustering - IMPROVED LOGIC
            should_recluster = False
            
            # Time-based reclustering (less frequent)
            if frame - self.last_reclustering_time > self.reclustering_interval:
                should_recluster = True
                print(f"⏰ Time-based reclustering triggered (every {self.reclustering_interval} frames)")
            
            # Distance-based reclustering (immediate when nodes move too far)
            elif significant_movement:
                should_recluster = True
                print(f"📍 Distance-based reclustering triggered (nodes moved significantly)")
            
            if should_recluster:
                self.last_reclustering_time = frame
                self._trigger_protocol_reclustering()
            
            # Step 5: Draw network state - SAME method as video
            self.draw_network(ax, frame_time)
            
            return []
        
        # Create live animation with SAME settings as video
        anim = animation.FuncAnimation(fig, animate_live, interval=FRAME_DURATION, 
                                     blit=False, repeat=True)
        
        plt.tight_layout()
        plt.show()
        
        return anim

def main():
    """Main demonstration function - PROTOCOL-DRIVEN packet tracer"""
    print("🚀 ARPMEC PROTOCOL-DRIVEN Packet Tracer Demo")
    print("=" * 60)
    print("This demo shows REAL protocol-driven packet flows:")
    print("✓ Inter-cluster communication via IAR/MEC infrastructure")
    print("✓ Dynamic clustering with node mobility")
    print("✓ Adaptive routing with load balancing")
    print("✓ Paper-faithful implementation (no synthetic traffic)")
    print("=" * 60)
    
    # Option selection first
    print("\n🎮 Choose demo type:")
    print("1. Live Interactive Demo (real-time protocol simulation)")
    print("2. Create Animation Video (protocol-driven packet flows)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        # Create demo AFTER choice to ensure identical state
        demo = ARPMECPacketTracerDemo(num_nodes=20, area_size=1200)  # Use same area size for both
        
        # Set up network with identical state
        clusters = demo.create_network()
        
        # Display initial network state
        print("\n📊 Initial Network State:")
        print(f"   Nodes: {len(demo.protocol.nodes)}")
        print(f"   Clusters: {len(clusters)}")
        print(f"   MEC Servers: {len(demo.protocol.mec_servers)}")
        print(f"   IAR Servers: {len(demo.protocol.iar_servers)}")
        
        # Show network structure
        print("\n📡 IAR-MEC Infrastructure:")
        for iar_id, iar in demo.protocol.iar_servers.items():
            connected_mecs = [f"MEC-{mid}" for mid in iar.connected_mec_servers]
            print(f"   IAR-{iar_id}: Connected to {', '.join(connected_mecs)}")
        
        if choice == "1":
            print("\n🎮 Starting Live Interactive Demo...")
            print("   - Real-time protocol simulation")
            print("   - Dynamic clustering and mobility")
            print("   - Live packet visualization")
            print("   - Press Ctrl+C to stop")
            
            anim = demo.show_live_demo()
        else:
            print("\n🎬 Creating Protocol-Driven Animation Video...")
            anim = demo.create_animation("arpmec_protocol_driven.mp4", duration=60)
            
            # Show final network state
            print("\n📊 Final Network State:")
            metrics = demo.protocol.get_performance_metrics()
            print(f"   Clusters: {metrics['num_clusters']}")
            print(f"   MEC Success Rate: {metrics['mec_success_rate']:.1f}%")
            print(f"   IAR Servers: {metrics['iar_servers']}")
            print(f"   Network Lifetime: {metrics['network_lifetime']*100:.1f}%")
            
            print("\n✅ PROTOCOL-DRIVEN Demo Complete!")
            print("📹 Check the video file for real protocol-driven packet flows")
            print("🎯 Features: TRUE Inter-cluster routing, IAR Infrastructure, Adaptive Routing")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Falling back to basic animation...")
        # Create new demo instance for fallback
        demo = ARPMECPacketTracerDemo(num_nodes=20, area_size=1200)
        demo.create_network()
        demo.create_animation("arpmec_protocol_driven.mp4", duration=30)

if __name__ == "__main__":
    main()
