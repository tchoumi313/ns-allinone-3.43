#!/usr/bin/env python3
"""
Enhanced ARPMEC Visualization Demo with IAR Infrastructure and Packet Animation
Creates Packet Tracer-style visualizations with animated packet movements
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from arpmec_faithful import (ARPMECProtocol, IARServer, MECServer, Node,
                             NodeState)

# Set up matplotlib for better visualization
plt.style.use('default')
plt.rcParams['figure.figsize'] = (18, 14)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

@dataclass
class AnimatedPacket:
    """Animated packet for Packet Tracer-style visualization"""
    packet_id: str
    source_pos: Tuple[float, float]
    dest_pos: Tuple[float, float]
    via_positions: List[Tuple[float, float]]  # For multi-hop paths
    current_pos: Tuple[float, float]
    progress: float  # 0.0 to 1.0
    packet_type: str  # 'data', 'control', 'hello', 'mec_task'
    color: str
    size: int
    speed: float  # Animation speed
    active: bool = True
    current_hop: int = 0  # Current hop in via_positions
    
    def update_position(self, dt: float):
        """Update packet position for animation"""
        if not self.active:
            return
            
        # If we have via positions, animate hop by hop
        if self.via_positions:
            if self.current_hop < len(self.via_positions):
                # Current segment
                if self.current_hop == 0:
                    start_pos = self.source_pos
                else:
                    start_pos = self.via_positions[self.current_hop - 1]
                    
                if self.current_hop < len(self.via_positions):
                    end_pos = self.via_positions[self.current_hop]
                else:
                    end_pos = self.dest_pos
                
                # Animate along current segment
                self.progress += self.speed * dt
                
                if self.progress >= 1.0:
                    # Move to next hop
                    self.current_hop += 1
                    self.progress = 0.0
                    
                    # Check if we've reached the destination
                    if self.current_hop >= len(self.via_positions):
                        # Final hop to destination
                        start_pos = self.via_positions[-1]
                        end_pos = self.dest_pos
                        
                # Interpolate position
                if self.current_hop < len(self.via_positions):
                    end_pos = self.via_positions[self.current_hop]
                else:
                    end_pos = self.dest_pos
                    
                t = min(self.progress, 1.0)
                self.current_pos = (
                    start_pos[0] + t * (end_pos[0] - start_pos[0]),
                    start_pos[1] + t * (end_pos[1] - start_pos[1])
                )
                
                # Deactivate if reached final destination
                if self.current_hop >= len(self.via_positions) and self.progress >= 1.0:
                    self.current_pos = self.dest_pos
                    self.active = False
            else:
                self.active = False
        else:
            # Direct path animation
            self.progress += self.speed * dt
            if self.progress >= 1.0:
                self.progress = 1.0
                self.active = False
                
            # Interpolate position
            t = self.progress
            self.current_pos = (
                self.source_pos[0] + t * (self.dest_pos[0] - self.source_pos[0]),
                self.source_pos[1] + t * (self.dest_pos[1] - self.source_pos[1])
            )

def create_realistic_network(N: int, area_size: int) -> List[Node]:
    """Create a realistic network with some clustered groups"""
    nodes = []
    random.seed(42)  # Fixed seed for reproducible results
    
    # Create some clustered groups to ensure cluster formation
    cluster_centers = [
        (area_size * 0.25, area_size * 0.25),  # Bottom-left cluster
        (area_size * 0.75, area_size * 0.25),  # Bottom-right cluster  
        (area_size * 0.5, area_size * 0.75),   # Top-center cluster
    ]
    
    nodes_per_cluster = N // len(cluster_centers)
    remaining_nodes = N % len(cluster_centers)
    
    node_id = 0
    
    # Create clustered nodes
    for i, (cx, cy) in enumerate(cluster_centers):
        cluster_size = nodes_per_cluster + (1 if i < remaining_nodes else 0)
        
        for j in range(cluster_size):
            # Place nodes within 80m of cluster center (within communication range)
            angle = random.uniform(0, 2 * 3.14159)
            radius = random.uniform(0, 80)  # Within communication range
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            
            # Keep within area bounds
            x = max(50, min(area_size - 50, x))
            y = max(50, min(area_size - 50, y))
            
            energy = random.uniform(90, 110)
            nodes.append(Node(node_id, x, y, energy))
            node_id += 1
    
    print(f"Created {len(nodes)} nodes in {len(cluster_centers)} clustered groups")
    return nodes

class PacketTracerStyleARPMECDemo:
    """Packet Tracer-style ARPMEC demonstration with animated packet movements"""
    
    def __init__(self, N: int = 20, area_size: int = 800):
        self.N = N
        self.area_size = area_size
        self.protocol = None
        self.rounds = 0
        self.max_rounds = 40
        
        # Enhanced color palette for different components
        self.cluster_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        
        # Packet colors for different types
        self.packet_colors = {
            'hello': '#FFD700',      # Gold for HELLO messages
            'data': '#32CD32',       # Green for data packets
            'control': '#FF69B4',    # Pink for control messages
            'mec_task': '#FF4500',   # Orange for MEC tasks
            'inter_cluster': '#8A2BE2'  # Blue-violet for inter-cluster
        }
        
        # Infrastructure colors
        self.infra_colors = {
            'mec_server': '#8B0000',   # Dark red for MEC servers
            'iar_server': '#4B0082',   # Indigo for IAR servers
            'ch_node': '#FF0000',      # Red for cluster heads
            'member_node': '#0000FF',  # Blue for members
            'idle_node': '#808080'     # Gray for idle nodes
        }
        
        # Animation storage
        self.animated_packets = []
        self.packet_counter = 0
        
        # Communication tracking for realistic packet generation
        self.recent_communications = []
        
    def create_animated_packet(self, source_pos: Tuple[float, float], 
                              dest_pos: Tuple[float, float],
                              via_positions: List[Tuple[float, float]] = None,
                              packet_type: str = 'data',
                              speed: float = 0.02) -> AnimatedPacket:
        """Create an animated packet"""
        self.packet_counter += 1
        
        return AnimatedPacket(
            packet_id=f"pkt_{self.packet_counter}",
            source_pos=source_pos,
            dest_pos=dest_pos,
            via_positions=via_positions or [],
            current_pos=source_pos,
            progress=0.0,
            packet_type=packet_type,
            color=self.packet_colors.get(packet_type, '#FFFFFF'),
            size=80 if packet_type == 'mec_task' else 60,
            speed=speed,
            active=True,
            current_hop=0
        )
    
    def generate_realistic_packet_animations(self):
        """Generate realistic packet animations based on protocol operations"""
        # Clear old inactive packets
        self.animated_packets = [p for p in self.animated_packets if p.active]
        
        cluster_heads = self.protocol._get_cluster_heads()
        cluster_members = [n for n in self.protocol.nodes.values() 
                          if n.state == NodeState.CLUSTER_MEMBER and n.is_alive()]
        
        # 1. Member-to-CH data packets (most frequent)
        for member in cluster_members:
            if member.cluster_head_id and random.random() < 0.3:  # 30% chance
                ch = next((ch for ch in cluster_heads if ch.id == member.cluster_head_id), None)
                if ch:
                    packet = self.create_animated_packet(
                        (member.x, member.y), (ch.x, ch.y),
                        packet_type='data', speed=0.03
                    )
                    self.animated_packets.append(packet)
        
        # 2. CH-to-IAR-to-MEC task offloading
        for ch in cluster_heads:
            if random.random() < 0.2:  # 20% chance for MEC task
                nearest_iar = self.protocol._find_nearest_iar_server(ch)
                if nearest_iar:
                    # Find best MEC for this IAR
                    best_mec = None
                    if nearest_iar.connected_mec_servers:
                        mec_id = nearest_iar.connected_mec_servers[0]
                        best_mec = self.protocol.mec_servers.get(mec_id)
                    
                    if best_mec:
                        # CH -> IAR -> MEC packet animation
                        packet = self.create_animated_packet(
                            (ch.x, ch.y), (best_mec.x, best_mec.y),
                            via_positions=[(nearest_iar.x, nearest_iar.y)],
                            packet_type='mec_task', speed=0.025
                        )
                        self.animated_packets.append(packet)
        
        # 3. Inter-cluster communication via IAR and MEC
        if len(cluster_heads) > 1 and random.random() < 0.15:  # 15% chance
            source_ch = random.choice(cluster_heads)
            target_ch = random.choice([ch for ch in cluster_heads if ch.id != source_ch.id])
            
            source_iar = self.protocol._find_nearest_iar_server(source_ch)
            target_iar = self.protocol._find_nearest_iar_server(target_ch)
            
            if source_iar and target_iar:
                # Find MEC servers for routing
                source_mec = None
                target_mec = None
                
                if source_iar.connected_mec_servers:
                    mec_id = source_iar.connected_mec_servers[0]
                    source_mec = self.protocol.mec_servers.get(mec_id)
                    
                if target_iar.connected_mec_servers:
                    mec_id = target_iar.connected_mec_servers[0]
                    target_mec = self.protocol.mec_servers.get(mec_id)
                
                if source_mec and target_mec:
                    # Multi-hop: CH -> IAR -> MEC -> MEC -> IAR -> CH
                    via_positions = [
                        (source_iar.x, source_iar.y),
                        (source_mec.x, source_mec.y),
                        (target_mec.x, target_mec.y),
                        (target_iar.x, target_iar.y)
                    ]
                    
                    packet = self.create_animated_packet(
                        (source_ch.x, source_ch.y), (target_ch.x, target_ch.y),
                        via_positions=via_positions,
                        packet_type='inter_cluster', speed=0.02
                    )
                    self.animated_packets.append(packet)
        
    def setup_network(self):
        """Set up the network with realistic parameters"""
        print("Setting up crystal clear network...")
        
        # Create nodes with good distribution
        random.seed(42)
        np.random.seed(42)
        
        nodes = create_realistic_network(self.N, self.area_size)
        self.protocol = ARPMECProtocol(nodes, C=4, R=5, K=3)
        
        # Initial clustering
        print("Performing initial clustering...")
        clusters = self.protocol.clustering_algorithm()
        print(f"Created {len(clusters)} clusters")
        
        return clusters
    
    def create_enhanced_frame(self, round_num: int, title_suffix: str = "") -> plt.Figure:
        """Create an enhanced frame with IAR infrastructure and animated packets"""
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Enhanced background
        ax.set_facecolor('#F0F8FF')  # Alice blue background
        ax.set_xlim(-50, self.area_size + 50)
        ax.set_ylim(-50, self.area_size + 50)
        
        # Get current network state
        cluster_heads = self.protocol._get_cluster_heads()
        cluster_members = [n for n in self.protocol.nodes.values() 
                          if n.state == NodeState.CLUSTER_MEMBER and n.is_alive()]
        idle_nodes = [n for n in self.protocol.nodes.values() 
                     if n.state == NodeState.IDLE and n.is_alive()]
        
        # Create cluster color mapping
        cluster_color_map = {}
        for i, ch in enumerate(cluster_heads):
            cluster_color_map[ch.id] = self.cluster_colors[i % len(self.cluster_colors)]
        
        # 1. Draw IAR servers first (NEW INFRASTRUCTURE)
        for iar in self.protocol.iar_servers.values():
            # IAR coverage area
            coverage = plt.Circle((iar.x, iar.y), iar.coverage_radius, 
                                fill=True, color='indigo', alpha=0.15, zorder=1)
            ax.add_patch(coverage)
            
            # IAR server as diamond
            diamond_x = [iar.x, iar.x + 25, iar.x, iar.x - 25, iar.x]
            diamond_y = [iar.y + 25, iar.y, iar.y - 25, iar.y, iar.y + 25]
            ax.plot(diamond_x, diamond_y, color='indigo', linewidth=4)
            ax.fill(diamond_x, diamond_y, color=self.infra_colors['iar_server'], alpha=0.8, zorder=3)
            
            # IAR label
            ax.annotate(f'IAR-{iar.id}', (iar.x, iar.y), 
                       xytext=(0, -35), textcoords='offset points',
                       fontsize=12, fontweight='bold', ha='center', color='indigo')
            
            # Show connected clusters count
            cluster_count = len(iar.connected_clusters)
            ax.annotate(f'{cluster_count} clusters', (iar.x, iar.y), 
                       xytext=(0, -50), textcoords='offset points',
                       fontsize=10, ha='center', color='indigo')
        
        # 2. Draw IAR-to-MEC connections (INFRASTRUCTURE BACKBONE)
        for iar in self.protocol.iar_servers.values():
            for mec_id in iar.connected_mec_servers:
                if mec_id in self.protocol.mec_servers:
                    mec = self.protocol.mec_servers[mec_id]
                    ax.plot([iar.x, mec.x], [iar.y, mec.y],
                           color='purple', linewidth=3, alpha=0.6, 
                           linestyle=':', zorder=2, label='IAR-MEC Link' if iar.id == 1 and mec_id == 1 else "")
        
        # 3. Draw MEC servers (ENHANCED)
        for server in self.protocol.mec_servers.values():
            # MEC server coverage area
            coverage = plt.Circle((server.x, server.y), 100, 
                                fill=True, color='darkred', alpha=0.1, zorder=1)
            ax.add_patch(coverage)
            
            # MEC server as large square with load indicator
            load_pct = server.get_load_percentage()
            load_color = 'red' if load_pct > 80 else 'orange' if load_pct > 60 else 'green'
            
            ax.scatter(server.x, server.y, c=self.infra_colors['mec_server'], s=800, marker='s', 
                      edgecolors='black', linewidth=4, alpha=0.9, zorder=4)
            
            # Load indicator ring
            load_ring = plt.Circle((server.x, server.y), 35, 
                                 fill=False, color=load_color, linewidth=6, alpha=0.8, zorder=4)
            ax.add_patch(load_ring)
            
            ax.annotate(f'MEC-{server.id}', (server.x, server.y), 
                       xytext=(0, -45), textcoords='offset points',
                       fontsize=14, fontweight='bold', ha='center', color='darkred')
            ax.annotate(f'{load_pct:.0f}% load', (server.x, server.y), 
                       xytext=(0, -60), textcoords='offset points',
                       fontsize=10, ha='center', color=load_color, fontweight='bold')
        
        # 4. Draw cluster areas and boundaries
        for ch in cluster_heads:
            if ch.id in cluster_color_map:
                color = cluster_color_map[ch.id]
                
                # Cluster core area
                cluster_area = plt.Circle((ch.x, ch.y), 15,
                                        fill=True, color=color, alpha=0.3, zorder=2)
                ax.add_patch(cluster_area)
                
                # Communication range boundary
                boundary = plt.Circle((ch.x, ch.y), self.protocol.communication_range,
                                    fill=False, color=color, alpha=0.8, 
                                    linestyle='--', linewidth=2, zorder=2)
                ax.add_patch(boundary)
        
        # 5. Draw enhanced connections
        for ch in cluster_heads:
            if ch.id in cluster_color_map:
                color = cluster_color_map[ch.id]
                
                # CH to nearest IAR connection
                nearest_iar = self.protocol._find_nearest_iar_server(ch)
                if nearest_iar:
                    ax.plot([ch.x, nearest_iar.x], [ch.y, nearest_iar.y],
                           color=color, linewidth=5, alpha=0.7, zorder=4,
                           label='CH-IAR Link' if ch.id == cluster_heads[0].id else "")
                
                # Members to CH connections
                for member in cluster_members:
                    if member.cluster_head_id == ch.id:
                        ax.plot([member.x, ch.x], [member.y, ch.y],
                               color=color, linewidth=2, alpha=0.6, zorder=4)
        
        # 6. Draw animated packets (PACKET TRACER STYLE)
        for packet in self.animated_packets:
            if packet.active:
                # Draw packet as moving circle with trail effect
                ax.scatter(packet.current_pos[0], packet.current_pos[1], 
                          c=packet.color, s=packet.size, marker='o',
                          edgecolors='black', linewidth=2, alpha=0.9, zorder=10)
                
                # Add packet type label
                ax.annotate(packet.packet_type.upper(), packet.current_pos, 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, fontweight='bold', color='black',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                # Draw trail effect (previous positions)
                if hasattr(packet, 'trail_positions'):
                    for i, trail_pos in enumerate(packet.trail_positions[-5:]):  # Last 5 positions
                        alpha = 0.2 * (i + 1) / 5  # Fading trail
                        ax.scatter(trail_pos[0], trail_pos[1], 
                                  c=packet.color, s=packet.size * 0.5, 
                                  alpha=alpha, zorder=9)
        
        # 7. Draw nodes with enhanced symbols
        # Cluster heads
        for ch in cluster_heads:
            if ch.id in cluster_color_map:
                color = cluster_color_map[ch.id]
                ax.scatter(ch.x, ch.y, c=color, s=500, marker='^',
                          edgecolors='black', linewidth=3, zorder=6)
                ax.annotate(f'CH-{ch.id}', (ch.x, ch.y), xytext=(8, 8),
                           textcoords='offset points', fontsize=12, fontweight='bold')
                
                # Energy indicator
                energy_pct = (ch.energy / ch.initial_energy) * 100
                energy_color = 'red' if energy_pct < 30 else 'orange' if energy_pct < 60 else 'green'
                ax.annotate(f'{energy_pct:.0f}%', (ch.x, ch.y), xytext=(8, -15),
                           textcoords='offset points', fontsize=10, color=energy_color, fontweight='bold')
        
        # Cluster members
        for member in cluster_members:
            if member.cluster_head_id and member.cluster_head_id in cluster_color_map:
                color = cluster_color_map[member.cluster_head_id]
                ax.scatter(member.x, member.y, c=color, s=250, marker='o',
                          alpha=0.9, edgecolors='black', linewidth=2, zorder=7)
                
                ax.annotate(f'M{member.id}', (member.x, member.y), xytext=(-5, -18),
                           textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Idle nodes
        if idle_nodes:
            for node in idle_nodes:
                ax.scatter(node.x, node.y, c='lightgray', s=120, marker='s',
                          alpha=0.7, edgecolors='black', linewidth=1, zorder=5)
        
        # 8. Enhanced legend with IAR and packet types
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                      markersize=15, label='Cluster Heads'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=12, label='Cluster Members'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkred', 
                      markersize=15, label='MEC Servers'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='indigo', 
                      markersize=12, label='IAR Servers'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', 
                      markersize=8, label='Idle Nodes'),
            plt.Line2D([0], [0], color='purple', linewidth=3, linestyle=':', 
                      label='IAR-MEC Backbone'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
                      markersize=10, label='Data Packets'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=10, label='MEC Tasks'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blueviolet', 
                      markersize=10, label='Inter-cluster Msgs')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        # 9. Enhanced title and info
        if round_num % 5 == 0 and round_num > 0:
            title_suffix = f" 🔄 (RE-CLUSTERING EVENT!)"
        elif title_suffix == "":
            next_recluster = ((round_num // 5) + 1) * 5
            title_suffix = f" (Next re-clustering: Round {next_recluster})"
            
        main_title = f"ARPMEC Protocol with IAR Infrastructure - Round {round_num}{title_suffix}"
        subtitle = f"CH → IAR → MEC Communication | Adaptive Routing | Real-time Packet Animation"
        ax.set_title(f"{main_title}\n{subtitle}", fontsize=16, fontweight='bold', pad=20)
        
        # 10. Enhanced network stats
        active_packets = len([p for p in self.animated_packets if p.active])
        total_mec_load = sum(s.get_load_percentage() for s in self.protocol.mec_servers.values()) / len(self.protocol.mec_servers)
        
        stats_text = f"Infrastructure: {len(self.protocol.iar_servers)} IAR, {len(self.protocol.mec_servers)} MEC"
        stats_text += f"\nClusters: {len(cluster_heads)} | Members: {len(cluster_members)} | Idle: {len(idle_nodes)}"
        stats_text += f"\nActive Packets: {active_packets} | Avg MEC Load: {total_mec_load:.1f}%"
        
        # Show IAR connectivity
        iar_stats = []
        for iar in list(self.protocol.iar_servers.values())[:3]:  # Show first 3 IARs
            cluster_count = len(iar.connected_clusters)
            mec_count = len(iar.connected_mec_servers)
            iar_stats.append(f"IAR-{iar.id}: {cluster_count}C/{mec_count}M")
        
        if iar_stats:
            stats_text += f"\n{' | '.join(iar_stats)}"
        
        if round_num % 5 == 0 and round_num > 0:
            stats_text += f"\n🔄 RE-CLUSTERING ACTIVE!"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor='yellow' if round_num % 5 == 0 and round_num > 0 else 'lightcyan', 
                alpha=0.9))
        
        # Grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def create_clean_frame(self, round_num: int, title_suffix: str = ""):
        """Create a clean frame for the current round - FIXED METHOD"""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Set up the plot
        ax.set_xlim(-50, self.area_size + 50)
        ax.set_ylim(-50, self.area_size + 50)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        # Title with round information
        ax.set_title(f'ARPMEC Protocol with IAR Infrastructure - Round {round_num}\n'
                    f'Enhanced Adaptive Routing with Packet Animation', 
                    fontsize=16, weight='bold', pad=20)
        
        # Get current network state
        cluster_heads = self.protocol._get_cluster_heads()
        cluster_members = [n for n in self.protocol.nodes.values() 
                          if n.state == NodeState.CLUSTER_MEMBER and n.is_alive()]
        idle_nodes = [n for n in self.protocol.nodes.values() 
                     if n.state == NodeState.IDLE and n.is_alive()]
        
        # Create cluster color mapping
        cluster_color_map = {}
        for i, ch in enumerate(cluster_heads):
            cluster_color_map[ch.id] = self.cluster_colors[i % len(self.cluster_colors)]
        
        # 1. Draw IAR servers first
        for iar in self.protocol.iar_servers.values():
            # IAR coverage area
            coverage = plt.Circle((iar.x, iar.y), iar.coverage_radius, 
                                fill=True, color='purple', alpha=0.15, zorder=1)
            ax.add_patch(coverage)
            
            # IAR server as large diamond
            diamond_x = [iar.x, iar.x + 20, iar.x, iar.x - 20, iar.x]
            diamond_y = [iar.y + 20, iar.y, iar.y - 20, iar.y, iar.y + 20]
            ax.plot(diamond_x, diamond_y, color='purple', linewidth=3)
            ax.fill(diamond_x, diamond_y, color='mediumpurple', alpha=0.8, zorder=3)
            
            # IAR label
            ax.text(iar.x, iar.y - 35, f'IAR-{iar.id}', 
                   fontsize=11, ha='center', weight='bold', color='purple')
            ax.text(iar.x, iar.y - 45, f'{len(iar.connected_clusters)} clusters', 
                   fontsize=8, ha='center', color='gray')
        
        # 2. Draw MEC servers
        for server in self.protocol.mec_servers.values():
            # MEC server as large square with load indicator
            load_pct = server.get_load_percentage()
            load_level = server.get_load_level()
            load_color = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red', 'CRITICAL': 'darkred'}[load_level]
            
            ax.scatter(server.x, server.y, c='darkblue', s=600, marker='s', 
                      edgecolors=load_color, linewidth=4, alpha=0.9, zorder=4)
            
            ax.text(server.x, server.y - 35, f'MEC-{server.id}', 
                   fontsize=11, ha='center', weight='bold', color='darkblue')
            ax.text(server.x, server.y + 35, f'{load_pct:.0f}%\n{load_level}', 
                   fontsize=9, ha='center', color=load_color, weight='bold')
        
        # 3. Draw IAR-to-MEC connections
        for iar in self.protocol.iar_servers.values():
            for mec_id in iar.connected_mec_servers:
                if mec_id in self.protocol.mec_servers:
                    mec = self.protocol.mec_servers[mec_id]
                    ax.plot([iar.x, mec.x], [iar.y, mec.y],
                           color='purple', linewidth=2, alpha=0.6, 
                           linestyle=':', zorder=2)
        
        # 4. Draw cluster boundaries
        for ch in cluster_heads:
            if ch.id in cluster_color_map:
                color = cluster_color_map[ch.id]
                boundary = plt.Circle((ch.x, ch.y), self.protocol.communication_range,
                                    fill=False, color=color, alpha=0.8, 
                                    linestyle='--', linewidth=2, zorder=2)
                ax.add_patch(boundary)
        
        # 5. Draw nodes
        for node in self.protocol.nodes.values():
            if not node.is_alive():
                continue
                
            if node.state == NodeState.CLUSTER_HEAD:
                color = cluster_color_map.get(node.id, 'red')
                ax.scatter(node.x, node.y, c=color, s=300, marker='^', 
                          edgecolors='black', linewidth=2, alpha=0.9, zorder=6)
                ax.text(node.x, node.y + 15, f'CH-{node.id}', 
                       fontsize=9, ha='center', weight='bold')
                ax.text(node.x, node.y - 15, f'{node.energy:.0f}J', 
                       fontsize=7, ha='center', color='darkred')
                
            elif node.state == NodeState.CLUSTER_MEMBER:
                ch_id = node.cluster_head_id
                color = cluster_color_map.get(ch_id, 'blue') if ch_id else 'blue'
                ax.scatter(node.x, node.y, c=color, s=150, marker='o', 
                          edgecolors='black', linewidth=1, alpha=0.8, zorder=5)
                ax.text(node.x, node.y + 10, f'{node.id}', 
                       fontsize=7, ha='center')
                
            else:  # IDLE
                ax.scatter(node.x, node.y, c='lightgray', s=100, marker='o', 
                          edgecolors='gray', linewidth=1, alpha=0.7, zorder=5)
                ax.text(node.x, node.y + 10, f'{node.id}', 
                       fontsize=7, ha='center', color='gray')
        
        # 6. Draw communication links
        for ch in cluster_heads:
            # CH to IAR links
            nearest_iar = self.protocol._find_nearest_iar_server(ch)
            if nearest_iar:
                ax.plot([ch.x, nearest_iar.x], [ch.y, nearest_iar.y], 
                       color='darkviolet', alpha=0.7, linewidth=2, linestyle='--')
            
            # Member to CH links
            for member_id in ch.cluster_members:
                if member_id in self.protocol.nodes:
                    member = self.protocol.nodes[member_id]
                    if member.is_alive():
                        distance = ch.distance_to(member)
                        link_color = 'green' if distance <= 50 else 'orange' if distance <= 80 else 'red'
                        ax.plot([ch.x, member.x], [ch.y, member.y], 
                               color=link_color, alpha=0.6, linewidth=1)
        
        # 7. Generate and draw animated packets
        self.generate_realistic_packet_animations()
        for packet in self.animated_packets:
            if packet.active:
                # Update packet position
                packet.update_position(0.1)
                
                # Draw packet
                circle = plt.Circle(packet.current_pos, packet.size/10, 
                                  color=packet.color, alpha=0.8, zorder=100)
                ax.add_patch(circle)
                
                # Add packet label
                ax.text(packet.current_pos[0], packet.current_pos[1] + 8, 
                       packet.packet_type, fontsize=6, ha='center', 
                       color=packet.color, weight='bold')
        
        # 8. Create enhanced legend
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=12, label='Cluster Head'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Cluster Member'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=8, label='Idle Node'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkblue', markersize=12, label='MEC Server'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='mediumpurple', markersize=10, label='IAR Server'),
            plt.Line2D([0], [0], color='purple', linestyle=':', linewidth=2, label='IAR-MEC Link'),
            plt.Line2D([0], [0], color='darkviolet', linestyle='--', linewidth=2, label='CH-IAR Link'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                 fancybox=True, shadow=True, framealpha=0.9)
        
        # 9. Add network statistics
        active_packets = len([p for p in self.animated_packets if p.active])
        total_mec_load = sum(s.get_load_percentage() for s in self.protocol.mec_servers.values()) / len(self.protocol.mec_servers)
        
        stats_text = f"Round {round_num} Status:\n"
        stats_text += f"Infrastructure: {len(self.protocol.iar_servers)} IAR, {len(self.protocol.mec_servers)} MEC\n"
        stats_text += f"Clusters: {len(cluster_heads)} | Members: {len(cluster_members)} | Idle: {len(idle_nodes)}\n"
        stats_text += f"Active Packets: {active_packets} | Avg MEC Load: {total_mec_load:.1f}%"
        
        if round_num % 5 == 0 and round_num > 0:
            stats_text += f"\n🔄 RE-CLUSTERING ACTIVE!"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor='yellow' if round_num % 5 == 0 and round_num > 0 else 'lightcyan', 
                alpha=0.9))
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        
        return fig, ax
    
    def simulate_round(self, round_num: int):
        """Simulate one round of the protocol"""
        print(f"\nSimulating round {round_num}...")
        
        # Update mobility
        area_bounds = (0, self.area_size, 0, self.area_size)
        for node in self.protocol.nodes.values():
            if node.is_alive():
                node.update_mobility(area_bounds)
        
        # Protocol operations
        self.protocol.current_time_slot = round_num
        
        # Enhanced re-clustering logic every 5 rounds for better visibility
        if round_num > 0 and round_num % 5 == 0:
            print(f"  🔄 PERFORMING RE-CLUSTERING at round {round_num}...")
            
            # Force aggressive re-clustering
            nodes_changed = self.protocol._check_and_recluster()
            
            # Also check for cluster head changes due to energy
            ch_changes = self.protocol._check_cluster_head_validity()
            
            if nodes_changed or ch_changes:
                print(f"  ✅ Re-clustering completed - network topology changed")
            else:
                print(f"  ℹ️  Re-clustering completed - no changes needed")
        
        # Generate communications
        self.protocol._generate_inter_cluster_traffic()
        self.protocol._generate_mec_tasks()
        self.protocol._process_inter_cluster_messages()
        self.protocol._process_mec_servers()
        
        # Regular cluster head validity check (not just during re-clustering)
        if round_num % 5 != 0:
            self.protocol._check_cluster_head_validity()
    
    def create_gif_animation(self, filename: str = "arpmec_crystal_clear.gif"):
        """Create a GIF animation with long pauses between frames"""
        print(f"\nCreating crystal clear GIF animation...")
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        def animate(frame_num):
            ax.clear()
            
            # Simulate the round
            if frame_num > 0:
                self.simulate_round(frame_num)
            
            # Create clean frame content
            ax.set_facecolor('#F8F9FA')
            ax.set_xlim(-50, self.area_size + 50)
            ax.set_ylim(-50, self.area_size + 50)
            
            # Get current network state
            cluster_heads = self.protocol._get_cluster_heads()
            cluster_members = [n for n in self.protocol.nodes.values() 
                              if n.state == NodeState.CLUSTER_MEMBER and n.is_alive()]
            idle_nodes = [n for n in self.protocol.nodes.values() 
                         if n.state == NodeState.IDLE and n.is_alive()]
            
            # Create cluster color mapping
            cluster_color_map = {}
            for i, ch in enumerate(cluster_heads):
                cluster_color_map[ch.id] = self.cluster_colors[i % len(self.cluster_colors)]
            
            # Draw MEC servers
            for server in self.protocol.mec_servers.values():
                coverage = plt.Circle((server.x, server.y), 50, 
                                    fill=True, color='purple', alpha=0.2, zorder=1)
                ax.add_patch(coverage)
                
                ax.scatter(server.x, server.y, c='purple', s=600, marker='s', 
                          edgecolors='black', linewidth=3, alpha=0.9, zorder=3)
                ax.annotate(f'MEC-{server.id}', (server.x, server.y), 
                           xytext=(0, -40), textcoords='offset points',
                           fontsize=14, fontweight='bold', ha='center', color='purple')
            
            # Draw cluster areas
            for ch in cluster_heads:
                if ch.id in cluster_color_map:
                    color = cluster_color_map[ch.id]
                    
                    cluster_area = plt.Circle((ch.x, ch.y), 10,
                                            fill=True, color=color, alpha=0.25, zorder=2)
                    ax.add_patch(cluster_area)
                    
                    boundary = plt.Circle((ch.x, ch.y), self.protocol.communication_range,
                                        fill=False, color=color, alpha=0.8, 
                                        linestyle='--', linewidth=2, zorder=2)
                    ax.add_patch(boundary)
            
            # Draw connections
            for ch in cluster_heads:
                if ch.id in cluster_color_map:
                    color = cluster_color_map[ch.id]
                    
                    # CH to MEC connection
                    nearest_mec = self.protocol._find_nearest_mec_server(ch)
                    if nearest_mec:
                        ax.plot([ch.x, nearest_mec.x], [ch.y, nearest_mec.y],
                               color=color, linewidth=4, alpha=0.7, zorder=4)
                    
                    # Members to CH connections
                    for member in cluster_members:
                        if member.cluster_head_id == ch.id:
                            ax.plot([member.x, ch.x], [member.y, ch.y],
                                   color=color, linewidth=2, alpha=0.6, zorder=4)
            
            # Draw nodes
            for ch in cluster_heads:
                if ch.id in cluster_color_map:
                    color = cluster_color_map[ch.id]
                    ax.scatter(ch.x, ch.y, c=color, s=400, marker='^',
                              edgecolors='black', linewidth=3, zorder=6)
                    ax.annotate(f'CH-{ch.id}', (ch.x, ch.y), xytext=(8, 8),
                               textcoords='offset points', fontsize=12, fontweight='bold')
            
            for member in cluster_members:
                if member.cluster_head_id and member.cluster_head_id in cluster_color_map:
                    color = cluster_color_map[member.cluster_head_id]
                    ax.scatter(member.x, member.y, c=color, s=200, marker='o',
                              alpha=0.9, edgecolors='black', linewidth=2, zorder=7)
                    # Add member labels for debugging
                    ax.annotate(f'M{member.id}', (member.x, member.y), xytext=(-5, -15),
                               textcoords='offset points', fontsize=10, fontweight='bold')
            
            if idle_nodes:
                for node in idle_nodes:
                    ax.scatter(node.x, node.y, c='lightgray', s=80, marker='s',
                              alpha=0.7, edgecolors='black', linewidth=1, zorder=5)
            
            # Title and stats with better re-clustering indication
            title_suffix = ""
            if frame_num == 0:
                title_suffix = "(Initial Network)"
            elif frame_num % 5 == 0 and frame_num > 0:
                title_suffix = "🔄 (RE-CLUSTERING EVENT!)"
            
            main_title = f"ARPMEC Protocol - Round {frame_num} {title_suffix}"
            ax.set_title(main_title, fontsize=16, fontweight='bold', pad=20)
            
            # Enhanced stats with re-clustering info
            next_recluster = ((frame_num // 5) + 1) * 5
            stats_text = f"Clusters: {len(cluster_heads)} | Members: {len(cluster_members)} | Idle: {len(idle_nodes)}"
            if frame_num % 5 == 0 and frame_num > 0:
                stats_text += f"\n🔄 RE-CLUSTERING ACTIVE!"
            else:
                stats_text += f"\nNext re-clustering: Round {next_recluster}"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor='yellow' if frame_num % 5 == 0 and frame_num > 0 else 'white', 
                    alpha=0.9))
            
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X Position (meters)', fontsize=12)
            ax.set_ylabel('Y Position (meters)', fontsize=12)
            
            return ax.collections + ax.patches + ax.texts
        
        # Create animation with LONG intervals
        anim = animation.FuncAnimation(fig, animate, frames=self.max_rounds,
                                     interval=5000, repeat=True, blit=False)
        
        # Save with slow FPS for long viewing time
        anim.save(filename, writer='pillow', fps=0.2)  # 5 seconds per frame
        plt.close(fig)
        
        print(f"✅ GIF saved as {filename}")
        return filename

def create_crystal_clear_gif_demo():
    """Create and run the crystal clear GIF demo"""
    print("🎬 ARPMEC Crystal Clear GIF Demo")
    print("=" * 50)
    
    # Create demo instance
    demo = PacketTracerStyleARPMECDemo(N=20, area_size=800)
    
    # Set up network
    clusters = demo.setup_network()
    
    if len(clusters) == 0:
        print("❌ No clusters formed! Check network parameters.")
        return None
    
    # Create the GIF with long pauses
    gif_filename = demo.create_gif_animation("arpmec_crystal_clear.gif")
    
    print(f"\n✅ Crystal Clear Demo Complete!")
    print(f"📁 GIF saved as: {gif_filename}")
    print(f"⏱️  Each frame shows for 5 seconds")
    print(f"🎯 Focus: Clean cluster visualization and MEC communication")
    
    return demo.protocol

def run_crystal_clear_demo():
    """Run the crystal clear demo"""
    print("ARPMEC Crystal Clear Demo Starting...")
    print("=" * 50)
    
    # Create and run the GIF demo
    protocol = create_crystal_clear_gif_demo()
    
    # Also create a final static visualization
    if protocol:
        print("\nCreating final static visualization...")
        try:
            demo = PacketTracerStyleARPMECDemo()
            demo.protocol = protocol
            demo.max_rounds = 40  # Set max_rounds attribute
            fig = demo.create_clean_frame(39, "(Final State)")
            plt.savefig('arpmec_final_state.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("✅ Static visualization saved as 'arpmec_final_state.png'")
        except Exception as e:
            print(f"⚠️  Static visualization error: {e}")
    
    print("\n✅ Demo complete!")
    print("Check the GIF file for the animated visualization.")

if __name__ == "__main__":
    run_crystal_clear_demo()
