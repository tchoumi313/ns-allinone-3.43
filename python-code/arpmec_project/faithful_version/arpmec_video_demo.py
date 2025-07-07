#!/usr/bin/env python3
"""
ARPMEC Video Demo - Creates MP4 animation with packet movement like Packet Tracer

This script creates a real video file showing:
- IAR infrastructure with CH → IAR → MEC communication
- Animated packets moving along routes
- Re-clustering events
- MEC server load visualization
- Cluster formation and mobility
"""

import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
# Import the ENHANCED implementation with IAR
from arpmec_faithful import ARPMECProtocol, IARServer, Node
from matplotlib.patches import Circle, FancyBboxPatch


def create_realistic_network(num_nodes: int = 20, area_size: float = 800.0) -> List[Node]:
    """Create a realistic network that forms visible clusters"""
    nodes = []
    
    # Create 4 initial cluster areas for better visualization
    cluster_centers = [
        (0.25, 0.25),  # Bottom-left
        (0.75, 0.25),  # Bottom-right  
        (0.25, 0.75),  # Top-left
        (0.75, 0.75),  # Top-right
    ]
    
    for i in range(num_nodes):
        # Assign nodes to cluster areas
        cluster_idx = i % 4
        center_x, center_y = cluster_centers[cluster_idx]
        
        # Add some randomness around cluster center
        x = center_x * area_size + random.uniform(-60, 60)
        y = center_y * area_size + random.uniform(-60, 60)
        
        # Keep within bounds
        x = max(50, min(area_size - 50, x))
        y = max(50, min(area_size - 50, y))
        
        energy = random.uniform(90, 110)
        nodes.append(Node(i, x, y, energy))
    
    return nodes

class PacketAnimation:
    """Animated packet for visualization"""
    def __init__(self, start_pos: Tuple[float, float], end_pos: Tuple[float, float], 
                 packet_type: str, duration: int = 30):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.packet_type = packet_type
        self.duration = duration
        self.current_frame = 0
        self.active = True
        
        # Packet visual properties
        self.colors = {
            'data': 'gold',
            'mec_task': 'orange', 
            'inter_cluster': 'blueviolet',
            'control': 'cyan'
        }
        self.size = 100
        
    def get_current_position(self) -> Tuple[float, float]:
        """Get current packet position based on animation frame"""
        if not self.active or self.current_frame >= self.duration:
            return self.end_pos
            
        # Linear interpolation between start and end
        progress = self.current_frame / self.duration
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * progress
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * progress
        
        return (x, y)
    
    def update(self):
        """Update packet animation"""
        self.current_frame += 1
        if self.current_frame >= self.duration:
            self.active = False

class ARPMECVideoDemo:
    """ARPMEC Video Demo with Packet Tracer style animation"""
    
    def __init__(self, protocol: ARPMECProtocol, output_file: str = "arpmec_animation.mp4"):
        self.protocol = protocol
        self.output_file = output_file
        
        # Animation settings
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        self.frames = []
        self.packets = []  # Active packet animations
        
        # Colors for visualization
        self.colors = {
            'cluster_head': 'red',
            'cluster_member': 'blue', 
            'mec_server': 'darkred',
            'iar_server': 'indigo',
            'idle_node': 'lightgray',
            'cluster_area': 'lightblue',
            'mec_coverage': 'lightcoral',
            'iar_coverage': 'lavender'
        }
        
        # Network state
        self.current_round = 0
        self.cluster_colors = {}
        self.setup_network()
        
    def setup_network(self):
        """Setup initial network clustering"""
        print("Setting up network for video demo...")
        clusters = self.protocol.clustering_algorithm()
        
        # Assign colors to clusters
        cluster_color_palette = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 
                               'lightcyan', 'lightgray', 'wheat', 'lavender', 'mistyrose']
        
        for i, (ch_id, members) in enumerate(clusters.items()):
            color = cluster_color_palette[i % len(cluster_color_palette)]
            self.cluster_colors[ch_id] = color
            
        print(f"✅ Network setup complete: {len(clusters)} clusters formed")
        return clusters
    
    def add_packet_animation(self, start_node_id: int, end_node_id: int, packet_type: str, 
                           via_iar: Optional[int] = None, via_mec: Optional[int] = None):
        """Add animated packet to visualization"""
        start_node = self.protocol.nodes[start_node_id]
        
        if via_iar and via_mec:
            # Multi-hop: Node → IAR → MEC → IAR → Node
            iar_server = self.protocol.iar_servers[via_iar]
            mec_server = self.protocol.mec_servers[via_mec]
            end_node = self.protocol.nodes[end_node_id]
            
            # Create packet segments
            self.packets.append(PacketAnimation(
                (start_node.x, start_node.y), (iar_server.x, iar_server.y), 
                packet_type, duration=20))
            self.packets.append(PacketAnimation(
                (iar_server.x, iar_server.y), (mec_server.x, mec_server.y), 
                packet_type, duration=20))
            self.packets.append(PacketAnimation(
                (mec_server.x, mec_server.y), (end_node.x, end_node.y), 
                packet_type, duration=20))
        else:
            # Simple: Node → Node
            end_node = self.protocol.nodes[end_node_id]
            self.packets.append(PacketAnimation(
                (start_node.x, start_node.y), (end_node.x, end_node.y), 
                packet_type, duration=25))
    
    def draw_frame(self, round_num: int) -> plt.Figure:
        """Draw a single frame of the animation"""
        self.ax.clear()
        self.ax.set_xlim(0, 1000)
        self.ax.set_ylim(0, 1000)
        self.ax.set_aspect('equal')
        
        # 1. Draw cluster areas (background)
        cluster_heads = [n for n in self.protocol.nodes.values() 
                        if n.state.value == 'cluster_head']
        
        for ch in cluster_heads:
            if ch.cluster_id in self.cluster_colors:
                color = self.cluster_colors[ch.cluster_id]
                circle = Circle((ch.x, ch.y), self.protocol.communication_range,
                              facecolor=color, alpha=0.2, edgecolor='gray', 
                              linestyle='--', linewidth=1, zorder=1)
                self.ax.add_patch(circle)
        
        # 2. Draw IAR coverage areas
        for iar in self.protocol.iar_servers.values():
            circle = Circle((iar.x, iar.y), iar.coverage_radius,
                          facecolor=self.colors['iar_coverage'], alpha=0.15,
                          edgecolor='purple', linestyle=':', linewidth=2, zorder=1)
            self.ax.add_patch(circle)
        
        # 3. Draw IAR-MEC backbone connections
        for iar in self.protocol.iar_servers.values():
            for mec_id in iar.connected_mec_servers:
                if mec_id in self.protocol.mec_servers:
                    mec = self.protocol.mec_servers[mec_id]
                    self.ax.plot([iar.x, mec.x], [iar.y, mec.y], 
                               'purple', linewidth=3, linestyle=':', alpha=0.6, zorder=2)
        
        # 4. Draw cluster member connections
        for ch in cluster_heads:
            for member_id in ch.cluster_members:
                if member_id in self.protocol.nodes:
                    member = self.protocol.nodes[member_id]
                    self.ax.plot([ch.x, member.x], [ch.y, member.y], 
                               'gray', linewidth=1, alpha=0.4, zorder=2)
        
        # 5. Draw CH-to-IAR connections
        for ch in cluster_heads:
            nearest_iar = self.protocol._find_nearest_iar_server(ch)
            if nearest_iar:
                self.ax.plot([ch.x, nearest_iar.x], [ch.y, nearest_iar.y],
                           'green', linewidth=2, alpha=0.7, zorder=3)
        
        # 6. Draw nodes
        for node in self.protocol.nodes.values():
            if not node.is_alive():
                continue
                
            if node.state.value == 'cluster_head':
                # Cluster heads - red triangles
                self.ax.scatter(node.x, node.y, c=self.colors['cluster_head'], 
                              s=200, marker='^', edgecolors='black', linewidth=2, zorder=6)
                # Add CH label
                self.ax.annotate(f'CH-{node.id}', (node.x+10, node.y+10), 
                               fontsize=10, fontweight='bold', color='darkred')
            elif node.state.value == 'cluster_member':
                # Cluster members - blue circles
                self.ax.scatter(node.x, node.y, c=self.colors['cluster_member'], 
                              s=100, marker='o', edgecolors='black', linewidth=1, zorder=5)
            else:
                # Idle nodes - gray squares
                self.ax.scatter(node.x, node.y, c=self.colors['idle_node'], 
                              s=80, marker='s', edgecolors='black', linewidth=1, zorder=5)
        
        # 7. Draw MEC servers
        for mec in self.protocol.mec_servers.values():
            # MEC server as dark red rectangle
            rect = FancyBboxPatch((mec.x-20, mec.y-15), 40, 30,
                                boxstyle="round,pad=3", 
                                facecolor=self.colors['mec_server'], 
                                edgecolor='black', linewidth=2, alpha=0.9, zorder=7)
            self.ax.add_patch(rect)
            
            # MEC label and load
            load_pct = mec.get_load_percentage()
            load_level = mec.get_load_level()
            self.ax.annotate(f'MEC-{mec.id}\n{load_pct:.0f}%\n{load_level}', 
                           (mec.x, mec.y-35), ha='center', fontsize=9, 
                           fontweight='bold', color='white',
                           bbox=dict(boxstyle="round,pad=2", facecolor='darkred', alpha=0.8))
        
        # 8. Draw IAR servers
        for iar in self.protocol.iar_servers.values():
            # IAR server as purple diamond
            diamond = patches.RegularPolygon((iar.x, iar.y), 4, 25,
                                           orientation=math.pi/4,
                                           facecolor=self.colors['iar_server'],
                                           edgecolor='black', linewidth=2, alpha=0.9, zorder=6)
            self.ax.add_patch(diamond)
            
            # IAR label
            self.ax.annotate(f'IAR-{iar.id}', (iar.x, iar.y-40), ha='center', 
                           fontsize=10, fontweight='bold', color='indigo')
        
        # 9. Draw animated packets
        for packet in self.packets[:]:  # Copy to avoid modification during iteration
            if packet.active:
                x, y = packet.get_current_position()
                color = packet.colors.get(packet.packet_type, 'yellow')
                
                # Draw packet with trail effect
                self.ax.scatter(x, y, c=color, s=packet.size, marker='o', 
                              edgecolors='black', linewidth=2, alpha=0.8, zorder=10)
                
                # Add packet type label
                self.ax.annotate(packet.packet_type.upper(), (x+15, y+15), 
                               fontsize=8, fontweight='bold', color=color,
                               bbox=dict(boxstyle="round,pad=1", facecolor='white', alpha=0.7))
                
                packet.update()
            else:
                self.packets.remove(packet)
        
        # 10. Enhanced legend
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                      markersize=15, label='Cluster Heads'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=12, label='Cluster Members'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkred', 
                      markersize=15, label='MEC Servers'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='indigo', 
                      markersize=12, label='IAR Servers'),
            plt.Line2D([0], [0], color='purple', linewidth=3, linestyle=':', 
                      label='IAR-MEC Backbone'),
            plt.Line2D([0], [0], color='green', linewidth=2, 
                      label='CH-IAR Links'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
                      markersize=10, label='Data Packets'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=10, label='MEC Tasks')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # 11. Title and info
        if round_num % 5 == 0 and round_num > 0:
            title_suffix = " 🔄 RE-CLUSTERING EVENT!"
        else:
            title_suffix = f" (Next re-clustering: Round {((round_num // 5) + 1) * 5})"
            
        main_title = f"ARPMEC with IAR Infrastructure - Round {round_num}{title_suffix}"
        subtitle = "CH → IAR → MEC Communication | Adaptive Routing | Packet Animation"
        self.ax.set_title(f"{main_title}\n{subtitle}", fontsize=14, fontweight='bold', pad=15)
        
        # 12. Network statistics
        alive_nodes = sum(1 for n in self.protocol.nodes.values() if n.is_alive())
        active_clusters = len([n for n in self.protocol.nodes.values() 
                             if n.state.value == 'cluster_head'])
        avg_mec_load = np.mean([mec.get_load_percentage() 
                               for mec in self.protocol.mec_servers.values()])
        
        stats_text = (f"Alive Nodes: {alive_nodes}/{len(self.protocol.nodes)} | "
                     f"Active Clusters: {active_clusters} | "
                     f"Avg MEC Load: {avg_mec_load:.1f}% | "
                     f"Active Packets: {len(self.packets)}")
        
        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes, 
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=5", facecolor='lightblue', alpha=0.8))
        
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position (meters)', fontsize=12)
        self.ax.set_ylabel('Y Position (meters)', fontsize=12)
        
        return self.fig
    
    def simulate_round(self, round_num: int):
        """Simulate one round and generate packet animations"""
        print(f"Simulating round {round_num}...")
        
        # Update node mobility
        area_bounds = (0, 1000, 0, 1000)
        for node in self.protocol.nodes.values():
            if node.is_alive():
                node.update_mobility(area_bounds)
        
        # Check for re-clustering
        if round_num % 5 == 0 and round_num > 0:
            print(f"  🔄 Re-clustering at round {round_num}")
            self.protocol._check_and_recluster()
            self.protocol._check_cluster_head_validity()
        
        # Generate inter-cluster traffic with animations
        cluster_heads = [n for n in self.protocol.nodes.values() 
                        if n.state.value == 'cluster_head']
        
        # Simulate some inter-cluster communication
        if len(cluster_heads) >= 2 and random.random() < 0.4:
            source_ch = random.choice(cluster_heads)
            target_ch = random.choice([ch for ch in cluster_heads if ch.id != source_ch.id])
            
            # Find IAR and MEC for animation
            source_iar = self.protocol._find_nearest_iar_server(source_ch)
            if source_iar and source_iar.connected_mec_servers:
                mec_id = source_iar.connected_mec_servers[0]
                
                print(f"  📡 Inter-cluster: CH-{source_ch.id} → CH-{target_ch.id} via IAR-{source_iar.id}")
                
                # Add animated packet
                self.add_packet_animation(source_ch.id, target_ch.id, 'inter_cluster',
                                        via_iar=source_iar.id, via_mec=mec_id)
        
        # Simulate MEC task submissions
        for ch in cluster_heads:
            if random.random() < 0.3:  # 30% chance
                nearest_iar = self.protocol._find_nearest_iar_server(ch)
                if nearest_iar and nearest_iar.connected_mec_servers:
                    mec_id = nearest_iar.connected_mec_servers[0]
                    
                    print(f"  💼 MEC Task: CH-{ch.id} → MEC-{mec_id} via IAR-{nearest_iar.id}")
                    
                    # Add animated packet for MEC task
                    self.add_packet_animation(ch.id, ch.id, 'mec_task',
                                            via_iar=nearest_iar.id, via_mec=mec_id)
        
        # Simulate regular data transmission within clusters
        for ch in cluster_heads:
            for member_id in ch.cluster_members:
                if random.random() < 0.2:  # 20% chance
                    print(f"  📊 Data: Node-{member_id} → CH-{ch.id}")
                    self.add_packet_animation(member_id, ch.id, 'data')
        
        # Process MEC servers
        self.protocol._process_mec_servers()
    
    def create_video(self, max_rounds: int = 30, fps: int = 2):
        """Create the complete video animation"""
        print(f"🎬 Creating ARPMEC video animation...")
        print(f"   Rounds: {max_rounds}, FPS: {fps}")
        print(f"   Output: {self.output_file}")
        
        frames = []
        
        # Generate frames
        for round_num in range(max_rounds + 1):
            if round_num > 0:
                self.simulate_round(round_num)
            
            # Create frame (repeat each frame multiple times for slower animation)
            for _ in range(3):  # Each round shown for 3 frames
                frame_fig = self.draw_frame(round_num)
                frames.append(frame_fig)
        
        print(f"✅ Generated {len(frames)} frames")
        
        # Create animation
        def animate(frame_idx):
            return frames[frame_idx]
        
        print("🎞️  Rendering video...")
        
        # Create animation object
        anim = animation.FuncAnimation(self.fig, animate, frames=len(frames), 
                                     interval=500, blit=False, repeat=True)
        
        # Save as MP4
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='ARPMEC'), bitrate=1800)
            anim.save(self.output_file, writer=writer)
            print(f"✅ Video saved as: {self.output_file}")
            
        except Exception as e:
            print(f"❌ Error saving MP4: {e}")
            print("💡 Trying to save as GIF instead...")
            
            # Fallback to GIF
            gif_file = self.output_file.replace('.mp4', '.gif')
            anim.save(gif_file, writer='pillow', fps=fps)
            print(f"✅ GIF saved as: {gif_file}")
        
        plt.close()
        
        return anim

def main():
    """Main function to run the video demo"""
    print("🚀 ARPMEC Video Demo - Packet Tracer Style Animation")
    print("=" * 60)
    
    # Create network
    nodes = create_realistic_network(20, 800)
    protocol = ARPMECProtocol(nodes, C=4, R=10, K=3)
    
    # Create video demo
    demo = ARPMECVideoDemo(protocol, "arpmec_packet_animation.mp4")
    
    # Generate video
    animation_obj = demo.create_video(max_rounds=20, fps=1)
    
    print("\n✅ Video Demo Complete!")
    print("📁 Check the generated video file for packet animations")
    print("🎯 Features: IAR infrastructure, packet movement, re-clustering, MEC loads")

if __name__ == "__main__":
    main()
