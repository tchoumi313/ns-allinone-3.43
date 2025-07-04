#!/usr/bin/env python3
"""
CRYSTAL CLEAR ARPMEC Visualization with GIF Animation

This creates a clean, understandable visualization of the ARPMEC protocol
with long pauses between rounds for easy observation.
"""

import math
import os
import random
import time
from typing import Dict, List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
# Import the FIXED implementation
from arpmec_faithful import ARPMECProtocol, MECServer, Node
from matplotlib.patches import Circle, ConnectionPatch, FancyBboxPatch


def create_realistic_network(num_nodes: int, area_size: float = 800.0) -> List[Node]:
    """Create a realistic network that can actually form clusters"""
    nodes = []
    
    # Create nodes in a more clustered pattern to ensure connectivity
    for i in range(num_nodes):
        # Create 3-4 distinct clusters initially
        cluster_center_x = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
        cluster_idx = i % 4
        center_x, center_y = cluster_center_x[cluster_idx]
        
        x = center_x * area_size + random.uniform(-80, 80)
        y = center_y * area_size + random.uniform(-80, 80)
        
        # Keep within bounds
        x = max(50, min(area_size - 50, x))
        y = max(50, min(area_size - 50, y))
        
        energy = random.uniform(90, 110)  # Realistic initial energy
        nodes.append(Node(i, x, y, energy))
    return nodes

class CrystalClearARPMECVisualizer:
    """Crystal clear ARPMEC visualizer with clean, understandable graphics"""
    
    def __init__(self, protocol: ARPMECProtocol):
        self.protocol = protocol
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.round_num = 0
        self.communication_log = []
        self.max_log_entries = 8
        
        # Clean color scheme
        self.colors = {
            'cluster_head': '#FF4444',
            'cluster_member': '#4444FF',
            'idle': '#CCCCCC',
            'mec_server': '#FF8800',
            'dead': '#888888'
        }
        
        # Communication colors
        self.comm_colors = {
            'member_to_ch': '#00AA00',
            'ch_to_mec': '#AA00AA',
            'mec_to_mec': '#FFAA00',
            'ch_to_ch_via_mec': '#AA0000'
        }
        
    def clear_and_setup_plot(self):
        """Clear and set up the plot for a new frame"""
        self.ax.clear()
        self.ax.set_xlim(0, 800)
        self.ax.set_ylim(0, 800)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#F8F8F8')
        
    def draw_clean_network(self, round_num: int):
        """Draw a clean, understandable network visualization"""
        self.clear_and_setup_plot()
        
        # Title
        self.ax.set_title(f'ARPMEC Protocol - Round {round_num}\n'
                         f'Crystal Clear Visualization', 
                         fontsize=16, fontweight='bold', pad=20)
        
        # 1. Draw cluster boundaries first (subtle circles)
        self.draw_cluster_boundaries()
        
        # 2. Draw communication lines (before nodes so they're behind)
        self.draw_communication_lines()
        
        # 3. Draw MEC servers (large, visible)
        self.draw_mec_servers()
        
        # 4. Draw nodes (on top of everything)
        self.draw_nodes()
        
        # 5. Add clean legend
        self.add_clean_legend()
        
        # 6. Add network statistics
        self.add_network_stats()
        
        # 7. Add communication log
        self.add_communication_log()
        
    def draw_cluster_boundaries(self):
        """Draw subtle cluster boundary circles"""
        cluster_heads = self.protocol._get_cluster_heads()
        
        # Cluster colors
        cluster_colors = ['#FF9999', '#9999FF', '#99FF99', '#FFFF99', '#FF99FF']
        
        for i, ch in enumerate(cluster_heads):
            color = cluster_colors[i % len(cluster_colors)]
            
            # Draw communication range circle
            circle = Circle((ch.x, ch.y), self.protocol.communication_range, 
                          fill=True, facecolor=color, alpha=0.3, 
                          edgecolor='black', linewidth=1, linestyle='--')
            self.ax.add_patch(circle)
            
            # Add cluster label
            self.ax.text(ch.x, ch.y + self.protocol.communication_range + 15,
                        f'Cluster {ch.id}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    def draw_communication_lines(self):
        """Draw clear communication lines"""
        # Only draw recent communications to avoid clutter
        recent_comms = self.communication_log[-5:] if self.communication_log else []
        
        for comm in recent_comms:
            if comm['type'] == 'member_to_ch':
                self.draw_arrow(comm['from'], comm['to'], 
                               self.comm_colors['member_to_ch'], 'Member→CH')
            elif comm['type'] == 'ch_to_mec':
                self.draw_arrow(comm['from'], comm['to'], 
                               self.comm_colors['ch_to_mec'], 'CH→MEC')
            elif comm['type'] == 'ch_to_ch_via_mec':
                self.draw_curved_arrow(comm['from'], comm['to'], comm['via'],
                                     self.comm_colors['ch_to_ch_via_mec'], 'CH→CH via MEC')
    
    def draw_arrow(self, from_pos, to_pos, color, label, linewidth=2):
        """Draw a simple arrow between two points"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        self.ax.annotate('', xy=to_pos, xytext=from_pos,
                        arrowprops=dict(arrowstyle='->', color=color, 
                                      lw=linewidth, alpha=0.8))
    
    def draw_curved_arrow(self, from_pos, to_pos, via_pos, color, label, linewidth=2):
        """Draw a curved arrow via an intermediate point (MEC server)"""
        # Draw two arrows: from->via and via->to
        self.draw_arrow(from_pos, via_pos, color, label, linewidth)
        self.draw_arrow(via_pos, to_pos, color, label, linewidth)
        
        # Add a small circle at the via point to show the relay
        circle = Circle(via_pos, 8, fill=True, facecolor=color, alpha=0.5)
        self.ax.add_patch(circle)
    
    def draw_mec_servers(self):
        """Draw MEC servers as large, distinctive squares"""
        for server in self.protocol.mec_servers.values():
            # Draw server as a large square
            square = FancyBboxPatch((server.x - 25, server.y - 25), 50, 50,
                                   boxstyle="round,pad=3", 
                                   facecolor=self.colors['mec_server'],
                                   edgecolor='black', linewidth=2)
            self.ax.add_patch(square)
            
            # Add server label
            self.ax.text(server.x, server.y, f'MEC\n{server.id}', 
                        ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Add load indicator
            load_text = f'{server.get_load_percentage():.0f}%'
            self.ax.text(server.x, server.y - 35, load_text,
                        ha='center', va='top', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    def draw_nodes(self):
        """Draw nodes with clear, distinctive symbols"""
        for node in self.protocol.nodes.values():
            if not node.is_alive():
                # Dead node
                self.ax.scatter(node.x, node.y, c=self.colors['dead'], s=50, 
                              marker='x', alpha=0.7, linewidths=2)
                continue
                
            # Choose node appearance based on state
            if node.state.value == 'cluster_head':
                self.ax.scatter(node.x, node.y, c=self.colors['cluster_head'], 
                              s=200, marker='^', edgecolors='black', linewidths=2)
                self.ax.text(node.x, node.y-15, f'CH{node.id}', ha='center', va='top',
                            fontsize=8, fontweight='bold')
                
            elif node.state.value == 'cluster_member':
                self.ax.scatter(node.x, node.y, c=self.colors['cluster_member'], 
                              s=100, marker='o', edgecolors='black', linewidths=1)
                self.ax.text(node.x, node.y-10, f'{node.id}', ha='center', va='top',
                            fontsize=6)
                
            else:
                # Idle node
                self.ax.scatter(node.x, node.y, c=self.colors['idle'], 
                              s=80, marker='s', edgecolors='black', linewidths=1)
    
    def add_clean_legend(self):
        """Add a clean, readable legend"""
        legend_elements = []
        
        # Node types
        legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                        markerfacecolor=self.colors['cluster_head'],
                                        markersize=12, label='Cluster Head'))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=self.colors['cluster_member'],
                                        markersize=10, label='Cluster Member'))
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor=self.colors['mec_server'],
                                        markersize=12, label='MEC Server'))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=self.colors['idle'],
                                        markersize=8, label='Idle Node'))
        
        # Communication types
        legend_elements.append(plt.Line2D([0], [0], color=self.comm_colors['member_to_ch'],
                                        linewidth=3, label='Member→CH'))
        legend_elements.append(plt.Line2D([0], [0], color=self.comm_colors['ch_to_mec'],
                                        linewidth=3, label='CH→MEC'))
        legend_elements.append(plt.Line2D([0], [0], color=self.comm_colors['ch_to_ch_via_mec'],
                                        linewidth=3, label='CH→CH via MEC'))
        
        # Place legend outside the plot
        self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    def add_network_stats(self):
        """Add network statistics in a clean box"""
        alive_nodes = len([n for n in self.protocol.nodes.values() if n.is_alive()])
        cluster_heads = len(self.protocol._get_cluster_heads())
        total_energy = sum(n.initial_energy - n.energy for n in self.protocol.nodes.values())
        
        stats_text = (f"Network Status:\n"
                     f"• Alive Nodes: {alive_nodes}/{len(self.protocol.nodes)}\n"
                     f"• Active Clusters: {cluster_heads}\n"
                     f"• Energy Used: {total_energy:.1f}J\n"
                     f"• MEC Servers: {len(self.protocol.mec_servers)}")
        
        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    def add_communication_log(self):
        """Add recent communication log"""
        if not self.communication_log:
            return
            
        recent_comms = self.communication_log[-self.max_log_entries:]
        
        log_text = "Recent Communications:\n"
        for i, comm in enumerate(recent_comms):
            log_text += f"• {comm['description']}\n"
        
        self.ax.text(0.02, 0.5, log_text, transform=self.ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
    
    def log_communication(self, comm_type: str, from_node: Node, to_node: Node, 
                         via_node: Node = None, description: str = ""):
        """Log a communication event"""
        comm_event = {
            'type': comm_type,
            'from': (from_node.x, from_node.y),
            'to': (to_node.x, to_node.y),
            'via': (via_node.x, via_node.y) if via_node else None,
            'description': description,
            'round': self.round_num
        }
        
        self.communication_log.append(comm_event)
        
        # Keep only recent communications
        if len(self.communication_log) > 20:
            self.communication_log = self.communication_log[-20:]

def create_crystal_clear_gif_demo():
    """Create a crystal clear GIF animation of ARPMEC with long pauses"""
    print("Creating Crystal Clear ARPMEC GIF Demo...")
    print("=" * 60)
    
    # Create network
    nodes = create_realistic_network(20, 800)  # 20 nodes in 800x800 area
    protocol = ARPMECProtocol(nodes, C=4, R=10, K=3)
    
    print(f"Network created: {len(nodes)} nodes, {len(protocol.mec_servers)} MEC servers")
    
    # Run initial clustering
    print("Running initial clustering...")
    clusters = protocol.clustering_algorithm()
    print(f"Initial clusters: {len(clusters)}")
    
    # Create visualizer
    visualizer = CrystalClearARPMECVisualizer(protocol)
    
    # Store frames for GIF
    frames = []
    
    # Initial state (long pause)
    print("Capturing initial state...")
    for pause_frame in range(15):  # 15 frames = 3 seconds at 5fps
        visualizer.draw_clean_network(0)
        visualizer.ax.text(400, 750, f"Initial Network State", 
                          ha='center', va='center', fontsize=14, fontweight='bold',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))
        
        # Save frame
        frame_path = f'/tmp/arpmec_frame_{len(frames):03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        frames.append(frame_path)
        
        if pause_frame == 0:
            print(f"  Frame {len(frames)}: Initial state")
    
    # Run simulation with long pauses between rounds
    for round_num in range(1, 8):  # Only 7 rounds for manageable GIF
        print(f"\nRound {round_num}...")
        
        # Simulate one round
        protocol.current_time_slot = round_num
        
        # Update mobility
        for node in protocol.nodes.values():
            if node.is_alive():
                node.update_mobility((0, 800, 0, 800))
        
        # Generate communications
        protocol._generate_inter_cluster_traffic()
        protocol._generate_mec_tasks()
        
        # Log some communications for visualization
        cluster_heads = protocol._get_cluster_heads()
        if cluster_heads:
            ch = cluster_heads[0]
            nearest_mec = protocol._find_nearest_mec_server(ch)
            if nearest_mec:
                visualizer.log_communication('ch_to_mec', ch, 
                                           Node(-1, nearest_mec.x, nearest_mec.y, 100),
                                           description=f"CH-{ch.id} → MEC-{nearest_mec.id}")
        
        # Re-clustering check
        if round_num % 3 == 0:
            protocol._check_and_recluster()
            visualizer.communication_log.append({
                'type': 'system',
                'description': f"🔄 Re-clustering check at round {round_num}",
                'round': round_num
            })
        
        # Draw frame with round transition
        visualizer.round_num = round_num
        visualizer.draw_clean_network(round_num)
        
        # Add round transition text
        visualizer.ax.text(400, 750, f"Round {round_num} - Network Operation", 
                          ha='center', va='center', fontsize=14, fontweight='bold',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8))
        
        # Save multiple frames for this round (long pause)
        frames_per_round = 12  # 12 frames = 2.4 seconds at 5fps
        for pause_frame in range(frames_per_round):
            frame_path = f'/tmp/arpmec_frame_{len(frames):03d}.png'
            plt.savefig(frame_path, dpi=100, bbox_inches='tight')
            frames.append(frame_path)
            
            if pause_frame == 0:
                print(f"  Frame {len(frames)}: Round {round_num} state")
    
    # Final state (extra long pause)
    print("\nCapturing final state...")
    for pause_frame in range(20):  # 20 frames = 4 seconds at 5fps
        visualizer.draw_clean_network(7)
        visualizer.ax.text(400, 750, f"Final Network State", 
                          ha='center', va='center', fontsize=14, fontweight='bold',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        
        frame_path = f'/tmp/arpmec_frame_{len(frames):03d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        frames.append(frame_path)
        
        if pause_frame == 0:
            print(f"  Frame {len(frames)}: Final state")
    
    plt.close()
    
    # Create GIF
    print(f"\nCreating GIF from {len(frames)} frames...")
    gif_path = '/home/donsoft/ns-allinone-3.43/python-code/arpmec_project/faithful_version/arpmec_crystal_clear.gif'
    
    # Use matplotlib animation to create GIF
    fig, ax = plt.subplots(figsize=(14, 10))
    
    def animate_frame(frame_num):
        if frame_num < len(frames):
            ax.clear()
            img = plt.imread(frames[frame_num])
            ax.imshow(img)
            ax.axis('off')
            return []
        return []
    
    anim = animation.FuncAnimation(fig, animate_frame, frames=len(frames), 
                                  interval=200, blit=False, repeat=True)  # 200ms = 5fps
    
    # Save as GIF
    anim.save(gif_path, writer='pillow', fps=5)
    plt.close()
    
    # Clean up temporary files
    for frame_path in frames:
        if os.path.exists(frame_path):
            os.remove(frame_path)
    
    print(f"✅ Crystal Clear ARPMEC GIF saved to: {gif_path}")
    print(f"   Total frames: {len(frames)}")
    print(f"   Duration: ~{len(frames) * 0.2:.1f} seconds")
    print(f"   Frame rate: 5 fps (slow for easy viewing)")
    print(f"\n🎯 KEY FEATURES:")
    print(f"   • 3+ second pauses between rounds")
    print(f"   • Clean cluster boundaries")
    print(f"   • Clear node roles (CH, Member, Idle)")
    print(f"   • Distinct MEC servers")
    print(f"   • Communication flow visualization")
    print(f"   • Network statistics display")
    
    return protocol

def visualize_network_with_communications(protocol: ARPMECProtocol, title: str = "ARPMEC Network Communications"):
    """Static visualization for testing"""
    visualizer = CrystalClearARPMECVisualizer(protocol)
    visualizer.draw_clean_network(0)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def run_crystal_clear_demo():
    """Run the crystal clear demo"""
    print("🎬 ARPMEC Crystal Clear Demo Starting...")
    print("=" * 50)
    
    # Create and run the GIF demo
    protocol = create_crystal_clear_gif_demo()
    
    # Also create a final static visualization
    print("\n📊 Creating final static visualization...")
    visualize_network_with_communications(protocol, "ARPMEC Final Network State")
    
    print("\n✅ Demo complete!")
    print("📁 Check the GIF file: arpmec_crystal_clear.gif")
    print("🎥 Long pauses between rounds for easy observation")
    print("🔍 Crystal clear cluster boundaries and message flows")

if __name__ == "__main__":
    run_crystal_clear_demo()
