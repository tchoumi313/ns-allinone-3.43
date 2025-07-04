#!/usr/bin/env python3
"""
Crystal Clear ARPMEC Visualization Demo
Creates clean, understandable visualizations with long pauses for GIF creation
"""

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from arpmec_faithful import ARPMECProtocol, Node, NodeState

# Set up matplotlib for better visualization
plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

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

class CrystalClearARPMECDemo:
    """Crystal clear ARPMEC demonstration with clean visualization"""
    
    def __init__(self, N: int = 20, area_size: int = 800):
        self.N = N
        self.area_size = area_size
        self.protocol = None
        self.rounds = 0
        self.max_rounds = 40
        
        # Clean color palette
        self.cluster_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        
        # Animation frames storage
        self.frames = []
        
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
    
    def create_clean_frame(self, round_num: int, title_suffix: str = "") -> plt.Figure:
        """Create a single clean frame of the visualization"""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Clear, simple background
        ax.set_facecolor('#F8F9FA')
        ax.set_xlim(-50, self.area_size + 50)
        ax.set_ylim(-50, self.area_size + 50)
        
        # Get current network state
        cluster_heads = self.protocol._get_cluster_heads()
        cluster_members = [n for n in self.protocol.nodes.values() 
                          if n.state == NodeState.CLUSTER_MEMBER and n.is_alive()]
        idle_nodes = [n for n in self.protocol.nodes.values() 
                     if n.state == NodeState.IDLE and n.is_alive()]
        
        # Debug information
        print(f"Debug - Round {round_num}: CHs={len(cluster_heads)}, Members={len(cluster_members)}, Idle={len(idle_nodes)}")
        for ch in cluster_heads:
            member_count = len([m for m in cluster_members if m.cluster_head_id == ch.id])
            print(f"  CH-{ch.id}: {member_count} members")
            for member in cluster_members:
                if member.cluster_head_id == ch.id:
                    print(f"    Member-{member.id} at ({member.x:.1f}, {member.y:.1f})")
        
        # Create cluster color mapping
        cluster_color_map = {}
        for i, ch in enumerate(cluster_heads):
            cluster_color_map[ch.id] = self.cluster_colors[i % len(self.cluster_colors)]
        
        # 1. Draw MEC servers first (background)
        for server in self.protocol.mec_servers.values():
            # MEC server coverage area (50m as requested)
            coverage = plt.Circle((server.x, server.y), 50, 
                                fill=True, color='purple', alpha=0.2, zorder=1)
            ax.add_patch(coverage)
            
            # MEC server
            ax.scatter(server.x, server.y, c='purple', s=600, marker='s', 
                      edgecolors='black', linewidth=3, alpha=0.9, zorder=3)
            ax.annotate(f'MEC-{server.id}', (server.x, server.y), 
                       xytext=(0, -40), textcoords='offset points',
                       fontsize=14, fontweight='bold', ha='center', color='purple')
        
        # 2. Draw cluster areas (clean circles)
        for ch in cluster_heads:
            if ch.id in cluster_color_map:
                color = cluster_color_map[ch.id]
                
                # Cluster area (10m coverage as requested)
                cluster_area = plt.Circle((ch.x, ch.y), 10,
                                        fill=True, color=color, alpha=0.25, zorder=2)
                ax.add_patch(cluster_area)
                
                # Cluster boundary
                boundary = plt.Circle((ch.x, ch.y), self.protocol.communication_range,
                                    fill=False, color=color, alpha=0.8, 
                                    linestyle='--', linewidth=2, zorder=2)
                ax.add_patch(boundary)
        
        # 3. Draw clean connections
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
        
        # 4. Draw nodes with clear symbols
        # Cluster heads
        for ch in cluster_heads:
            if ch.id in cluster_color_map:
                color = cluster_color_map[ch.id]
                ax.scatter(ch.x, ch.y, c=color, s=400, marker='^',
                          edgecolors='black', linewidth=3, zorder=6)
                ax.annotate(f'CH-{ch.id}', (ch.x, ch.y), xytext=(8, 8),
                           textcoords='offset points', fontsize=12, fontweight='bold')
        
        # Cluster members (enhanced with distance labels)
        for member in cluster_members:
            if member.cluster_head_id and member.cluster_head_id in cluster_color_map:
                color = cluster_color_map[member.cluster_head_id]
                ax.scatter(member.x, member.y, c=color, s=200, marker='o',
                          alpha=0.9, edgecolors='black', linewidth=2, zorder=7)
                
                # Add member labels for debugging
                ax.annotate(f'M{member.id}', (member.x, member.y), xytext=(-5, -15),
                           textcoords='offset points', fontsize=10, fontweight='bold')
                
                # Add distance label on connection line to CH
                ch = next((ch for ch in cluster_heads if ch.id == member.cluster_head_id), None)
                if ch:
                    distance = member.distance_to(ch)
                    mid_x = (member.x + ch.x) / 2
                    mid_y = (member.y + ch.y) / 2
                    ax.annotate(f'{distance:.0f}m', (mid_x, mid_y), fontsize=9,
                               ha='center', va='center', 
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor=color),
                               zorder=5)
        
        # Idle nodes
        if idle_nodes:
            for node in idle_nodes:
                ax.scatter(node.x, node.y, c='lightgray', s=80, marker='s',
                          alpha=0.7, edgecolors='black', linewidth=1, zorder=5)
        
        # 5. Clean legend
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                      markersize=15, label='Cluster Heads'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=10, label='Cluster Members'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', 
                      markersize=15, label='MEC Servers'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', 
                      markersize=8, label='Idle Nodes'),
            plt.Line2D([0], [0], color='black', linewidth=3, alpha=0.7, 
                      label='CH-to-MEC Communication'),
            plt.Line2D([0], [0], color='gray', linewidth=2, alpha=0.6, 
                      label='Member-to-CH Communication')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        # 6. Clean title and info with re-clustering indication
        if round_num % 5 == 0 and round_num > 0:
            title_suffix = f" 🔄 (RE-CLUSTERING EVENT!)"
        elif title_suffix == "":
            next_recluster = ((round_num // 5) + 1) * 5
            title_suffix = f" (Next re-clustering: Round {next_recluster})"
            
        main_title = f"ARPMEC Protocol - Round {round_num}{title_suffix}"
        subtitle = f"CH Election: Energy + Link Quality + Distance (<{self.protocol.communication_range}m)"
        ax.set_title(f"{main_title}\n{subtitle}", fontsize=14, fontweight='bold', pad=20)
        
        # Network stats with enhanced debugging info
        ch_member_counts = {}
        total_distances = 0
        connection_count = 0
        
        for ch in cluster_heads:
            member_count = len([m for m in cluster_members if m.cluster_head_id == ch.id])
            ch_member_counts[ch.id] = member_count
            
            # Calculate average distance for this cluster
            for m in cluster_members:
                if m.cluster_head_id == ch.id:
                    dist = m.distance_to(ch)
                    total_distances += dist
                    connection_count += 1
        
        avg_distance = total_distances / max(connection_count, 1)
        
        stats_text = f"Clusters: {len(cluster_heads)} | Members: {len(cluster_members)} | Idle: {len(idle_nodes)}"
        stats_text += f"\nAvg Member Distance: {avg_distance:.1f}m"
        
        # Show cluster details
        cluster_details = []
        for ch_id, count in list(ch_member_counts.items())[:4]:  # Show first 4 clusters
            cluster_details.append(f"CH-{ch_id}: {count}m")
        if cluster_details:
            stats_text += f"\n{' | '.join(cluster_details)}"
        
        if round_num % 5 == 0 and round_num > 0:
            stats_text += f"\n🔄 RE-CLUSTERING ACTIVE!"
        
        # Debug printing
        print(f"\n=== FRAME DEBUG - Round {round_num} ===")
        print(f"Cluster Heads: {len(cluster_heads)} - IDs: {[ch.id for ch in cluster_heads]}")
        print(f"Cluster Members: {len(cluster_members)} - IDs: {[m.id for m in cluster_members]}")
        print(f"Idle Nodes: {len(idle_nodes)} - IDs: {[n.id for n in idle_nodes]}")
        
        # Detailed cluster membership for first 3 clusters
        for ch in cluster_heads[:3]:
            member_details = []
            for m in cluster_members:
                if m.cluster_head_id == ch.id:
                    dist = m.distance_to(ch)
                    member_details.append(f"Node-{m.id}({dist:.0f}m)")
            print(f"  CH-{ch.id} at ({ch.x:.0f},{ch.y:.0f}): {len(member_details)} members: {member_details}")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor='yellow' if round_num % 5 == 0 and round_num > 0 else 'white', 
                alpha=0.9))
        
        # Grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        
        plt.tight_layout()
        return fig
    
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
    demo = CrystalClearARPMECDemo(N=20, area_size=800)
    
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
            demo = CrystalClearARPMECDemo()
            demo.protocol = protocol
            fig = demo.create_clean_frame(demo.max_rounds - 1, "(Final State)")
            plt.savefig('arpmec_final_state.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("✅ Static visualization saved as 'arpmec_final_state.png'")
        except Exception as e:
            print(f"⚠️  Static visualization error: {e}")
    
    print("\n✅ Demo complete!")
    print("Check the GIF file for the animated visualization.")

if __name__ == "__main__":
    run_crystal_clear_demo()
