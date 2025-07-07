#!/usr/bin/env python3
"""
Working ARPMEC Demonstration with Clear Visualization
This version focuses on clarity and actually shows the network properly
"""

import math
import random
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from arpmec_faithful import (ARPMECProtocol, IARServer, MECServer, Node,
                             NodeState)

# Configure matplotlib for better display
plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11

class WorkingARPMECDemo:
    """A working ARPMEC demonstration that actually displays properly"""
    
    def __init__(self, N: int = 20, area_size: int = 800):
        self.N = N
        self.area_size = area_size
        self.protocol = None
        self.current_round = 0
        
        # Colors for different node types
        self.colors = {
            'ch': '#FF0000',         # Red for cluster heads
            'member': '#0000FF',     # Blue for cluster members
            'idle': '#808080',       # Gray for idle nodes
            'mec': '#8B0000',        # Dark red for MEC servers
            'iar': '#4B0082',        # Indigo for IAR servers
        }
        
        # Cluster colors
        self.cluster_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        
    def create_realistic_network(self) -> List[Node]:
        """Create a realistic network with clustered nodes"""
        nodes = []
        random.seed(42)  # For reproducible results
        
        # Create 3 cluster areas
        cluster_centers = [
            (200, 200),   # Bottom-left
            (600, 200),   # Bottom-right
            (400, 600),   # Top-center
        ]
        
        nodes_per_cluster = self.N // 3
        remaining = self.N % 3
        
        node_id = 0
        for i, (cx, cy) in enumerate(cluster_centers):
            count = nodes_per_cluster + (1 if i < remaining else 0)
            
            for j in range(count):
                # Place nodes within 60m of cluster center
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(10, 60)
                
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                
                # Keep within bounds
                x = max(50, min(self.area_size - 50, x))
                y = max(50, min(self.area_size - 50, y))
                
                energy = random.uniform(90, 110)
                nodes.append(Node(node_id, x, y, energy))
                node_id += 1
        
        print(f"Created {len(nodes)} nodes in clustered formation")
        return nodes
    
    def setup_network(self):
        """Set up the ARPMEC network"""
        print("Setting up ARPMEC network...")
        
        # Create nodes
        nodes = self.create_realistic_network()
        
        # Initialize protocol
        self.protocol = ARPMECProtocol(nodes, self.area_size)
        
        # Perform initial clustering
        clusters = self.protocol.clustering_algorithm()
        print(f"Initial clustering created {len(clusters)} clusters")
        
        # Show cluster assignments
        for cluster_id, member_ids in clusters.items():
            print(f"  Cluster {cluster_id}: CH={cluster_id}, Members={member_ids}")
        
        return clusters
    
    def draw_network(self, round_num: int = 0, show_connections: bool = True):
        """Draw the current network state"""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Set up the plot
        ax.set_xlim(-50, self.area_size + 50)
        ax.set_ylim(-50, self.area_size + 50)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        
        # Title
        ax.set_title(f'ARPMEC Protocol Visualization - Round {round_num}\n'
                    f'Adaptive Routing with IAR Infrastructure',
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
        
        # 1. Draw cluster areas first
        for ch in cluster_heads:
            if ch.id in cluster_color_map:
                color = cluster_color_map[ch.id]
                
                # Cluster communication range
                circle = plt.Circle((ch.x, ch.y), self.protocol.communication_range,
                                  fill=False, color=color, alpha=0.6, 
                                  linestyle='--', linewidth=2, zorder=1)
                ax.add_patch(circle)
                
                # Cluster core area
                core = plt.Circle((ch.x, ch.y), 30,
                                fill=True, color=color, alpha=0.2, zorder=1)
                ax.add_patch(core)
        
        # 2. Draw IAR servers
        for iar in self.protocol.iar_servers.values():
            # IAR coverage area
            coverage = plt.Circle((iar.x, iar.y), iar.coverage_radius,
                                fill=True, color='purple', alpha=0.1, zorder=1)
            ax.add_patch(coverage)
            
            # IAR server as diamond
            diamond = patches.RegularPolygon((iar.x, iar.y), 4, 25,
                                           orientation=math.pi/4,
                                           facecolor=self.colors['iar'],
                                           edgecolor='black',
                                           linewidth=2, alpha=0.9, zorder=4)
            ax.add_patch(diamond)
            
            # IAR label
            ax.text(iar.x, iar.y - 40, f'IAR-{iar.id}',
                   fontsize=12, ha='center', weight='bold', color='purple')
            ax.text(iar.x, iar.y - 52, f'{len(iar.connected_clusters)} clusters',
                   fontsize=9, ha='center', color='gray')
        
        # 3. Draw MEC servers
        for server in self.protocol.mec_servers.values():
            # MEC coverage area
            coverage = plt.Circle((server.x, server.y), 80,
                                fill=True, color='darkred', alpha=0.1, zorder=1)
            ax.add_patch(coverage)
            
            # MEC server as square
            load_pct = server.get_load_percentage()
            load_color = 'red' if load_pct > 80 else 'orange' if load_pct > 60 else 'green'
            
            square = patches.Rectangle((server.x - 20, server.y - 20), 40, 40,
                                     facecolor=self.colors['mec'],
                                     edgecolor=load_color,
                                     linewidth=3, alpha=0.9, zorder=4)
            ax.add_patch(square)
            
            # MEC label
            ax.text(server.x, server.y - 35, f'MEC-{server.id}',
                   fontsize=12, ha='center', weight='bold', color='darkred')
            ax.text(server.x, server.y - 47, f'{load_pct:.0f}% load',
                   fontsize=9, ha='center', color=load_color, weight='bold')
        
        # 4. Draw connections if requested
        if show_connections:
            # IAR to MEC connections
            for iar in self.protocol.iar_servers.values():
                for mec_id in iar.connected_mec_servers:
                    if mec_id in self.protocol.mec_servers:
                        mec = self.protocol.mec_servers[mec_id]
                        ax.plot([iar.x, mec.x], [iar.y, mec.y],
                               color='purple', linewidth=2, alpha=0.5,
                               linestyle=':', zorder=2)
            
            # Cluster head to IAR connections
            for ch in cluster_heads:
                if ch.assigned_iar_id in self.protocol.iar_servers:
                    iar = self.protocol.iar_servers[ch.assigned_iar_id]
                    ax.plot([ch.x, iar.x], [ch.y, iar.y],
                           color='blue', linewidth=1, alpha=0.4, zorder=2)
        
        # 5. Draw nodes
        # Cluster heads
        for ch in cluster_heads:
            color = cluster_color_map.get(ch.id, '#FF0000')
            ax.scatter(ch.x, ch.y, c=color, s=200, marker='*',
                      edgecolors='black', linewidth=2, zorder=5)
            ax.text(ch.x + 15, ch.y + 15, f'CH-{ch.id}',
                   fontsize=10, weight='bold', color=color)
        
        # Cluster members
        for member in cluster_members:
            ch_id = member.cluster_head_id
            color = cluster_color_map.get(ch_id, '#0000FF')
            ax.scatter(member.x, member.y, c=color, s=100, marker='o',
                      edgecolors='black', linewidth=1, alpha=0.8, zorder=5)
            ax.text(member.x + 10, member.y + 10, f'{member.id}',
                   fontsize=8, color=color)
        
        # Idle nodes
        for idle in idle_nodes:
            ax.scatter(idle.x, idle.y, c=self.colors['idle'], s=80, marker='o',
                      edgecolors='black', linewidth=1, alpha=0.6, zorder=5)
            ax.text(idle.x + 10, idle.y + 10, f'{idle.id}',
                   fontsize=8, color='gray')
        
        # 6. Statistics
        total_mec_load = sum(s.get_load_percentage() for s in self.protocol.mec_servers.values())
        avg_mec_load = total_mec_load / len(self.protocol.mec_servers) if self.protocol.mec_servers else 0
        
        stats_text = f"Network Status (Round {round_num}):\n"
        stats_text += f"Infrastructure: {len(self.protocol.iar_servers)} IAR, {len(self.protocol.mec_servers)} MEC\n"
        stats_text += f"Clusters: {len(cluster_heads)} | Members: {len(cluster_members)} | Idle: {len(idle_nodes)}\n"
        stats_text += f"Average MEC Load: {avg_mec_load:.1f}%\n"
        stats_text += f"Total Alive Nodes: {len([n for n in self.protocol.nodes.values() if n.is_alive()])}"
        
        if round_num > 0 and round_num % 5 == 0:
            stats_text += "\n🔄 RE-CLUSTERING ACTIVE!"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5",
                facecolor='yellow' if round_num % 5 == 0 else 'lightcyan',
                alpha=0.9))
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                      markersize=12, label='Cluster Head'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                      markersize=8, label='Cluster Member'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                      markersize=6, label='Idle Node'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkred',
                      markersize=10, label='MEC Server'),
            plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='purple',
                      markersize=10, label='IAR Server'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Y Position (meters)', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def simulate_round(self, round_num: int):
        """Simulate one round of the protocol"""
        print(f"\n--- Simulating Round {round_num} ---")
        
        # Update mobility
        area_bounds = (0, self.area_size, 0, self.area_size)
        for node in self.protocol.nodes.values():
            if node.is_alive():
                node.update_mobility(area_bounds)
        
        # Update protocol time
        self.protocol.current_time_slot = round_num
        
        # Re-clustering every 5 rounds
        if round_num > 0 and round_num % 5 == 0:
            print(f"  🔄 Performing re-clustering...")
            self.protocol._check_and_recluster()
        
        # Generate network traffic
        self.protocol._generate_inter_cluster_traffic()
        self.protocol._generate_mec_tasks()
        
        # Process protocol operations
        self.protocol._process_inter_cluster_messages()
        self.protocol._process_mec_servers()
        self.protocol._check_cluster_head_validity()
        
        # Update current round
        self.current_round = round_num
        
        print(f"  ✓ Round {round_num} completed")
    
    def run_interactive_demo(self, max_rounds: int = 20):
        """Run an interactive demonstration"""
        print("Starting ARPMEC Interactive Demo...")
        print("=" * 50)
        
        # Setup network
        clusters = self.setup_network()
        
        if not clusters:
            print("❌ No clusters formed! Check network parameters.")
            return
        
        # Show initial state
        print("\n🎯 Showing initial network state...")
        fig = self.draw_network(0, show_connections=True)
        plt.show()
        
        # Run simulation rounds
        for round_num in range(1, max_rounds + 1):
            print(f"\n{'='*20} ROUND {round_num} {'='*20}")
            
            # Simulate the round
            self.simulate_round(round_num)
            
            # Show visualization every 5 rounds or on user request
            if round_num % 5 == 0 or round_num == 1:
                print(f"\n🎯 Showing network state after round {round_num}...")
                fig = self.draw_network(round_num, show_connections=True)
                plt.show()
                
                # Wait for user input
                if round_num < max_rounds:
                    input(f"\nPress Enter to continue to next phase (Round {round_num + 1})...")
        
        print("\n✅ Demo completed!")
        print("Network simulation finished successfully.")
    
    def create_static_visualization(self, filename: str = "arpmec_network.png"):
        """Create a static visualization of the current network"""
        print(f"\nCreating static visualization: {filename}")
        
        fig = self.draw_network(self.current_round, show_connections=True)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved as {filename}")
        
        return fig

def main():
    """Main function to run the demo"""
    print("🚀 ARPMEC Protocol Working Demo")
    print("=" * 40)
    
    # Create demo instance
    demo = WorkingARPMECDemo(N=20, area_size=800)
    
    # Run the interactive demo
    try:
        demo.run_interactive_demo(max_rounds=15)
        
        # Create final static visualization
        demo.create_static_visualization("arpmec_final_network.png")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
