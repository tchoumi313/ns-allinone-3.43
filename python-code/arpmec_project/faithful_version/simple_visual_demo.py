#!/usr/bin/env python3
"""
SIMPLE VISUAL DEMO - Just show the nodes moving and working
===========================================================
This is a simplified version that focuses only on showing:
1. Nodes positioned and moving
2. Cluster formation
3. Attacks happening
4. Basic protocol operations

No complex plots - just the network visualization you requested.
"""

import math
import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import the working components
from research_ready_arpmec import SecureARPMECProtocol, AttackerNode, HoneypotNode
from arpmec_faithful import Node, NodeState

class SimpleNetworkVisualization:
    """Simple, clean network visualization"""
    
    def __init__(self, protocol, area_size=800):
        self.protocol = protocol
        self.area_size = area_size
        self.fig = None
        self.ax = None
        self.frame_count = 0
        
    def setup_plot(self):
        """Setup simple plot"""
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3, color='gray')
        
        # Remove axis ticks for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
    def draw_network(self):
        """Draw the network state"""
        self.ax.clear()
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        self.ax.set_facecolor('black')
        self.ax.grid(True, alpha=0.2, color='gray')
        
        # Count status for title
        alive_nodes = sum(1 for n in self.protocol.nodes.values() if n.is_alive())
        cluster_heads = sum(1 for n in self.protocol.nodes.values() 
                          if n.state == NodeState.CLUSTER_HEAD and n.is_alive())
        active_attacks = len(self.protocol.active_attacks)
        
        # Draw cluster areas first (background)
        self.draw_cluster_areas()
        
        # Draw connections
        self.draw_connections()
        
        # Draw infrastructure
        self.draw_mec_servers()
        self.draw_iar_servers()
        
        # Draw nodes by type
        self.draw_regular_nodes()
        self.draw_cluster_heads()
        self.draw_attackers()
        self.draw_honeypots()
        
        # Draw attacks
        self.draw_attacks()
        
        # Set title with status
        title = f'SECURE ARPMEC NETWORK | Nodes: {alive_nodes} | CHs: {cluster_heads} | Attacks: {active_attacks} | Round: {self.frame_count}'
        self.ax.set_title(title, color='white', fontsize=14, weight='bold', pad=20)
        
    def draw_cluster_areas(self):
        """Draw cluster coverage areas"""
        cluster_heads = [n for n in self.protocol.nodes.values() 
                        if n.state == NodeState.CLUSTER_HEAD and n.is_alive()]
        
        for ch in cluster_heads:
            # Simple cluster boundary
            circle = plt.Circle((ch.x, ch.y), 120, fill=False, 
                              color='cyan', alpha=0.3, linestyle='--', linewidth=1)
            self.ax.add_patch(circle)
    
    def draw_connections(self):
        """Draw cluster member connections"""
        for node in self.protocol.nodes.values():
            if (node.state == NodeState.CLUSTER_MEMBER and node.is_alive() and
                hasattr(node, 'cluster_head_id') and node.cluster_head_id in self.protocol.nodes):
                
                ch = self.protocol.nodes[node.cluster_head_id]
                if ch.is_alive():
                    self.ax.plot([node.x, ch.x], [node.y, ch.y], 
                                color='gray', alpha=0.4, linewidth=0.5, zorder=1)
    
    def draw_regular_nodes(self):
        """Draw regular nodes"""
        regular_nodes = [n for n in self.protocol.nodes.values() 
                        if (n.state == NodeState.CLUSTER_MEMBER and n.is_alive() and
                            not isinstance(n, (AttackerNode, HoneypotNode)))]
        
        if regular_nodes:
            for node in regular_nodes:
                # Color by energy
                if node.energy > 70:
                    color = 'lightgreen'
                elif node.energy > 40:
                    color = 'yellow'
                else:
                    color = 'orange'
                
                self.ax.scatter(node.x, node.y, c=color, s=50, marker='o', 
                              edgecolors='white', linewidth=1, alpha=0.8, zorder=3)
                
                # Node ID
                self.ax.text(node.x, node.y-12, f'{node.id}', ha='center', va='center',
                           fontsize=7, color='white', weight='bold')
    
    def draw_cluster_heads(self):
        """Draw cluster head nodes"""
        cluster_heads = [n for n in self.protocol.nodes.values() 
                        if n.state == NodeState.CLUSTER_HEAD and n.is_alive()]
        
        for ch in cluster_heads:
            self.ax.scatter(ch.x, ch.y, c='blue', s=150, marker='^', 
                          edgecolors='white', linewidth=2, alpha=0.9, zorder=5)
            
            # CH label
            self.ax.text(ch.x, ch.y-20, f'CH-{ch.id}', ha='center', va='center',
                       fontsize=9, color='white', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.7))
    
    def draw_attackers(self):
        """Draw attacker nodes"""
        attackers = [n for n in self.protocol.nodes.values() 
                    if isinstance(n, AttackerNode) and n.is_alive()]
        
        for attacker in attackers:
            # Pulsing effect
            pulse = 120 + 30 * math.sin(self.frame_count * 0.3)
            
            self.ax.scatter(attacker.x, attacker.y, c='red', s=pulse, marker='X', 
                          edgecolors='darkred', linewidth=2, alpha=0.8, zorder=6)
            
            # Attacker label
            self.ax.text(attacker.x, attacker.y-25, f'ATK-{attacker.id}', 
                       ha='center', va='center', fontsize=8, color='white', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.8))
    
    def draw_honeypots(self):
        """Draw honeypot nodes"""
        honeypots = [n for n in self.protocol.nodes.values() 
                    if isinstance(n, HoneypotNode) and n.is_alive()]
        
        for honeypot in honeypots:
            captures = len(getattr(honeypot, 'captured_attacks', []))
            color = 'orange' if captures > 0 else 'gold'
            
            self.ax.scatter(honeypot.x, honeypot.y, c=color, s=100, marker='h', 
                          edgecolors='darkorange', linewidth=2, alpha=0.9, zorder=5)
            
            # Honeypot label
            self.ax.text(honeypot.x, honeypot.y-20, f'HP-{honeypot.id}({captures})', 
                       ha='center', va='center', fontsize=8, color='white', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='gold', alpha=0.7))
    
    def draw_mec_servers(self):
        """Draw MEC servers"""
        for mec_id, mec in self.protocol.mec_servers.items():
            load = mec.get_load_percentage()
            
            if load > 80:
                color = 'red'
            elif load > 50:
                color = 'orange'
            else:
                color = 'green'
            
            self.ax.scatter(mec.x, mec.y, c=color, s=250, marker='s', 
                          edgecolors='white', linewidth=2, alpha=0.9, zorder=7)
            
            # MEC label
            self.ax.text(mec.x, mec.y-30, f'MEC-{mec_id}\\n{load:.0f}%', 
                       ha='center', va='center', fontsize=9, color='white', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
    
    def draw_iar_servers(self):
        """Draw IAR servers"""
        for iar_id, iar in self.protocol.iar_servers.items():
            self.ax.scatter(iar.x, iar.y, c='purple', s=200, marker='D', 
                          edgecolors='white', linewidth=2, alpha=0.9, zorder=6)
            
            # IAR label
            self.ax.text(iar.x, iar.y-25, f'IAR-{iar_id}', ha='center', va='center',
                       fontsize=9, color='white', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='purple', alpha=0.7))
    
    def draw_attacks(self):
        """Draw active attacks"""
        for attack in self.protocol.active_attacks:
            if (attack.attacker_id in self.protocol.nodes and 
                attack.target_id in self.protocol.nodes):
                
                attacker = self.protocol.nodes[attack.attacker_id]
                target = self.protocol.nodes[attack.target_id]
                
                # Attack line
                self.ax.plot([attacker.x, target.x], [attacker.y, target.y], 
                           color='red', linewidth=2, alpha=0.7, zorder=4)
                
                # Attack indicator on target
                if attack.detected:
                    # Show detection
                    circle = plt.Circle((target.x, target.y), 30, fill=False, 
                                      color='yellow', linewidth=3, alpha=0.8)
                    self.ax.add_patch(circle)
    
    def update_visualization(self, frame):
        """Update the visualization"""
        self.frame_count = frame
        
        # Run protocol step
        if hasattr(self.protocol, 'run_secure_protocol_step'):
            self.protocol.run_secure_protocol_step(frame)
        
        # Redraw network
        self.draw_network()
        
        return []
    
    def run_demo(self, total_frames=100, interval=1000):
        """Run the demonstration"""
        self.setup_plot()
        
        print("\\n🎬 VISUALIZATION STARTING...")
        print("You should see:")
        print("  🔵 Blue triangles = Cluster Heads")
        print("  ⚪ Colored circles = Regular Nodes (color = energy level)")
        print("  🔴 Red X's = Attackers (pulsing)")
        print("  🟡 Gold hexagons = Honeypots")
        print("  🟢 Green squares = MEC Servers")
        print("  🟣 Purple diamonds = IAR Servers")
        print("  🔗 Gray lines = Cluster connections")
        print("  ⚡ Red lines = Active attacks")
        print("  🟡 Yellow circles = Attack detection")
        print()
        print("Close the window to stop the demo...")
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self.update_visualization, 
            frames=total_frames, interval=interval, 
            blit=False, repeat=True
        )
        
        plt.tight_layout()
        plt.show()
        
        return anim

def create_test_network():
    """Create a test network for visualization"""
    print("🌐 Creating test network...")
    
    # Create normal nodes
    nodes = []
    for i in range(15):
        x = random.uniform(100, 700)
        y = random.uniform(100, 700)
        energy = random.uniform(80, 120)
        nodes.append(Node(i, x, y, energy))
    
    print(f"✓ Created {len(nodes)} normal nodes")
    
    # Create attackers
    attackers = []
    for i in range(2):
        x = random.uniform(150, 650)
        y = random.uniform(150, 650)
        attacker = AttackerNode(15 + i, x, y, 100)
        attackers.append(attacker)
    
    print(f"✓ Created {len(attackers)} attackers")
    
    # Create honeypots
    honeypots = []
    for i in range(2):
        x = random.uniform(200, 600)
        y = random.uniform(200, 600)
        honeypot = HoneypotNode(17 + i, x, y, 120)
        honeypots.append(honeypot)
    
    print(f"✓ Created {len(honeypots)} honeypots")
    
    return nodes, attackers, honeypots

def main():
    """Main demo function"""
    print("🎓 SIMPLE VISUAL DEMO - SECURE ARPMEC")
    print("=" * 50)
    print("This shows the network topology with:")
    print("✓ Node movement and clustering")  
    print("✓ Attack generation and detection")
    print("✓ Security responses")
    print("✓ Real-time protocol operations")
    print("=" * 50)
    
    try:
        # Create network
        nodes, attackers, honeypots = create_test_network()
        
        # Create protocol
        print("🔧 Initializing protocol...")
        protocol = SecureARPMECProtocol(nodes, attackers, honeypots)
        
        # Run initial clustering
        clusters = protocol.clustering_algorithm()
        print(f"✓ Created {len(clusters)} clusters")
        
        # Create and run visualization
        print("📊 Starting visualization...")
        visualizer = SimpleNetworkVisualization(protocol)
        animation_obj = visualizer.run_demo(total_frames=200, interval=800)  # 0.8 seconds per frame
        
        input("\\nPress Enter to exit...")
        
    except KeyboardInterrupt:
        print("\\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
