#!/usr/bin/env python3
"""
VISUAL RESEARCH DEMO - SECURE ARPMEC PROTOCOL
============================================

This provides real-time visualization of the research simulation so you can:
1. See nodes moving and clustering
2. Watch attacks happening in real-time  
3. Observe security responses (detection, honeypots, countermeasures)
4. Validate that the metrics being collected are accurate
5. Ensure the protocol is working as expected

Perfect for research validation and demonstration!
"""

import json
import math
import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

# Import the research simulation components
from research_ready_arpmec import (
    SecureARPMECProtocol, AttackerNode, HoneypotNode, 
    AttackType, Attack, SecurityMetrics, PerformanceMetrics
)
from arpmec_faithful import Node, NodeState

# =====================================================================================
# ENHANCED VISUALIZATION CLASS FOR RESEARCH
# =====================================================================================

class ResearchVisualization:
    """Real-time visualization for research validation"""
    
    def __init__(self, protocol: SecureARPMECProtocol, area_size: int = 800):
        self.protocol = protocol
        self.area_size = area_size
        self.fig = None
        self.ax = None
        self.frame_count = 0
        self.start_time = time.time()
        
        # Visualization state
        self.attack_animations = []
        self.detection_animations = []
        self.message_trails = []
        
        # Metrics for display
        self.metrics_history = {
            'detection_rate': [],
            'energy_levels': [],
            'pdr': [],
            'active_attacks': []
        }
        
    def setup_visualization(self):
        """Setup simple network visualization"""
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        
        # Main network visualization only
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        self.ax.set_aspect('equal')
        self.ax.set_title('🔐 SECURE ARPMEC NETWORK - RESEARCH VALIDATION', 
                         fontsize=16, weight='bold', color='white')
        self.ax.set_xlabel('X Position (meters)', color='white')
        self.ax.set_ylabel('Y Position (meters)', color='white')
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def draw_network_topology(self):
        """Draw the complete network with all components"""
        self.ax.clear()
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        self.ax.grid(True, alpha=0.3)
        
        # Draw cluster boundaries first
        self.draw_cluster_boundaries()
        
        # Draw infrastructure
        self.draw_mec_servers()
        self.draw_iar_servers()
        
        # Draw all nodes
        self.draw_regular_nodes()
        self.draw_cluster_heads()
        self.draw_attackers()
        self.draw_honeypots()
        
        # Draw connections
        self.draw_cluster_connections()
        
        # Draw active attacks and responses
        self.draw_attack_animations()
        self.draw_security_responses()
        
        # Draw protocol status
        self.draw_protocol_status()
        
    def draw_cluster_boundaries(self):
        """Draw cluster coverage areas"""
        cluster_heads = [node for node in self.protocol.nodes.values() 
                        if node.state == NodeState.CLUSTER_HEAD and node.is_alive()]
        
        for ch in cluster_heads:
            # Determine boundary color based on security threat level
            threat_color = 'green'
            alpha = 0.2
            
            # Check for nearby attacks
            nearby_attacks = sum(1 for attack in self.protocol.active_attacks 
                               if hasattr(self.protocol.nodes.get(attack.target_id), 'x') and
                               math.sqrt((self.protocol.nodes[attack.target_id].x - ch.x)**2 + 
                                       (self.protocol.nodes[attack.target_id].y - ch.y)**2) < 150)
            
            if nearby_attacks > 2:
                threat_color = 'red'
                alpha = 0.4
            elif nearby_attacks > 0:
                threat_color = 'orange'
                alpha = 0.3
                
            circle = plt.Circle((ch.x, ch.y), 120, fill=True, 
                              color=threat_color, alpha=alpha, 
                              linestyle='--', linewidth=2)
            self.ax.add_patch(circle)
    
    def draw_regular_nodes(self):
        """Draw regular member nodes"""
        regular_nodes = [node for node in self.protocol.nodes.values() 
                        if (node.state == NodeState.CLUSTER_MEMBER and 
                            node.is_alive() and 
                            not isinstance(node, (AttackerNode, HoneypotNode)))]
        
        if regular_nodes:
            x_coords = [node.x for node in regular_nodes]
            y_coords = [node.y for node in regular_nodes]
            energies = [node.energy for node in regular_nodes]
            
            # Color by energy level
            colors = ['lightgreen' if e > 70 else 'yellow' if e > 40 else 'red' 
                     for e in energies]
            
            self.ax.scatter(x_coords, y_coords, c=colors, s=60, 
                               marker='o', edgecolors='white', linewidth=1, 
                               alpha=0.8, zorder=3)
            
            # Add node labels
            for node in regular_nodes:
                self.ax.text(node.x, node.y-15, f'N{node.id}', 
                                ha='center', va='center', fontsize=8, 
                                color='white', weight='bold')
    
    def draw_cluster_heads(self):
        """Draw cluster head nodes"""
        cluster_heads = [node for node in self.protocol.nodes.values() 
                        if node.state == NodeState.CLUSTER_HEAD and node.is_alive()]
        
        if cluster_heads:
            x_coords = [node.x for node in cluster_heads]
            y_coords = [node.y for node in cluster_heads]
            
            self.ax.scatter(x_coords, y_coords, c='blue', s=200, 
                               marker='^', edgecolors='white', linewidth=2, 
                               alpha=0.9, zorder=5)
            
            # Add labels with cluster info
            for ch in cluster_heads:
                members = sum(1 for node in self.protocol.nodes.values() 
                            if node.cluster_head_id == ch.id and node.is_alive())
                self.ax.text(ch.x, ch.y-25, f'CH-{ch.id}\n({members})', 
                                ha='center', va='center', fontsize=10, 
                                color='white', weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='blue', alpha=0.7))
    
    def draw_attackers(self):
        """Draw attacker nodes with attack indicators"""
        attackers = [node for node in self.protocol.nodes.values() 
                    if isinstance(node, AttackerNode) and node.is_alive()]
        
        if attackers:
            for attacker in attackers:
                # Pulsing effect for active attackers
                pulse_size = 150
                if hasattr(attacker, 'attack_cooldown') and attacker.attack_cooldown <= 0:
                    pulse_size += 50 * math.sin(self.frame_count * 0.5)
                
                self.ax.scatter(attacker.x, attacker.y, c='red', s=pulse_size, 
                                   marker='X', edgecolors='darkred', linewidth=3, 
                                   alpha=0.8, zorder=6)
                
                # Attack range indicator
                circle = plt.Circle((attacker.x, attacker.y), 100, fill=False, 
                                  color='red', alpha=0.5, linestyle=':')
                self.ax.add_patch(circle)
                
                # Label
                self.ax.text(attacker.x, attacker.y-25, f'ATK-{attacker.id}', 
                                ha='center', va='center', fontsize=9, 
                                color='white', weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8))
    
    def draw_honeypots(self):
        """Draw honeypot nodes with capture indicators"""
        honeypots = [node for node in self.protocol.nodes.values() 
                    if isinstance(node, HoneypotNode) and node.is_alive()]
        
        if honeypots:
            for honeypot in honeypots:
                # Different colors based on activity
                color = 'gold'
                if hasattr(honeypot, 'captured_attacks') and len(honeypot.captured_attacks) > 0:
                    color = 'orange'  # Active honeypot
                
                self.ax.scatter(honeypot.x, honeypot.y, c=color, s=120, 
                                   marker='h', edgecolors='darkorange', linewidth=2, 
                                   alpha=0.9, zorder=5)
                
                # Attraction radius
                circle = plt.Circle((honeypot.x, honeypot.y), 80, fill=False, 
                                  color='gold', alpha=0.4, linestyle='--')
                self.ax.add_patch(circle)
                
                # Label with captures
                captures = len(getattr(honeypot, 'captured_attacks', []))
                self.ax.text(honeypot.x, honeypot.y-20, f'HP-{honeypot.id}\\n({captures})', 
                                ha='center', va='center', fontsize=9, 
                                color='white', weight='bold',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='gold', alpha=0.7))
    
    def draw_mec_servers(self):
        """Draw MEC servers with load indicators"""
        for mec_id, mec in self.protocol.mec_servers.items():
            # Color based on load
            load = mec.get_load_percentage()
            if load > 80:
                color = 'red'
            elif load > 50:
                color = 'orange' 
            else:
                color = 'green'
            
            self.ax.scatter(mec.x, mec.y, c=color, s=300, 
                               marker='s', edgecolors='white', linewidth=3, 
                               alpha=0.9, zorder=7)
            
            # Load bar
            bar_height = load / 100 * 30
            self.ax.add_patch(plt.Rectangle((mec.x-10, mec.y+20), 20, bar_height, 
                                               facecolor=color, alpha=0.7))
            
            # Label
            self.ax.text(mec.x, mec.y-30, f'MEC-{mec_id}\\n{load:.1f}%', 
                            ha='center', va='center', fontsize=10, 
                            color='white', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
    
    def draw_iar_servers(self):
        """Draw IAR servers with connectivity"""
        for iar_id, iar in self.protocol.iar_servers.items():
            self.ax.scatter(iar.x, iar.y, c='purple', s=250, 
                               marker='D', edgecolors='white', linewidth=2, 
                               alpha=0.9, zorder=6)
            
            # Coverage area
            circle = plt.Circle((iar.x, iar.y), getattr(iar, 'coverage_radius', 200), 
                              fill=False, color='purple', alpha=0.3, linestyle=':')
            self.ax.add_patch(circle)
            
            # Label
            self.ax.text(iar.x, iar.y-25, f'IAR-{iar_id}', 
                            ha='center', va='center', fontsize=9, 
                            color='white', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='purple', alpha=0.7))
    
    def draw_cluster_connections(self):
        """Draw connections between cluster members and heads"""
        for node in self.protocol.nodes.values():
            if (node.state == NodeState.CLUSTER_MEMBER and 
                node.is_alive() and 
                hasattr(node, 'cluster_head_id') and 
                node.cluster_head_id in self.protocol.nodes):
                
                ch = self.protocol.nodes[node.cluster_head_id]
                if ch.is_alive():
                    self.ax.plot([node.x, ch.x], [node.y, ch.y], 
                                    'gray', alpha=0.3, linewidth=1, zorder=1)
    
    def draw_attack_animations(self):
        """Draw active attacks with animation"""
        for attack in self.protocol.active_attacks:
            if (attack.attacker_id in self.protocol.nodes and 
                attack.target_id in self.protocol.nodes):
                
                attacker = self.protocol.nodes[attack.attacker_id]
                target = self.protocol.nodes[attack.target_id]
                
                # Attack beam
                self.ax.plot([attacker.x, target.x], [attacker.y, target.y], 
                                'red', linewidth=3, alpha=0.7, zorder=4)
                
                # Moving attack packet
                progress = (time.time() - attack.timestamp) % 2.0 / 2.0
                packet_x = attacker.x + (target.x - attacker.x) * progress
                packet_y = attacker.y + (target.y - attacker.y) * progress
                
                self.ax.scatter(packet_x, packet_y, c='red', s=80, 
                                   marker='*', alpha=0.9, zorder=8)
    
    def draw_security_responses(self):
        """Draw security detection and response indicators"""
        # Detection events
        current_time = time.time()
        
        for attack in self.protocol.active_attacks:
            if attack.detected and (current_time - attack.timestamp) < 3.0:
                target = self.protocol.nodes.get(attack.target_id)
                if target:
                    # Detection flash
                    flash_alpha = 0.8 * (1 - (current_time - attack.timestamp) / 3.0)
                    circle = plt.Circle((target.x, target.y), 50, fill=True, 
                                      color='yellow', alpha=flash_alpha, zorder=7)
                    self.ax.add_patch(circle)
    
    def draw_protocol_status(self):
        """Draw protocol status information"""
        # Count various node types
        alive_nodes = sum(1 for node in self.protocol.nodes.values() if node.is_alive())
        cluster_heads = sum(1 for node in self.protocol.nodes.values() 
                          if node.state == NodeState.CLUSTER_HEAD and node.is_alive())
        active_attacks = len(self.protocol.active_attacks)
        detected_attacks = sum(1 for attack in self.protocol.active_attacks if attack.detected)
        
        # Protocol status panel
        status_text = f"""🔐 PROTOCOL STATUS
Time: {time.time() - self.start_time:.1f}s
Alive Nodes: {alive_nodes}
Cluster Heads: {cluster_heads}
MEC Servers: {len(self.protocol.mec_servers)}
IAR Servers: {len(self.protocol.iar_servers)}
Active Attacks: {active_attacks}
Detected: {detected_attacks}
Detection Rate: {(detected_attacks/max(active_attacks,1)*100):.1f}%"""
        
        self.ax.text(0.02, 0.98, status_text, transform=self.ax.transAxes,
                         fontsize=11, verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.8),
                         color='white', weight='bold', zorder=200)
    
    def update_frame(self, frame):
        """Update the network visualization"""
        self.frame_count = frame
        
        # Run one step of the protocol
        if hasattr(self.protocol, 'run_secure_protocol_step'):
            self.protocol.run_secure_protocol_step(frame)
        
        # Update network visualization
        self.draw_network_topology()
        
        # Update title with current status
        current_time = time.time() - self.start_time
        alive_nodes = sum(1 for node in self.protocol.nodes.values() if node.is_alive())
        active_attacks = len(self.protocol.active_attacks)
        detected_attacks = sum(1 for attack in self.protocol.active_attacks if attack.detected)
        detection_rate = (detected_attacks / max(active_attacks, 1)) * 100
        
        title = f'🔐 SECURE ARPMEC RESEARCH - Time: {current_time:.1f}s | Nodes: {alive_nodes} | Attacks: {active_attacks} | Detection: {detection_rate:.1f}%'
        self.ax.set_title(title, fontsize=14, weight='bold', color='white')
    
    def run_animation(self, interval: int = 500):
        """Run the research visualization animation"""
        self.setup_visualization()
        
        anim = animation.FuncAnimation(
            self.fig, self.update_frame, interval=interval,
            blit=False, cache_frame_data=False, repeat=True
        )
        
        plt.show()
        return anim

# =====================================================================================
# RESEARCH DEMO FUNCTION
# =====================================================================================

def run_visual_research_demo(num_rounds: int = 50):
    """Run a single visual research demonstration"""
    print("🎓 VISUAL RESEARCH DEMO - SECURE ARPMEC")
    print("=" * 50)
    print("This demo shows:")
    print("✓ Real-time node movement and clustering")
    print("✓ Attack generation and detection")
    print("✓ Security responses (honeypots, countermeasures)")
    print("✓ Performance metrics collection")
    print("✓ Research data validation")
    print("=" * 50)
    
    # Create network topology (simplified for compatibility)
    print("🌐 Creating network topology...")
    
    # Import the actual research protocol and node types
    from research_ready_arpmec import SecureARPMECProtocol, AttackerNode, HoneypotNode
    
    # Create nodes using the same method as research simulation
    nodes = []
    for i in range(20):  # 20 normal nodes
        x = random.uniform(50, 750)
        y = random.uniform(50, 750)
        energy = random.uniform(90, 110)
        nodes.append(Node(i, x, y, energy))
    
    print(f"✓ Created {len(nodes)} normal nodes")
    
    # Add attackers
    attackers = []
    for i in range(3):  # 3 attackers
        x = random.uniform(100, 700)
        y = random.uniform(100, 700)
        attacker = AttackerNode(20 + i, x, y, 120)
        attackers.append(attacker)
    
    print(f"✓ Created {len(attackers)} attacker nodes")
    
    # Add honeypots
    honeypots = []
    for i in range(2):  # 2 honeypots
        x = random.uniform(150, 650)
        y = random.uniform(150, 650)
        honeypot = HoneypotNode(23 + i, x, y, 150)
        honeypots.append(honeypot)
    
    print(f"✓ Created {len(honeypots)} honeypot nodes")
    
    # Create protocol with all node types
    print("🔧 Initializing secure ARPMEC protocol...")
    protocol = SecureARPMECProtocol(nodes, attackers, honeypots)
    
    # Initialize clustering
    clusters = protocol.clustering_algorithm()
    print(f"✓ Created {len(clusters)} initial clusters")
    
    # Create visualization
    print("📊 Setting up visualization...")
    visualizer = ResearchVisualization(protocol, area_size=800)
    
    print("\\n🎬 Starting visual demo...")
    print("Close the window to end the demonstration")
    print("Watch for:")
    print("  🔵 Blue triangles = Cluster Heads")
    print("  🔴 Red X's = Attackers (with pulsing when active)")
    print("  🟡 Gold hexagons = Honeypots")
    print("  🟢 Green squares = MEC Servers")
    print("  🟣 Purple diamonds = IAR Servers")
    print("  ⚡ Real-time attacks and security responses")
    
    # Run the visual demonstration
    return visualizer.run_animation(interval=1000)  # 1 second per frame

# =====================================================================================
# MAIN EXECUTION
# =====================================================================================

def main():
    """Main execution for visual research demo"""
    print("🎓 SECURE ARPMEC - VISUAL RESEARCH VALIDATION")
    print("📊 Real-time visualization of research simulation")
    print("🔍 Validate protocol behavior and metrics collection")
    print("⏱️  Perfect for research verification!")
    print()
    
    choice = input("Choose mode:\\n1. Visual Demo (recommended for validation)\\n2. Data Collection Only\\nEnter choice (1/2): ")
    
    if choice == "1" or choice == "":
        print("\\n🎬 Starting visual research demonstration...")
        try:
            animation_obj = run_visual_research_demo(num_rounds=50)
            input("\\nPress Enter to exit...")
        except KeyboardInterrupt:
            print("\\n🛑 Demo interrupted by user")
        except Exception as e:
            print(f"\\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\\n📊 Running data collection simulation...")
        from research_ready_arpmec import main as research_main
        research_main()

if __name__ == "__main__":
    main()
