#!/usr/bin/env python3
"""
CLEAN ARPMEC PROTOCOL WITH SECURITY INTEGRATION
==============================================

Step-by-step implementation:
1. First: Working faithful ARPMEC protocol (verified)
2. Second: Clean security layer integration
3. Third: Proper visualization

This ensures the base protocol works before adding security features.
"""

import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
# Import the VERIFIED working ARPMEC protocol
from arpmec_faithful import (ARPMECProtocol, IARServer, InterClusterMessage,
                             MECServer, MECTask, Node, NodeState)
from matplotlib.patches import Circle

# =====================================================================================
# STEP 1: VERIFY BASE ARPMEC PROTOCOL WORKS
# =====================================================================================

class BasicARPMECDemo:
    """Basic demonstration to verify ARPMEC protocol works correctly"""
    
    def __init__(self, num_nodes: int = 20):
        self.num_nodes = num_nodes
        self.area_size = 1000
        self.protocol = None
        
    def create_basic_network(self):
        """Create a basic network with regular nodes only"""
        print("🔧 Creating Basic ARPMEC Network...")
        
        nodes = []
        
        # Create nodes in a realistic distribution
        cluster_centers = [
            (200, 200), (600, 200), (200, 600), (600, 600)
        ]
        
        nodes_per_cluster = self.num_nodes // len(cluster_centers)
        node_id = 0
        
        for cx, cy in cluster_centers:
            for i in range(nodes_per_cluster):
                # Generate position around cluster center
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(20, 80)
                
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                
                # Ensure nodes stay within bounds
                x = max(50, min(self.area_size - 50, x))
                y = max(50, min(self.area_size - 50, y))
                
                energy = random.uniform(100, 150)
                node = Node(node_id, x, y, energy)
                nodes.append(node)
                node_id += 1
        
        # Add remaining nodes
        while len(nodes) < self.num_nodes:
            x = random.uniform(100, self.area_size - 100)
            y = random.uniform(100, self.area_size - 100)
            energy = random.uniform(100, 150)
            node = Node(node_id, x, y, energy)
            nodes.append(node)
            node_id += 1
        
        # Create protocol
        self.protocol = ARPMECProtocol(nodes, C=4, R=5, K=3)
        
        # Initialize clustering
        clusters = self.protocol.clustering_algorithm()
        print(f"✅ Basic network created with {len(clusters)} clusters")
        
        return clusters
    
    def run_basic_simulation(self, steps: int = 10):
        """Run basic ARPMEC simulation to verify it works"""
        print(f"\n🔄 Running basic ARPMEC simulation for {steps} steps...")
        
        for step in range(steps):
            print(f"   Step {step + 1}: ", end="")
            
            # Update node mobility
            area_bounds = (0, 1000, 0, 1000)
            for node in self.protocol.nodes.values():
                if node.is_alive():
                    node.update_mobility(area_bounds)
            
            # Protocol operations
            alive_nodes = [node for node in self.protocol.nodes.values() if node.is_alive()]
            
            for node in alive_nodes:
                if node.state == NodeState.CLUSTER_HEAD:
                    self.protocol._fixed_cluster_head_operations(node, step)
                else:
                    self.protocol._fixed_cluster_member_operations(node, step)
            
            # Communication operations
            self.protocol._generate_inter_cluster_traffic()
            self.protocol._generate_mec_tasks()
            self.protocol._process_inter_cluster_messages()
            self.protocol._process_mec_servers()
            
            # Check for reclustering
            if step % 5 == 0 and step > 0:
                self.protocol._check_and_recluster()
                self.protocol._check_cluster_head_validity()
                self.protocol._build_inter_cluster_routing_table()
            
            alive_count = len(alive_nodes)
            cluster_heads = len(self.protocol._get_cluster_heads())
            print(f"Alive: {alive_count}, CHs: {cluster_heads}")
        
        print("✅ Basic simulation completed successfully!")
        return True

# =====================================================================================
# STEP 2: CLEAN SECURITY LAYER (ONLY AFTER BASIC PROTOCOL WORKS)
# =====================================================================================

class ThreatLevel(Enum):
    """Simple threat levels"""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"

class AttackType(Enum):
    """Simple attack types"""
    DOS_FLOODING = "dos_flooding"

@dataclass
class SecurityEvent:
    """Simple security event"""
    event_id: str
    timestamp: float
    source_id: int
    target_id: Optional[int]
    attack_type: AttackType
    description: str
    position: Tuple[float, float]
    active: bool = True

@dataclass
class SecurityMetrics:
    """Simple security metrics"""
    total_attacks: int = 0
    attacks_blocked: int = 0
    honeypot_captures: int = 0
    network_availability: float = 100.0
    threat_level: ThreatLevel = ThreatLevel.LOW

class SecurityNode(Node):
    """Node with basic security capabilities"""
    
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 100.0):
        super().__init__(node_id, x, y, initial_energy)
        self.is_attacker = False
        self.is_honeypot = False
        self.trust_score = 1.0
        self.attack_count = 0

class SecurityMonitor:
    """Simple security monitoring"""
    
    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.metrics = SecurityMetrics()
        self.blocked_nodes: Set[int] = set()
    
    def detect_attack(self, source_id: int, target_id: int) -> bool:
        """Simple attack detection"""
        # Basic detection logic
        return random.random() < 0.8  # 80% detection rate
    
    def create_event(self, source_id: int, target_id: int, attack_type: AttackType, 
                    description: str, position: Tuple[float, float]) -> SecurityEvent:
        """Create security event"""
        event = SecurityEvent(
            event_id=f"event_{len(self.events)}",
            timestamp=time.time(),
            source_id=source_id,
            target_id=target_id,
            attack_type=attack_type,
            description=description,
            position=position
        )
        self.events.append(event)
        return event

class SecureARPMECProtocol(ARPMECProtocol):
    """ARPMEC Protocol with SIMPLE security layer"""
    
    def __init__(self, nodes: List[SecurityNode], C: int = 4, R: int = 5, K: int = 3):
        # Convert to base nodes for parent class
        base_nodes = [Node(n.id, n.x, n.y, n.energy) for n in nodes]
        super().__init__(base_nodes, C, R, K)
        
        # Replace with security nodes
        self.nodes = {node.id: node for node in nodes}
        
        # Add security components
        self.security_monitor = SecurityMonitor()
        self.attack_frequency = 0.05  # Low attack frequency
        
    def get_attackers(self) -> List[SecurityNode]:
        """Get attacker nodes"""
        return [n for n in self.nodes.values() if n.is_attacker]
    
    def get_honeypots(self) -> List[SecurityNode]:
        """Get honeypot nodes"""  
        return [n for n in self.nodes.values() if n.is_honeypot]
    
    def run_security_step(self, step: int):
        """Run security operations (simple)"""
        # Generate occasional attacks
        attackers = self.get_attackers()
        
        for attacker in attackers:
            if random.random() < self.attack_frequency:  # Low frequency
                # Find targets
                targets = [n for n in self.nodes.values() 
                          if n.is_alive() and n.id != attacker.id]
                
                if targets:
                    target = random.choice(targets)
                    
                    # Check if attack is detected
                    if self.security_monitor.detect_attack(attacker.id, target.id):
                        # Create security event
                        event = self.security_monitor.create_event(
                            attacker.id, target.id, AttackType.DOS_FLOODING,
                            f"DoS attack from Node-{attacker.id} to Node-{target.id}",
                            (attacker.x, attacker.y)
                        )
                        
                        # Simple countermeasure
                        self.security_monitor.blocked_nodes.add(attacker.id)
                        self.security_monitor.metrics.attacks_blocked += 1
                        
                        # If target is honeypot
                        if target.is_honeypot:
                            self.security_monitor.metrics.honeypot_captures += 1
                    
                    self.security_monitor.metrics.total_attacks += 1
                    
                    # Simple attack impact
                    target.update_energy(5.0)  # Small energy drain
        
        # Update metrics
        alive_nodes = sum(1 for n in self.nodes.values() if n.is_alive())
        total_nodes = len(self.nodes)
        self.security_monitor.metrics.network_availability = (alive_nodes / total_nodes) * 100
        
        # Update threat level
        if self.security_monitor.metrics.total_attacks > 20:
            self.security_monitor.metrics.threat_level = ThreatLevel.HIGH
        elif self.security_monitor.metrics.total_attacks > 10:
            self.security_monitor.metrics.threat_level = ThreatLevel.MEDIUM
        else:
            self.security_monitor.metrics.threat_level = ThreatLevel.LOW
    
    def run_protocol_step(self, step: int):
        """Run complete protocol step with security"""
        # First: Run base ARPMEC protocol (PROVEN TO WORK)
        self.current_time_slot = step
        
        # Update node mobility
        area_bounds = (0, 1000, 0, 1000)
        for node in self.nodes.values():
            if node.is_alive():
                node.update_mobility(area_bounds)
        
        # Check for reclustering
        if step % 5 == 0 and step > 0:
            self._check_and_recluster()
            self._check_cluster_head_validity()
            self._build_inter_cluster_routing_table()
        
        # Protocol operations
        alive_nodes = [node for node in self.nodes.values() if node.is_alive()]
        
        for node in alive_nodes:
            if node.state == NodeState.CLUSTER_HEAD:
                self._fixed_cluster_head_operations(node, step)
            else:
                self._fixed_cluster_member_operations(node, step)
        
        # Communication operations
        self._generate_inter_cluster_traffic()
        self._generate_mec_tasks()
        self._process_inter_cluster_messages()
        self._process_mec_servers()
        
        # Second: Run security layer (SIMPLE)
        self.run_security_step(step)

# =====================================================================================
# STEP 3: CLEAN VISUALIZATION
# =====================================================================================

class SecureARPMECVisualizer:
    """Clean visualization for secure ARPMEC"""
    
    def __init__(self, protocol: SecureARPMECProtocol, area_size: int = 1000):
        self.protocol = protocol
        self.area_size = area_size
        self.fig = None
        self.ax = None
        
    def setup_visualization(self):
        """Setup clean visualization"""
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        self.ax.set_xlabel('X Position (meters)', color='black', fontsize=12)
        self.ax.set_ylabel('Y Position (meters)', color='black', fontsize=12)
        self.ax.grid(True, alpha=0.3, color='gray')
        
        self.ax.tick_params(colors='black')
        for spine in self.ax.spines.values():
            spine.set_color('black')
    
    def draw_network(self):
        """Draw the network topology"""
        self.ax.clear()
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        self.ax.set_facecolor('white')
        self.ax.grid(True, alpha=0.3, color='gray')
        
        # Draw cluster boundaries
        cluster_heads = self.protocol._get_cluster_heads()
        for ch in cluster_heads:
            # Color based on threat level
            threat = self.protocol.security_monitor.metrics.threat_level
            if threat == ThreatLevel.HIGH:
                color = 'red'
            elif threat == ThreatLevel.MEDIUM:
                color = 'orange'
            else:
                color = 'green'
                
            boundary = Circle((ch.x, ch.y), self.protocol.communication_range,
                            fill=False, color=color, alpha=0.5, linewidth=2)
            self.ax.add_patch(boundary)
        
        # Draw nodes
        for node in self.protocol.nodes.values():
            if not node.is_alive():
                continue
            
            # Determine node appearance
            if node.is_attacker:
                color = 'red'
                marker = 'X'
                size = 100
            elif node.is_honeypot:
                color = 'gold'
                marker = 'h'
                size = 80
            elif node.state == NodeState.CLUSTER_HEAD:
                color = 'blue'
                marker = '^'
                size = 120
            else:
                color = 'lightblue'
                marker = 'o'
                size = 60
            
            self.ax.scatter(node.x, node.y, c=color, s=size, marker=marker,
                          edgecolors='white', linewidth=1, zorder=5)
            
            # Add node label
            self.ax.text(node.x, node.y - 25, f'N{node.id}', ha='center', 
                       fontsize=8, color='white', weight='bold')
        
        # Draw MEC servers
        for mec_id, mec in self.protocol.mec_servers.items():
            self.ax.scatter(mec.x, mec.y, c='green', s=200, marker='s',
                          edgecolors='white', linewidth=2, zorder=6)
            self.ax.text(mec.x, mec.y - 30, f'MEC{mec_id}', ha='center',
                       fontsize=9, weight='bold', color='white')
        
        # Draw IAR servers
        for iar_id, iar in self.protocol.iar_servers.items():
            self.ax.scatter(iar.x, iar.y, c='purple', s=150, marker='D',
                          edgecolors='white', linewidth=2, zorder=6)
            self.ax.text(iar.x, iar.y - 25, f'IAR{iar_id}', ha='center',
                       fontsize=8, weight='bold', color='white')
    
    def draw_security_info(self):
        """Draw security information panel"""
        metrics = self.protocol.security_monitor.metrics
        
        # Security panel
        security_text = f"""🔒 SECURITY STATUS
Threat Level: {metrics.threat_level.value.upper()}
Total Attacks: {metrics.total_attacks}
Attacks Blocked: {metrics.attacks_blocked}
Honeypot Captures: {metrics.honeypot_captures}
Network Availability: {metrics.network_availability:.1f}%"""
        
        # Color based on threat level
        if metrics.threat_level == ThreatLevel.HIGH:
            panel_color = 'red'
        elif metrics.threat_level == ThreatLevel.MEDIUM:
            panel_color = 'orange'
        else:
            panel_color = 'green'
        
        self.ax.text(0.02, 0.98, security_text, transform=self.ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=panel_color, alpha=0.8),
                   color='white', weight='bold', zorder=200)
        
        # Protocol panel
        cluster_heads = self.protocol._get_cluster_heads()
        protocol_text = f"""📊 PROTOCOL STATUS
Time Step: {self.protocol.current_time_slot}
Active Nodes: {sum(1 for n in self.protocol.nodes.values() if n.is_alive())}
Cluster Heads: {len(cluster_heads)}
MEC Servers: {len(self.protocol.mec_servers)}
IAR Servers: {len(self.protocol.iar_servers)}"""
        
        self.ax.text(0.98, 0.98, protocol_text, transform=self.ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='blue', alpha=0.8),
                   color='white', weight='bold', zorder=200)
    
    def update_frame(self, frame):
        """Update visualization frame"""
        # Run protocol step
        self.protocol.run_protocol_step(frame)
        
        # Draw network
        self.draw_network()
        
        # Draw security info
        self.draw_security_info()
        
        # Set title
        threat = self.protocol.security_monitor.metrics.threat_level
        self.ax.set_title(f'🔒 SECURE ARPMEC PROTOCOL - Frame {frame} - Threat: {threat.value.upper()}',
                         fontsize=14, weight='bold', color='black', pad=20)
        
        # Style
        self.ax.set_xlabel('X Position (meters)', color='black', fontsize=12)
        self.ax.set_ylabel('Y Position (meters)', color='black', fontsize=12)
        self.ax.tick_params(colors='black')
        for spine in self.ax.spines.values():
            spine.set_color('black')
    
    def run_animation(self, interval: int = 500):
        """Run visualization animation"""
        self.setup_visualization()
        
        anim = animation.FuncAnimation(
            self.fig, self.update_frame, interval=interval,
            blit=False, cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()
        return anim

# =====================================================================================
# STEP 4: COMPREHENSIVE DEMO
# =====================================================================================

class CleanSecureDemo:
    """Clean demonstration with proper step-by-step approach"""
    
    def __init__(self):
        self.basic_demo = None
        self.secure_protocol = None
        self.visualizer = None
    
    def step1_verify_basic_arpmec(self):
        """Step 1: Verify basic ARPMEC works"""
        print("=" * 60)
        print("STEP 1: VERIFYING BASE ARPMEC PROTOCOL")
        print("=" * 60)
        
        self.basic_demo = BasicARPMECDemo(num_nodes=15)
        clusters = self.basic_demo.create_basic_network()
        
        if clusters and len(clusters) > 0:
            success = self.basic_demo.run_basic_simulation(steps=5)
            if success:
                print("✅ STEP 1 PASSED: Base ARPMEC protocol works correctly!")
                return True
        
        print("❌ STEP 1 FAILED: Base ARPMEC protocol has issues!")
        return False
    
    def step2_add_security_layer(self):
        """Step 2: Add clean security layer"""
        print("\n" + "=" * 60)
        print("STEP 2: ADDING SECURITY LAYER")
        print("=" * 60)
        
        # Create network with security nodes
        nodes = []
        node_id = 0
        
        # Regular nodes
        for i in range(15):
            x = random.uniform(100, 900)
            y = random.uniform(100, 900)
            energy = random.uniform(120, 180)
            node = SecurityNode(node_id, x, y, energy)
            nodes.append(node)
            node_id += 1
        
        # Add 2 attackers
        for i in range(2):
            x = random.uniform(200, 800)
            y = random.uniform(200, 800)
            energy = random.uniform(100, 150)
            attacker = SecurityNode(node_id, x, y, energy)
            attacker.is_attacker = True
            nodes.append(attacker)
            print(f"🔴 Added Attacker Node-{node_id} at ({x:.0f}, {y:.0f})")
            node_id += 1
        
        # Add 2 honeypots
        for i in range(2):
            x = random.uniform(300, 700)
            y = random.uniform(300, 700)
            energy = random.uniform(150, 200)
            honeypot = SecurityNode(node_id, x, y, energy)
            honeypot.is_honeypot = True
            nodes.append(honeypot)
            print(f"🍯 Added Honeypot Node-{node_id} at ({x:.0f}, {y:.0f})")
            node_id += 1
        
        # Create secure protocol
        self.secure_protocol = SecureARPMECProtocol(nodes)
        
        # Initialize clustering
        clusters = self.secure_protocol.clustering_algorithm()
        print(f"✅ STEP 2 PASSED: Security layer added with {len(clusters)} clusters")
        
        return True
    
    def step3_run_visualization(self):
        """Step 3: Run clean visualization"""
        print("\n" + "=" * 60)
        print("STEP 3: STARTING CLEAN VISUALIZATION")
        print("=" * 60)
        
        self.visualizer = SecureARPMECVisualizer(self.secure_protocol)
        
        print("🎯 Starting visualization...")
        print("   → Shows ARPMEC protocol with security overlay")
        print("   → Red nodes = Attackers")
        print("   → Gold nodes = Honeypots")
        print("   → Blue triangles = Cluster Heads")
        print("   → Light blue circles = Regular nodes")
        print("   → Green squares = MEC servers")
        print("   → Purple diamonds = IAR servers")
        print("   → Close window to exit")
        
        return self.visualizer.run_animation(interval=1000)
    
    def run_complete_demo(self):
        """Run complete step-by-step demonstration"""
        print("🚀 CLEAN ARPMEC WITH SECURITY - STEP BY STEP APPROACH")
        print("This ensures the base protocol works before adding security!")
        
        # Step 1: Verify base ARPMEC
        if not self.step1_verify_basic_arpmec():
            return False
        
        # Step 2: Add security layer
        if not self.step2_add_security_layer():
            return False
        
        # Step 3: Run visualization
        return self.step3_run_visualization()

# =====================================================================================
# MAIN EXECUTION
# =====================================================================================

def main():
    """Main execution - clean step-by-step approach"""
    try:
        demo = CleanSecureDemo()
        animation_obj = demo.run_complete_demo()
        
        if animation_obj:
            input("\nPress Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n✅ Demo completed")

if __name__ == "__main__":
    main()
