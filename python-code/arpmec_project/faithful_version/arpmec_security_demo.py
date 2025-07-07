#!/usr/bin/env python3
"""
ARPMEC SECURITY DEMO - Attack and Honeypot Implementation
This file implements security features including:
1. Malicious nodes (attackers)
2. DoS attacks targeting CH and MEC servers
3. Honeypot implementation
4. Attack detection and mitigation

Based on the ARPMEC protocol with security enhancements.
"""

import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from arpmec_faithful import (ARPMECProtocol, IARServer, InterClusterMessage,
                             MECServer, MECTask, Node, NodeState)


class AttackType(Enum):
    """Types of attacks that can be performed"""
    DOS_FLOODING = "dos_flooding"
    IDENTITY_SPOOFING = "identity_spoofing"
    CH_TARGETING = "ch_targeting"
    MEC_OVERLOAD = "mec_overload"
    ICMP_FLOOD = "icmp_flood"
    UDP_FLOOD = "udp_flood"

class NodeRole(Enum):
    """Extended node roles including security roles"""
    NORMAL = "normal"
    ATTACKER = "attacker"
    HONEYPOT = "honeypot"
    SECURITY_MONITOR = "security_monitor"

@dataclass
class AttackPacket:
    """Represents a malicious packet"""
    packet_id: str
    source_id: int
    target_id: int
    attack_type: AttackType
    payload_size: float  # Size in KB
    timestamp: float
    spoofed_identity: Optional[int] = None
    frequency: float = 1.0  # Packets per second

@dataclass
class AttackPattern:
    """Defines an attack pattern"""
    attack_type: AttackType
    target_types: List[str]  # ["CH", "MEC", "IAR"]
    packet_rate: float  # Packets per second
    duration: float  # Attack duration in seconds
    payload_size: float  # Packet size in KB
    stealth_level: float  # 0.0 (obvious) to 1.0 (highly stealthy)
    
class MaliciousNode(Node):
    """Malicious node that can perform various attacks"""
    
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 100.0):
        super().__init__(node_id, x, y, initial_energy)
        self.role = NodeRole.ATTACKER
        self.attack_patterns: List[AttackPattern] = []
        self.active_attacks: List[AttackPacket] = []
        self.spoofed_identities: List[int] = []
        self.attack_targets: List[int] = []
        self.stealth_mode = True
        self.attack_cooldown = 0.0
        self.detection_probability = 0.1  # Base detection probability
        
        # Attack capabilities
        self.dos_capability = random.uniform(0.5, 1.0)
        self.spoofing_capability = random.uniform(0.3, 0.9)
        self.stealth_capability = random.uniform(0.4, 0.8)
        
    def select_targets(self, protocol: ARPMECProtocol) -> List[Tuple[int, str]]:
        """Select attack targets (CH, MEC, IAR)"""
        targets = []
        
        # Target cluster heads (high priority)
        cluster_heads = protocol._get_cluster_heads()
        for ch in cluster_heads:
            if ch.id != self.id:
                distance = self.distance_to(ch)
                if distance <= protocol.inter_cluster_range:
                    targets.append((ch.id, "CH"))
        
        # Target MEC servers (critical infrastructure)
        for mec_id, mec in protocol.mec_servers.items():
            distance = math.sqrt((self.x - mec.x)**2 + (self.y - mec.y)**2)
            if distance <= protocol.mec_communication_range:
                targets.append((mec_id, "MEC"))
        
        # Target IAR servers (network infrastructure)
        for iar_id, iar in protocol.iar_servers.items():
            distance = math.sqrt((self.x - iar.x)**2 + (self.y - iar.y)**2)
            if distance <= iar.coverage_radius:
                targets.append((iar_id, "IAR"))
        
        return targets
    
    def launch_dos_attack(self, target_id: int, target_type: str, protocol: ARPMECProtocol):
        """Launch DoS attack against target"""
        
        if self.attack_cooldown > 0:
            return
        
        # Create attack pattern
        attack_pattern = AttackPattern(
            attack_type=AttackType.DOS_FLOODING,
            target_types=[target_type],
            packet_rate=random.uniform(50, 200),  # High packet rate
            duration=random.uniform(10, 30),
            payload_size=random.uniform(1, 10),
            stealth_level=self.stealth_capability
        )
        
        # Generate attack packets
        num_packets = int(attack_pattern.packet_rate * attack_pattern.duration)
        
        for i in range(num_packets):
            attack_packet = AttackPacket(
                packet_id=f"attack_{self.id}_{target_id}_{i}",
                source_id=self.id,
                target_id=target_id,
                attack_type=AttackType.DOS_FLOODING,
                payload_size=attack_pattern.payload_size,
                timestamp=time.time() + i / attack_pattern.packet_rate,
                frequency=attack_pattern.packet_rate
            )
            self.active_attacks.append(attack_packet)
        
        print(f"🔴 ATTACK: Node-{self.id} launched DoS against {target_type}-{target_id} "
              f"({num_packets} packets at {attack_pattern.packet_rate:.1f} pps)")
        
        # Set cooldown to avoid immediate detection
        self.attack_cooldown = random.uniform(20, 60)
        
        # Consume energy for attack
        attack_energy = attack_pattern.packet_rate * attack_pattern.duration * 0.05
        self.update_energy(attack_energy)
    
    def spoof_identity(self, target_identity: int):
        """Spoof another node's identity"""
        if target_identity not in self.spoofed_identities:
            self.spoofed_identities.append(target_identity)
            print(f"🎭 SPOOFING: Node-{self.id} is now impersonating Node-{target_identity}")
    
    def generate_excessive_traffic(self, target_servers: List[int], protocol: ARPMECProtocol):
        """Generate excessive traffic towards zone servers"""
        
        for server_id in target_servers:
            if server_id in protocol.mec_servers:
                server = protocol.mec_servers[server_id]
                
                # Create excessive MEC tasks
                for i in range(random.randint(5, 15)):
                    malicious_task = MECTask(
                        task_id=f"malicious_task_{self.id}_{server_id}_{i}",
                        source_cluster_id=self.cluster_id or self.id,
                        cpu_requirement=random.uniform(20, 50),  # Very high CPU requirement
                        memory_requirement=random.uniform(100, 200),  # Very high memory
                        deadline=protocol.current_time_slot + 1,  # Urgent deadline
                        data_size=random.uniform(50, 100),  # Large data
                        created_time=protocol.current_time_slot
                    )
                    
                    # Try to overwhelm server
                    if not server.accept_task(malicious_task):
                        print(f"🔴 ATTACK: MEC-{server_id} overwhelmed by Node-{self.id}")
                        break
    
    def maintain_stealth(self, protocol: ARPMECProtocol):
        """Maintain stealth to avoid detection"""
        
        # Reduce attack frequency if detection risk is high
        nearby_monitors = self._find_nearby_security_monitors(protocol)
        
        if len(nearby_monitors) > 0:
            self.detection_probability = min(0.8, self.detection_probability + 0.1)
            self.stealth_mode = True
            
            # Reduce attack activity
            self.attack_cooldown = max(self.attack_cooldown, 30.0)
        else:
            self.detection_probability = max(0.1, self.detection_probability - 0.05)
            self.stealth_mode = False
        
        # Update attack cooldown
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1.0
    
    def _find_nearby_security_monitors(self, protocol: ARPMECProtocol) -> List[int]:
        """Find nearby security monitoring nodes"""
        monitors = []
        
        for node_id, node in protocol.nodes.items():
            if (hasattr(node, 'role') and node.role == NodeRole.SECURITY_MONITOR and 
                node.id != self.id):
                distance = self.distance_to(node)
                if distance <= protocol.communication_range:
                    monitors.append(node_id)
        
        return monitors

class HoneypotNode(Node):
    """Honeypot node that attracts and analyzes attacks"""
    
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 150.0):
        super().__init__(node_id, x, y, initial_energy)
        self.role = NodeRole.HONEYPOT
        self.attracted_attacks: List[AttackPacket] = []
        self.attacker_profiles: Dict[int, Dict] = {}
        self.fake_vulnerabilities: List[str] = []
        self.fake_load = random.uniform(0.7, 0.9)  # Appear almost overloaded
        self.deception_level = random.uniform(0.6, 0.9)
        
        # Honeypot capabilities
        self.traffic_absorption_capacity = random.uniform(100, 500)  # Packets per second
        self.analysis_capability = random.uniform(0.7, 0.95)
        
        # Simulate attractive weaknesses
        self._create_fake_vulnerabilities()
    
    def _create_fake_vulnerabilities(self):
        """Create fake vulnerabilities to attract attackers"""
        vulnerabilities = [
            "high_cpu_load",
            "memory_overflow_risk",
            "weak_authentication",
            "buffer_overflow_potential",
            "unencrypted_communications"
        ]
        
        # Select random vulnerabilities to advertise
        num_vulns = random.randint(1, 3)
        self.fake_vulnerabilities = random.sample(vulnerabilities, num_vulns)
        
        print(f"🍯 HONEYPOT: Node-{self.id} advertising vulnerabilities: {', '.join(self.fake_vulnerabilities)}")
    
    def mimic_important_node(self, target_type: str):
        """Mimic an important node (CH, MEC) to attract attacks"""
        
        if target_type == "CH":
            # Pretend to be a cluster head
            self.state = NodeState.CLUSTER_HEAD
            self.cluster_id = self.id
            self.cluster_members = [random.randint(100, 999) for _ in range(3, 8)]
            print(f"🍯 HONEYPOT: Node-{self.id} mimicking Cluster Head with {len(self.cluster_members)} members")
            
        elif target_type == "MEC":
            # Pretend to be near a MEC server with high load
            print(f"🍯 HONEYPOT: Node-{self.id} mimicking overloaded MEC server ({self.fake_load*100:.1f}% load)")
    
    def simulate_weakness(self):
        """Simulate attractive weaknesses"""
        # Advertise high load to attract DoS attacks
        weakness_ads = []
        
        if "high_cpu_load" in self.fake_vulnerabilities:
            weakness_ads.append(f"CPU: {self.fake_load*100:.1f}%")
            
        if "memory_overflow_risk" in self.fake_vulnerabilities:
            weakness_ads.append(f"Memory: {(self.fake_load + 0.1)*100:.1f}%")
            
        return weakness_ads
    
    def absorb_malicious_traffic(self, attack_packet: AttackPacket) -> bool:
        """Absorb and analyze malicious traffic"""
        
        # Check if we can handle this attack
        current_load = len(self.attracted_attacks)
        
        if current_load < self.traffic_absorption_capacity:
            self.attracted_attacks.append(attack_packet)
            
            # Profile the attacker
            attacker_id = attack_packet.source_id
            if attacker_id not in self.attacker_profiles:
                self.attacker_profiles[attacker_id] = {
                    'attack_count': 0,
                    'attack_types': set(),
                    'total_payload': 0.0,
                    'avg_frequency': 0.0,
                    'first_seen': time.time(),
                    'threat_level': 'LOW'
                }
            
            profile = self.attacker_profiles[attacker_id]
            profile['attack_count'] += 1
            profile['attack_types'].add(attack_packet.attack_type.value)
            profile['total_payload'] += attack_packet.payload_size
            profile['avg_frequency'] = (profile['avg_frequency'] + attack_packet.frequency) / 2
            
            # Assess threat level
            if profile['attack_count'] > 50:
                profile['threat_level'] = 'CRITICAL'
            elif profile['attack_count'] > 20:
                profile['threat_level'] = 'HIGH'
            elif profile['attack_count'] > 5:
                profile['threat_level'] = 'MEDIUM'
            
            print(f"🍯 HONEYPOT: Node-{self.id} absorbed attack from Node-{attacker_id} "
                  f"(Type: {attack_packet.attack_type.value}, Threat: {profile['threat_level']})")
            
            return True
        
        return False
    
    def respond_to_attacks(self) -> Dict[int, str]:
        """Generate responses to maintain deception while gathering info"""
        responses = {}
        
        for attacker_id, profile in self.attacker_profiles.items():
            if profile['attack_count'] > 10:
                # Pretend to be overwhelmed to encourage more attacks
                responses[attacker_id] = "ERROR: Service temporarily unavailable"
            elif profile['attack_count'] > 5:
                # Show signs of stress
                responses[attacker_id] = "WARNING: High load detected"
            else:
                # Appear normal to avoid suspicion
                responses[attacker_id] = "OK: Request processed"
        
        return responses
    
    def protect_network(self, protocol: ARPMECProtocol):
        """Use gathered intelligence to protect the network"""
        
        # Share threat intelligence with security monitors
        threat_report = {
            'honeypot_id': self.id,
            'total_attacks_absorbed': len(self.attracted_attacks),
            'unique_attackers': len(self.attacker_profiles),
            'high_threat_attackers': [
                aid for aid, profile in self.attacker_profiles.items() 
                if profile['threat_level'] in ['HIGH', 'CRITICAL']
            ],
            'attack_patterns': list(set(
                attack.attack_type.value for attack in self.attracted_attacks
            ))
        }
        
        print(f"🛡️ PROTECTION: Honeypot-{self.id} sharing threat intel: "
              f"{len(threat_report['high_threat_attackers'])} high-threat attackers detected")
        
        return threat_report

class ARPMECSecurityDemo:
    """ARPMEC Security Demonstration with attacks and honeypots"""
    
    def __init__(self, num_nodes: int = 25, num_attackers: int = 3, num_honeypots: int = 2, area_size: int = 1200):
        self.num_nodes = num_nodes
        self.num_attackers = num_attackers
        self.num_honeypots = num_honeypots
        self.area_size = area_size
        self.protocol = None
        
        # Security tracking
        self.active_attacks: List[AttackPacket] = []
        self.detected_attackers: List[int] = []
        self.security_alerts: List[str] = []
        self.attack_timeline: List[Tuple[float, str]] = []
        
        # Colors for visualization
        self.security_colors = {
            'normal': '#4444FF',
            'attacker': '#FF0000',
            'honeypot': '#FFD700',
            'security_monitor': '#00FF00',
            'cluster_head': '#FF4444',
            'idle': '#CCCCCC'
        }
        
        self.attack_colors = {
            'dos_flooding': '#FF0000',
            'identity_spoofing': '#FF6600',
            'ch_targeting': '#CC0000',
            'mec_overload': '#990000',
            'icmp_flood': '#FF3333',
            'udp_flood': '#FF6666'
        }
    
    def create_secure_network(self):
        """Create network with normal nodes, attackers, and honeypots"""
        print("Creating ARPMEC network with security features...")
        
        nodes = []
        node_id = 0
        
        # Create normal nodes
        cluster_centers = [(200, 200), (600, 200), (400, 600), (800, 800)]
        normal_nodes = self.num_nodes - self.num_attackers - self.num_honeypots
        nodes_per_cluster = normal_nodes // len(cluster_centers)
        
        for cx, cy in cluster_centers:
            for i in range(nodes_per_cluster):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, 80)
                
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                x = max(50, min(self.area_size - 50, x))
                y = max(50, min(self.area_size - 50, y))
                
                energy = random.uniform(90, 110)
                node = Node(node_id, x, y, energy)
                node.role = NodeRole.NORMAL
                nodes.append(node)
                node_id += 1
        
        # Add remaining normal nodes randomly
        while len(nodes) < normal_nodes:
            x = random.uniform(50, self.area_size - 50)
            y = random.uniform(50, self.area_size - 50)
            energy = random.uniform(90, 110)
            node = Node(node_id, x, y, energy)
            node.role = NodeRole.NORMAL
            nodes.append(node)
            node_id += 1
        
        # Create malicious nodes (attackers)
        for i in range(self.num_attackers):
            x = random.uniform(100, self.area_size - 100)
            y = random.uniform(100, self.area_size - 100)
            energy = random.uniform(120, 150)  # Higher energy for sustained attacks
            
            attacker = MaliciousNode(node_id, x, y, energy)
            nodes.append(attacker)
            node_id += 1
            print(f"🔴 ATTACKER: Created malicious node {attacker.id} at ({x:.0f}, {y:.0f})")
        
        # Create honeypot nodes
        for i in range(self.num_honeypots):
            # Place honeypots strategically near cluster centers
            center = random.choice(cluster_centers)
            x = center[0] + random.uniform(-120, 120)
            y = center[1] + random.uniform(-120, 120)
            x = max(50, min(self.area_size - 50, x))
            y = max(50, min(self.area_size - 50, y))
            energy = random.uniform(150, 200)  # High energy for continuous operation
            
            honeypot = HoneypotNode(node_id, x, y, energy)
            honeypot.mimic_important_node("CH" if i % 2 == 0 else "MEC")
            nodes.append(honeypot)
            node_id += 1
            print(f"🍯 HONEYPOT: Created honeypot node {honeypot.id} at ({x:.0f}, {y:.0f})")
        
        # Create protocol
        self.protocol = ARPMECProtocol(nodes, C=4, R=5, K=3)
        
        # Perform initial clustering
        clusters = self.protocol.clustering_algorithm()
        print(f"Created {len(clusters)} clusters with security nodes")
        
        return clusters
    
    def simulate_security_scenario(self):
        """Simulate a complete security scenario"""
        print("\n🚨 SECURITY SIMULATION STARTING")
        print("=" * 50)
        
        # Phase 1: Network reconnaissance
        print("\n📡 Phase 1: Network Reconnaissance")
        attackers = [n for n in self.protocol.nodes.values() if hasattr(n, 'role') and n.role == NodeRole.ATTACKER]
        honeypots = [n for n in self.protocol.nodes.values() if hasattr(n, 'role') and n.role == NodeRole.HONEYPOT]
        
        for attacker in attackers:
            targets = attacker.select_targets(self.protocol)
            attacker.attack_targets = targets
            print(f"🔴 Attacker-{attacker.id} identified {len(targets)} potential targets")
        
        # Phase 2: Initial attacks
        print("\n⚔️ Phase 2: Attack Initiation")
        for attacker in attackers:
            if attacker.attack_targets:
                target_id, target_type = random.choice(attacker.attack_targets)
                attacker.launch_dos_attack(target_id, target_type, self.protocol)
        
        # Phase 3: Honeypot attraction
        print("\n🍯 Phase 3: Honeypot Operations")
        for honeypot in honeypots:
            weakness_ads = honeypot.simulate_weakness()
            print(f"🍯 Honeypot-{honeypot.id} advertising: {', '.join(weakness_ads)}")
        
        # Phase 4: Attack escalation and detection
        print("\n🔥 Phase 4: Attack Escalation")
        for attacker in attackers:
            if random.random() < 0.7:  # 70% chance to escalate
                # Generate excessive traffic
                mec_targets = [mid for mid, _ in attacker.attack_targets if _ == "MEC"]
                if mec_targets:
                    attacker.generate_excessive_traffic(mec_targets[:2], self.protocol)
                
                # Try identity spoofing
                if random.random() < attacker.spoofing_capability:
                    normal_nodes = [n.id for n in self.protocol.nodes.values() 
                                   if hasattr(n, 'role') and n.role == NodeRole.NORMAL]
                    if normal_nodes:
                        target_identity = random.choice(normal_nodes)
                        attacker.spoof_identity(target_identity)
        
        # Phase 5: Honeypot analysis and protection
        print("\n🛡️ Phase 5: Defense and Analysis")
        for honeypot in honeypots:
            # Simulate attacks being attracted to honeypot
            for attacker in attackers:
                if random.random() < 0.6:  # 60% chance attacker targets honeypot
                    fake_attack = AttackPacket(
                        packet_id=f"honeypot_attack_{attacker.id}_{honeypot.id}",
                        source_id=attacker.id,
                        target_id=honeypot.id,
                        attack_type=random.choice(list(AttackType)),
                        payload_size=random.uniform(5, 50),
                        timestamp=time.time(),
                        frequency=random.uniform(10, 100)
                    )
                    honeypot.absorb_malicious_traffic(fake_attack)
            
            # Generate protection report
            threat_report = honeypot.protect_network(self.protocol)
            self.security_alerts.append(f"Honeypot-{honeypot.id}: {len(threat_report['high_threat_attackers'])} threats identified")
        
        # Phase 6: Network protection
        print("\n🔒 Phase 6: Network Security Response")
        for alert in self.security_alerts:
            print(f"🚨 ALERT: {alert}")
        
        print(f"\n✅ Security simulation complete")
        print(f"   Attackers deployed: {len(attackers)}")
        print(f"   Honeypots active: {len(honeypots)}")
        print(f"   Security alerts: {len(self.security_alerts)}")

def main():
    """Main security demonstration"""
    print("🛡️ ARPMEC SECURITY DEMONSTRATION")
    print("=" * 60)
    print("Features:")
    print("✓ Malicious nodes with DoS attacks")
    print("✓ Identity spoofing attacks")
    print("✓ Honeypot nodes for attack detection")
    print("✓ Traffic analysis and threat profiling")
    print("✓ Network protection mechanisms")
    print("=" * 60)
    
    # Create security demo
    security_demo = ARPMECSecurityDemo(
        num_nodes=20, 
        num_attackers=3, 
        num_honeypots=2
    )
    
    # Set up secure network
    clusters = security_demo.create_secure_network()
    
    # Run security simulation
    security_demo.simulate_security_scenario()
    
    print("\n🎯 Security Demo Complete!")
    print("This demonstrates:")
    print("• Attacker node behavior and target selection")
    print("• DoS attacks against CH and MEC servers") 
    print("• Identity spoofing capabilities")
    print("• Honeypot attraction and analysis")
    print("• Network protection and threat intelligence")

if __name__ == "__main__":
    main()
