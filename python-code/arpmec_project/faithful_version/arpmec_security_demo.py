#!/usr/bin/env python3
"""
ARPMEC SECURITY DEMO - DoS Attack and Honeypot Implementation
This file implements simplified security features including:
1. Malicious nodes performing DoS attacks only
2. DoS attacks targeting CH and MEC servers (critical infrastructure)
3. Honeypot implementation for traffic absorption and analysis
4. Basic attack detection and visualization

Based on the ARPMEC protocol with simplified security enhancements.
Focus: DoS attacks vs Honeypot protection
"""

import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from arpmec_faithful import (ARPMECProtocol, IARServer, InterClusterMessage,
                             MECServer, MECTask, Node, NodeState)
from matplotlib.colors import LinearSegmentedColormap


class AttackType(Enum):
    """Types of attacks that can be performed - SIMPLIFIED TO DOS ONLY"""
    DOS_FLOODING = "dos_flooding"

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
    
@dataclass
class SecurityEvent:
    """Visual representation of security events"""
    event_id: str
    event_type: str  # "attack", "honeypot_capture", "countermeasure", "detection"
    source_pos: Tuple[float, float]
    target_pos: Tuple[float, float]
    timestamp: float
    duration: float = 3.0  # Default duration
    color: str = 'white'
    intensity: float = 1.0  # 0.0 to 1.0
    animation_phase: float = 0.0  # Current animation progress
    active: bool = True
    description: str = ""

@dataclass
class AttackVisualization:
    """Visual attack packet with animation"""
    attack_id: str
    source_pos: Tuple[float, float]
    target_pos: Tuple[float, float]
    current_pos: Tuple[float, float]
    attack_type: AttackType
    progress: float = 0.0
    active: bool = True
    color: str = "#FF0000"
    size: float = 15.0
    trail_positions: List[Tuple[float, float]] = None
    pulse_phase: float = 0.0
    speed: float = 50.0  # Added missing field
    creation_time: float = 0.0  # Added missing field
    
    def __post_init__(self):
        if self.trail_positions is None:
            self.trail_positions = []
        if self.creation_time == 0.0:
            self.creation_time = time.time()

@dataclass  
class HoneypotVisualization:
    """Visual honeypot effects"""
    honeypot_id: int
    position: Tuple[float, float]
    attraction_radius: float
    captured_attacks: int = 0
    threat_level: str = "LOW"
    deception_active: bool = True
    bait_animations: List[str] = None
    glow_intensity: float = 0.5
    creation_time: float = 0.0  # Added missing field
    
    def __post_init__(self):
        if self.bait_animations is None:
            self.bait_animations = []
        if self.creation_time == 0.0:
            self.creation_time = time.time()

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

class SecurityMonitor:
    """Advanced security monitoring and intrusion detection system"""
    
    def __init__(self, monitor_id: str):
        self.monitor_id = monitor_id
        self.traffic_baseline = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        self.detection_algorithms = ["statistical", "behavioral", "signature"]
        self.alert_severity_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        self.blocked_nodes = set()
        self.suspicious_patterns = {}
        
    def analyze_traffic_pattern(self, node_id: int, packet_rate: float, packet_size: float) -> str:
        """Analyze traffic patterns for anomalies"""
        
        # Statistical anomaly detection
        if node_id not in self.traffic_baseline:
            self.traffic_baseline[node_id] = {
                'avg_rate': packet_rate,
                'avg_size': packet_size,
                'sample_count': 1
            }
            return "NORMAL"
        
        baseline = self.traffic_baseline[node_id]
        
        # Update baseline (exponential moving average)
        alpha = 0.1
        baseline['avg_rate'] = alpha * packet_rate + (1 - alpha) * baseline['avg_rate']
        baseline['avg_size'] = alpha * packet_size + (1 - alpha) * baseline['avg_size']
        baseline['sample_count'] += 1
        
        # Check for anomalies
        rate_deviation = abs(packet_rate - baseline['avg_rate']) / max(baseline['avg_rate'], 1.0)
        size_deviation = abs(packet_size - baseline['avg_size']) / max(baseline['avg_size'], 1.0)
        
        if rate_deviation > self.anomaly_threshold or size_deviation > self.anomaly_threshold:
            severity = "HIGH" if rate_deviation > 3.0 or size_deviation > 3.0 else "MEDIUM"
            return severity
        
        return "NORMAL"
    
    def detect_coordinated_attack(self, attack_sources: List[int]) -> bool:
        """Detect coordinated attacks from multiple sources"""
        
        if len(attack_sources) >= 3:  # 3+ sources attacking simultaneously
            return True
        
        # Check for time-synchronized attacks
        current_time = time.time()
        recent_attacks = [
            src for src in attack_sources 
            if src in self.suspicious_patterns and 
            current_time - self.suspicious_patterns[src].get('last_attack', 0) < 10
        ]
        
        return len(recent_attacks) >= 2
    
    def implement_countermeasures(self, attacker_id: int, attack_type: AttackType) -> str:
        """Implement automated countermeasures"""
        
        countermeasures = []
        
        if attack_type == AttackType.DOS_FLOODING:
            # Rate limiting
            countermeasures.append(f"Rate limiting applied to Node-{attacker_id}")
            # Traffic filtering
            countermeasures.append(f"DPI filtering activated for Node-{attacker_id}")
        
        # Temporary isolation
        self.blocked_nodes.add(attacker_id)
        countermeasures.append(f"Node-{attacker_id} temporarily isolated")
        
        return "; ".join(countermeasures)

class AdvancedHoneypot(HoneypotNode):
    """Enhanced honeypot with advanced deception and analysis"""
    
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 200.0):
        super().__init__(node_id, x, y, initial_energy)
        self.deception_techniques = []
        self.adaptive_behavior = True
        self.forensics_capability = True
        self.ai_powered_analysis = True
        
    def deploy_advanced_deception(self):
        """Deploy sophisticated deception techniques"""
        
        deception_methods = [
            "fake_service_banners",
            "simulated_vulnerabilities", 
            "decoy_files_and_databases",
            "fake_network_topology",
            "credential_traps",
            "behavioral_mimicry"
        ]
        
        self.deception_techniques = random.sample(deception_methods, 3)
        
        for technique in self.deception_techniques:
            if technique == "fake_service_banners":
                print(f"🍯 Honeypot-{self.id}: Advertising fake SSH service (vulnerable version)")
            elif technique == "simulated_vulnerabilities":
                print(f"🍯 Honeypot-{self.id}: Simulating buffer overflow vulnerability")
            elif technique == "decoy_files_and_databases":
                print(f"🍯 Honeypot-{self.id}: Creating decoy sensor databases")
            elif technique == "fake_network_topology":
                print(f"🍯 Honeypot-{self.id}: Broadcasting fake network topology")
            elif technique == "credential_traps":
                print(f"🍯 Honeypot-{self.id}: Setting up credential honey traps")
            elif technique == "behavioral_mimicry":
                print(f"🍯 Honeypot-{self.id}: Mimicking legitimate node behavior patterns")
    
    def perform_deep_packet_inspection(self, attack_packet: AttackPacket) -> Dict:
        """Perform detailed forensic analysis of attack packets"""
        
        forensic_data = {
            'packet_fingerprint': f"fp_{hash(attack_packet.packet_id) % 10000}",
            'payload_analysis': self._analyze_payload(attack_packet),
            'attack_sophistication': self._assess_sophistication(attack_packet),
            'origin_analysis': self._trace_origin(attack_packet),
            'malware_indicators': self._detect_malware_signatures(attack_packet)
        }
        
        return forensic_data
    
    def _analyze_payload(self, packet: AttackPacket) -> Dict:
        """Analyze packet payload for attack patterns"""
        
        patterns = {
            'injection_attempts': random.choice([True, False]),
            'shellcode_detected': packet.payload_size > 20,
            'encryption_used': random.choice([True, False]),
            'obfuscation_level': random.uniform(0.0, 1.0)
        }
        
        return patterns
    
    def _assess_sophistication(self, packet: AttackPacket) -> str:
        """Assess attack sophistication level"""
        
        sophistication_score = 0
        
        if packet.payload_size > 50:
            sophistication_score += 2
        if hasattr(packet, 'spoofed_identity') and packet.spoofed_identity:
            sophistication_score += 3
        if packet.frequency > 100:
            sophistication_score += 1
        
        if sophistication_score >= 5:
            return "ADVANCED_PERSISTENT_THREAT"
        elif sophistication_score >= 3:
            return "SKILLED_ATTACKER"
        elif sophistication_score >= 1:
            return "SCRIPT_KIDDIE"
        else:
            return "AMATEUR"
    
    def _trace_origin(self, packet: AttackPacket) -> Dict:
        """Trace attack origin with geolocation simulation"""
        
        # Simulate network forensics
        possible_origins = [
            "Internal_Network", "External_Botnet", "State_Actor", 
            "Cybercriminal_Group", "Individual_Hacker"
        ]
        
        return {
            'likely_origin': random.choice(possible_origins),
            'confidence_level': random.uniform(0.6, 0.95),
            'hop_count': random.randint(3, 15),
            'anonymization_detected': random.choice([True, False])
        }
    
    def _detect_malware_signatures(self, packet: AttackPacket) -> List[str]:
        """Detect known malware signatures"""
        
        signatures = [
            "Mirai_IoT_Botnet", "Stuxnet_Industrial", "Zeus_Banking_Trojan",
            "APT1_Signature", "Conficker_Variant", "Custom_RAT_Tool"
        ]
        
        detected = []
        for sig in signatures:
            if random.random() < 0.15:  # 15% chance to detect each signature
                detected.append(sig)
        
        return detected

class ARPMECAdvancedSecurityDemo:
    """Enhanced ARPMEC Security Demonstration with comprehensive attack and defense visualization"""
    
    def __init__(self, num_nodes: int = 25, num_attackers: int = 3, num_honeypots: int = 2, area_size: int = 1200):
        self.num_nodes = num_nodes
        self.num_attackers = num_attackers
        self.num_honeypots = num_honeypots  # Fixed typo
        self.area_size = area_size
        self.protocol = None
        
        # Security tracking
        self.active_attacks: List[AttackPacket] = []
        self.detected_attackers: List[int] = []
        self.security_alerts: List[str] = []
        self.attack_timeline: List[Tuple[float, str]] = []
        
        # Advanced security components
        self.security_monitor = SecurityMonitor("main_monitor")
        self.forensics_data: Dict[str, Dict] = {}
        self.threat_intelligence_db: Dict[int, Dict] = {}
        self.countermeasures_deployed: List[str] = []
        
        # VISUALIZATION COMPONENTS
        self.security_events: List[SecurityEvent] = []
        self.attack_visualizations: List[AttackVisualization] = []
        self.honeypot_visualizations: List[HoneypotVisualization] = []
        self.current_frame = 0
        self.animation_time = 0.0
        
        # Attack generation timing
        self.last_attack_time = 0.0
        self.attack_interval = 3.0  # Generate attacks every 3 seconds
        
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
            'dos_flooding': '#FF0000'
        }
    
    def create_secure_network(self):
        """Create network with normal nodes, attackers, and honeypots"""
        print("Creating ARPMEC network with advanced security features...")
        
        nodes = []
        node_id = 0
        
        # Create normal nodes
        cluster_centers = [(200, 200), (600, 200), (400, 600), (800, 800)]
        normal_nodes = self.num_nodes - self.num_attackers - self.num_honeypots  # Fixed typo
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
        
        # Create advanced honeypot nodes
        for i in range(self.num_honeypots):
            # Place honeypots strategically near cluster centers
            center = random.choice(cluster_centers)
            x = center[0] + random.uniform(-120, 120)
            y = center[1] + random.uniform(-120, 120)
            x = max(50, min(self.area_size - 50, x))
            y = max(50, min(self.area_size - 50, y))
            energy = random.uniform(150, 200)  # High energy for continuous operation
            
            honeypot = AdvancedHoneypot(node_id, x, y, energy)
            honeypot.mimic_important_node("CH" if i % 2 == 0 else "MEC")
            honeypot.deploy_advanced_deception()
            nodes.append(honeypot)
            node_id += 1
            print(f"🍯 ADVANCED HONEYPOT: Created honeypot node {honeypot.id} at ({x:.0f}, {y:.0f})")
        
        # Create protocol
        self.protocol = ARPMECProtocol(nodes, C=4, R=5, K=3)
        
        # Perform initial clustering
        clusters = self.protocol.clustering_algorithm()
        print(f"Created {len(clusters)} clusters with advanced security nodes")
        
        return clusters
    
    def run_comprehensive_security_scenario(self):
        """Run a comprehensive security scenario with advanced features"""
        print("\n🚨 COMPREHENSIVE SECURITY SCENARIO")
        print("=" * 60)
        
        # Phase 1: Network reconnaissance and baseline establishment
        print("\n📡 Phase 1: Network Reconnaissance & Baseline Establishment")
        attackers = [n for n in self.protocol.nodes.values() if hasattr(n, 'role') and n.role == NodeRole.ATTACKER]
        honeypots = [n for n in self.protocol.nodes.values() if hasattr(n, 'role') and n.role == NodeRole.HONEYPOT]
        
        for attacker in attackers:
            targets = attacker.select_targets(self.protocol)
            attacker.attack_targets = targets
            print(f"🔴 Attacker-{attacker.id} identified {len(targets)} potential targets")
            
            # Establish baseline traffic patterns
            for target_id, target_type in targets[:3]:  # Monitor top 3 targets
                baseline_rate = random.uniform(1, 5)
                baseline_size = random.uniform(0.5, 2.0)
                anomaly_status = self.security_monitor.analyze_traffic_pattern(
                    attacker.id, baseline_rate, baseline_size
                )
        
        # Phase 2: Coordinated attack initiation
        print("\n⚔️ Phase 2: Coordinated Attack Campaign")
        attack_wave_1 = []
        
        for attacker in attackers:
            if attacker.attack_targets:
                target_id, target_type = random.choice(attacker.attack_targets)
                attacker.launch_dos_attack(target_id, target_type, self.protocol)
                attack_wave_1.append(attacker.id)
        
        # Detect coordinated attacks
        coordinated_attack = self.security_monitor.detect_coordinated_attack(attack_wave_1)
        if coordinated_attack:
            print("🚨 ALERT: Coordinated attack detected from multiple sources!")
            self.security_alerts.append("Coordinated DDoS attack detected")
        
        # Phase 3: Advanced honeypot operations
        print("\n🍯 Phase 3: Advanced Honeypot Intelligence Gathering")
        for honeypot in honeypots:
            if isinstance(honeypot, AdvancedHoneypot):
                # Simulate sophisticated attacks being attracted to honeypot
                for attacker in attackers:
                    if random.random() < 0.8:  # 80% chance attacker targets honeypot
                        attack_packet = AttackPacket(
                            packet_id=f"advanced_attack_{attacker.id}_{honeypot.id}",
                            source_id=attacker.id,
                            target_id=honeypot.id,
                            attack_type=random.choice(list(AttackType)),
                            payload_size=random.uniform(10, 100),
                            timestamp=time.time(),
                            frequency=random.uniform(20, 200)
                        )
                        
                        if honeypot.absorb_malicious_traffic(attack_packet):
                            # Perform deep packet inspection
                            forensics = honeypot.perform_deep_packet_inspection(attack_packet)
                            self.forensics_data[attack_packet.packet_id] = forensics
                            
                            print(f"🔍 FORENSICS: Honeypot-{honeypot.id} analyzed packet from Attacker-{attacker.id}")
                            print(f"   Sophistication: {forensics['attack_sophistication']}")
                            print(f"   Origin: {forensics['origin_analysis']['likely_origin']}")
                            if forensics['malware_indicators']:
                                print(f"   Malware detected: {', '.join(forensics['malware_indicators'])}")
        
        # Phase 4: DoS Attack Escalation
        print("\n🔥 Phase 4: DoS Attack Escalation")
        for attacker in attackers:
            if random.random() < 0.6:  # 60% chance to escalate DoS attacks
                # Multi-target DoS attack
                mec_targets = [mid for mid, _ in attacker.attack_targets if _ == "MEC"]
                if mec_targets:
                    # Launch DoS against multiple MEC servers
                    for target_id in mec_targets[:2]:
                        attacker.launch_dos_attack(target_id, "MEC", self.protocol)
                        
                        # Implement countermeasures
                        countermeasures = self.security_monitor.implement_countermeasures(
                            attacker.id, AttackType.DOS_FLOODING
                        )
                        self.countermeasures_deployed.append(countermeasures)
                        print(f"🛡️ COUNTERMEASURES: {countermeasures}")
                
                # Maintain stealth operations
                attacker.maintain_stealth(self.protocol)
        
        # Phase 5: Threat intelligence and network protection
        print("\n🛡️ Phase 5: Threat Intelligence & Network Protection")
        threat_reports = []
        
        for honeypot in honeypots:
            if isinstance(honeypot, AdvancedHoneypot):
                threat_report = honeypot.protect_network(self.protocol)
                threat_reports.append(threat_report)
                
                # Update threat intelligence database
                for attacker_id in threat_report['high_threat_attackers']:
                    if attacker_id not in self.threat_intelligence_db:
                        self.threat_intelligence_db[attacker_id] = {
                            'threat_level': 'HIGH',
                            'attack_patterns': [],
                            'targeted_assets': [],
                            'first_detected': time.time()
                        }
                    
                    self.threat_intelligence_db[attacker_id]['attack_patterns'].extend(
                        threat_report['attack_patterns']
                    )
        
        # Phase 6: Final security assessment
        print("\n📊 Phase 6: Security Assessment Report")
        
        total_attacks_detected = sum(len(report['attack_patterns']) for report in threat_reports)
        total_attackers_profiled = len(self.threat_intelligence_db)
        total_forensic_samples = len(self.forensics_data)
        
        print(f"🔍 DETECTION SUMMARY:")
        print(f"   Total attacks detected: {total_attacks_detected}")
        print(f"   Unique attackers profiled: {total_attackers_profiled}")
        print(f"   Forensic samples collected: {total_forensic_samples}")
        print(f"   Countermeasures deployed: {len(self.countermeasures_deployed)}")
        print(f"   Security alerts generated: {len(self.security_alerts)}")
        
        # Show advanced threat analysis
        print(f"\n🎯 ADVANCED THREAT ANALYSIS:")
        for attacker_id, intel in self.threat_intelligence_db.items():
            unique_patterns = set(intel['attack_patterns'])
            print(f"   Attacker-{attacker_id}: {intel['threat_level']} threat, {len(unique_patterns)} attack patterns")
        
        # Show forensic insights
        print(f"\n🔬 FORENSIC INSIGHTS:")
        sophistication_levels = [data['attack_sophistication'] for data in self.forensics_data.values()]
        if sophistication_levels:
            apt_count = sophistication_levels.count('ADVANCED_PERSISTENT_THREAT')
            skilled_count = sophistication_levels.count('SKILLED_ATTACKER')
            script_count = sophistication_levels.count('SCRIPT_KIDDIE')
            
            print(f"   APT-level attacks: {apt_count}")
            print(f"   Skilled attacker patterns: {skilled_count}")
            print(f"   Script kiddie attempts: {script_count}")
    
    def create_attack_visualization(self, attack_packet: AttackPacket, attack_type: AttackType):
        """Create visualization for an attack packet"""
        # Find source and target positions
        source_node = self.protocol.nodes.get(attack_packet.source_id)
        target_node = self.protocol.nodes.get(attack_packet.target_id)
        
        if not source_node or not target_node:
            return
        
        attack_viz = AttackVisualization(
            attack_id=attack_packet.packet_id,
            attack_type=attack_type,
            source_pos=(source_node.x, source_node.y),
            target_pos=(target_node.x, target_node.y),
            current_pos=(source_node.x, source_node.y),
            trail_positions=[(source_node.x, source_node.y)],
            color=self.attack_colors.get(attack_type.value, '#FF0000'),
            size=150,
            speed=50.0,  # pixels per second
            active=True,
            pulse_phase=0.0,
            creation_time=time.time()
        )
        
        self.attack_visualizations.append(attack_viz)
        
        # Create security event for attack launch
        attack_event = SecurityEvent(
            event_id=f"attack_{len(self.security_events)}",
            event_type="attack_launch",
            timestamp=time.time(),
            source_pos=(source_node.x, source_node.y),
            target_pos=(target_node.x, target_node.y),
            description=f"🚨 {attack_type.value.upper()} attack from Node-{attack_packet.source_id} to Node-{attack_packet.target_id}",
            intensity=1.0,
            color='red',
            animation_phase=0.0
        )
        self.security_events.append(attack_event)
    
    def create_honeypot_visualization(self, honeypot_node):
        """Create visualization for a honeypot node"""
        honeypot_viz = HoneypotVisualization(
            honeypot_id=honeypot_node.id,
            position=(honeypot_node.x, honeypot_node.y),
            attraction_radius=150.0,
            glow_intensity=0.5,
            captured_attacks=0,
            deception_active=True,
            threat_level="LOW",
            creation_time=time.time()
        )
        
        self.honeypot_visualizations.append(honeypot_viz)
    
    def create_security_event(self, event_type: str, source_pos: Tuple[float, float], 
                            target_pos: Tuple[float, float] = None, description: str = "",
                            intensity: float = 1.0, color: str = 'white'):
        """Create a new security event for visualization"""
        event = SecurityEvent(
            event_id=f"event_{len(self.security_events)}",
            event_type=event_type,
            timestamp=time.time(),
            source_pos=source_pos,
            target_pos=target_pos or source_pos,
            description=description,
            intensity=intensity,
            color=color,
            animation_phase=0.0
        )
        self.security_events.append(event)
        return event
    
    def update_attack_visualizations(self, dt: float):
        """Update positions and states of attack visualizations"""
        current_time = time.time()
        
        for attack_viz in self.attack_visualizations[:]:  # Copy list to modify during iteration
            if not attack_viz.active:
                continue
            
            # Update pulse animation
            attack_viz.pulse_phase += dt * 10.0  # 10 radians per second
            
            # Calculate movement towards target
            dx = attack_viz.target_pos[0] - attack_viz.current_pos[0]
            dy = attack_viz.target_pos[1] - attack_viz.current_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > 5.0:  # Still moving
                # Normalize direction and apply speed
                move_x = (dx / distance) * attack_viz.speed * dt
                move_y = (dy / distance) * attack_viz.speed * dt
                
                # Update position
                new_x = attack_viz.current_pos[0] + move_x
                new_y = attack_viz.current_pos[1] + move_y
                attack_viz.current_pos = (new_x, new_y)
                
                # Add to trail (limit trail length)
                attack_viz.trail_positions.append(attack_viz.current_pos)
                if len(attack_viz.trail_positions) > 20:
                    attack_viz.trail_positions.pop(0)
            else:
                # Attack reached target - create impact event
                self.create_security_event(
                    "impact",
                    attack_viz.target_pos,
                    description=f"💥 {attack_viz.attack_type.value.upper()} impact at target",
                    intensity=1.0,
                    color='orange'
                )
                attack_viz.active = False
            
            # Remove old attacks (after 10 seconds)
            if current_time - attack_viz.creation_time > 10.0:
                attack_viz.active = False
        
        # Clean up inactive visualizations
        self.attack_visualizations = [av for av in self.attack_visualizations if av.active]
    
    def update_security_events(self, dt: float):
        """Update security event animations"""
        current_time = time.time()
        
        for event in self.security_events[:]:
            # Update animation phase
            event.animation_phase += dt * 2.0  # 2 radians per second
            
            # Fade out old events
            age = current_time - event.timestamp
            if age > 3.0:  # Events last 3 seconds
                event.intensity = max(0.0, 1.0 - (age - 3.0) / 2.0)  # Fade over 2 seconds
            
            # Remove very old events
            if age > 5.0:
                self.security_events.remove(event)
    
    def update_honeypot_visualizations(self, dt: float):
        """Update honeypot visualization states"""
        for hviz in self.honeypot_visualizations:
            # Update glow intensity based on nearby threats
            nearby_attacks = 0
            for attack_viz in self.attack_visualizations:
                if attack_viz.active:
                    distance = math.sqrt(
                        (attack_viz.current_pos[0] - hviz.position[0])**2 + 
                        (attack_viz.current_pos[1] - hviz.position[1])**2
                    )
                    if distance < hviz.attraction_radius:
                        nearby_attacks += 1
            
            # Increase glow with nearby threats
            target_glow = min(1.0, 0.3 + nearby_attacks * 0.2)
            hviz.glow_intensity = hviz.glow_intensity * 0.9 + target_glow * 0.1
            
            # Update threat level based on captured attacks
            if hviz.captured_attacks > 10:
                hviz.threat_level = "CRITICAL"
            elif hviz.captured_attacks > 5:
                hviz.threat_level = "HIGH"
            elif hviz.captured_attacks > 2:
                hviz.threat_level = "MEDIUM"
            else:
                hviz.threat_level = "LOW"
    
    def generate_live_attacks(self):
        """Generate realistic attacks during live demonstration"""
        current_time = time.time()
        
        # Generate attacks based on interval
        if current_time - self.last_attack_time > self.attack_interval:
            # Get potential attackers and targets
            attackers = [n for n in self.protocol.nodes.values() 
                        if hasattr(n, 'role') and n.role == NodeRole.ATTACKER and n.is_alive()]
            
            if not attackers:
                return
            
            # Choose random attacker
            attacker = random.choice(attackers)
            
            # Only DoS attacks
            attack_type = AttackType.DOS_FLOODING
            
            # Choose targets - prioritize CH and MEC servers
            potential_targets = []
            
            # Add cluster heads as high-priority targets
            cluster_heads = self.protocol._get_cluster_heads()
            potential_targets.extend(cluster_heads)
            
            # Add MEC servers as critical targets
            potential_targets.extend(list(self.protocol.mec_servers.values()))
            
            # Add some normal nodes as backup targets
            normal_nodes = [n for n in self.protocol.nodes.values() 
                           if n.is_alive() and n.id != attacker.id and 
                           hasattr(n, 'role') and n.role == NodeRole.NORMAL]
            potential_targets.extend(normal_nodes[:5])  # Add only first 5 normal nodes
            
            if not potential_targets:
                return
            
            target = random.choice(potential_targets)
            
            # Create attack packet
            attack_packet = AttackPacket(
                packet_id=f"attack_{current_time}_{attacker.id}",
                source_id=attacker.id,
                target_id=target.id,
                attack_type=attack_type,
                payload_size=random.randint(100, 1000),
                timestamp=current_time,
                spoofed_identity=None
            )
            
            # Create visualization
            self.create_attack_visualization(attack_packet, attack_type)
            
            # Check if attack hits honeypot
            honeypots = [n for n in self.protocol.nodes.values() 
                        if hasattr(n, 'role') and n.role == NodeRole.HONEYPOT]
            
            for honeypot in honeypots:
                if target.id == honeypot.id:
                    # Honeypot capture
                    for hviz in self.honeypot_visualizations:
                        if hviz.honeypot_id == honeypot.id:
                            hviz.captured_attacks += 1
                            self.create_security_event(
                                "honeypot_capture",
                                (honeypot.x, honeypot.y),
                                description=f"🍯 Honeypot captured {attack_type.value} attack",
                                intensity=1.0,
                                color='gold'
                            )
                            break
                    break
            
            # Generate countermeasures occasionally
            if random.random() < 0.3:  # 30% chance of immediate countermeasure
                self.create_security_event(
                    "countermeasure",
                    (target.x, target.y),
                    description=f"🛡️ Deploying countermeasures against {attack_type.value}",
                    intensity=0.8,
                    color='cyan'
                )
            
            self.last_attack_time = current_time
            
            # Vary attack interval (1-5 seconds)
            self.attack_interval = random.uniform(1.0, 5.0)
    
    def draw_security_visualization(self, ax, frame_time: float):
        """Draw complete security visualization"""
        
        self.animation_time = frame_time
        
        # Clear and set up the plot
        ax.clear()
        ax.set_xlim(-100, self.area_size + 200)
        ax.set_ylim(-100, self.area_size + 100)
        ax.set_facecolor('#0A0A0A')  # Dark background for security theme
        ax.grid(True, alpha=0.2, color='white')
        
        # Update animations
        self.update_attack_visualizations(0.1)
        self.update_security_events(0.1)
        self.update_honeypot_visualizations(0.1)
        self.generate_live_attacks()
        
        # 1. Draw network infrastructure with security overlay
        cluster_heads = self.protocol._get_cluster_heads()
        
        # Draw cluster boundaries with threat level coloring
        threat_colors = {'LOW': 'green', 'MEDIUM': 'yellow', 'HIGH': 'orange', 'CRITICAL': 'red'}
        for i, ch in enumerate(cluster_heads):
            # Determine cluster threat level
            cluster_threat = "LOW"
            for hviz in self.honeypot_visualizations:
                if math.sqrt((hviz.position[0] - ch.x)**2 + (hviz.position[1] - ch.y)**2) < 200:
                    cluster_threat = hviz.threat_level
                    break
            
            boundary_color = threat_colors.get(cluster_threat, 'green')
            boundary = plt.Circle((ch.x, ch.y), self.protocol.communication_range,
                                fill=False, color=boundary_color, alpha=0.4, linewidth=2)
            ax.add_patch(boundary)
        
        # 2. Draw nodes with security status
        for node in self.protocol.nodes.values():
            if not node.is_alive():
                continue
            
            node_color = self.security_colors['normal']
            node_size = 80
            marker = 'o'
            edge_color = 'white'
            edge_width = 1
            
            if hasattr(node, 'role'):
                if node.role == NodeRole.ATTACKER:
                    node_color = self.security_colors['attacker']
                    node_size = 120
                    marker = 'X'
                    edge_color = 'red'
                    edge_width = 3
                elif node.role == NodeRole.HONEYPOT:
                    node_color = self.security_colors['honeypot']
                    node_size = 100
                    marker = 'h'
                    edge_color = 'gold'
                    edge_width = 2
                    
                    # Add honeypot glow effect
                    for hviz in self.honeypot_visualizations:
                        if hviz.honeypot_id == node.id:
                            glow_circle = plt.Circle((node.x, node.y), 50 * hviz.glow_intensity,
                                                   fill=True, color='gold', alpha=0.3)
                            ax.add_patch(glow_circle)
            
            if node.state == NodeState.CLUSTER_HEAD:
                node_color = self.security_colors['cluster_head']
                marker = '^'
                node_size = 150
            
            ax.scatter(node.x, node.y, c=node_color, s=node_size, marker=marker,
                      edgecolors=edge_color, linewidth=edge_width, zorder=5)
            
            # Add node labels
            label_color = 'white' if hasattr(node, 'role') and node.role == NodeRole.ATTACKER else 'black'
            ax.text(node.x, node.y - 25, f'N-{node.id}', ha='center', fontsize=8, 
                   color=label_color, weight='bold')
        
        # 3. Draw infrastructure with security status
        for mec in self.protocol.mec_servers.values():
            load_pct = mec.get_load_percentage()
            
            # Color based on security status and load
            if load_pct > 90:
                mec_color = '#FF0000'  # Critical - under attack
                security_status = "UNDER ATTACK"
            elif load_pct > 70:
                mec_color = '#FF6600'  # Warning
                security_status = "HIGH LOAD"
            else:
                mec_color = '#0066FF'  # Normal
                security_status = "SECURE"
            
            ax.scatter(mec.x, mec.y, c=mec_color, s=400, marker='s', 
                      edgecolors='white', linewidth=2, zorder=6)
            ax.text(mec.x, mec.y - 40, f'MEC-{mec.id}', ha='center', 
                   fontsize=9, weight='bold', color='white')
            ax.text(mec.x, mec.y + 35, f'{load_pct:.0f}%\n{security_status}', 
                   ha='center', fontsize=7, color='white', weight='bold')
        
        for iar in self.protocol.iar_servers.values():
            ax.scatter(iar.x, iar.y, c='#8000FF', s=250, marker='D', 
                      edgecolors='white', linewidth=2, zorder=6)
            ax.text(iar.x, iar.y - 30, f'IAR-{iar.id}', ha='center', 
                   fontsize=8, weight='bold', color='white')
        
        # 4. Draw attack visualizations
        for attack_viz in self.attack_visualizations:
            if attack_viz.active:
                # Draw attack trail
                if len(attack_viz.trail_positions) > 1:
                    trail_x = [pos[0] for pos in attack_viz.trail_positions]
                    trail_y = [pos[1] for pos in attack_viz.trail_positions]
                    ax.plot(trail_x, trail_y, color=attack_viz.color, alpha=0.6, 
                           linewidth=3, linestyle='--')
                
                # Draw pulsing attack packet
                pulse_size = attack_viz.size * (1.0 + 0.5 * math.sin(attack_viz.pulse_phase))
                ax.scatter(attack_viz.current_pos[0], attack_viz.current_pos[1], 
                          c=attack_viz.color, s=pulse_size, marker='*', 
                          edgecolors='white', linewidth=2, zorder=100, alpha=0.9)
                
                # Add attack type label
                ax.text(attack_viz.current_pos[0], attack_viz.current_pos[1] + 25, 
                       attack_viz.attack_type.value.upper(), ha='center', fontsize=7, 
                       color='white', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor=attack_viz.color, alpha=0.8))
        
        # 5. Draw security events (impacts, countermeasures, etc.)
        for event in self.security_events:
            if event.event_type == "impact":
                # Draw explosion effect
                explosion_size = 200 * event.intensity
                explosion = plt.Circle(event.source_pos, explosion_size, 
                                     fill=True, color=event.color, alpha=0.4)
                ax.add_patch(explosion)
                
            elif event.event_type == "countermeasure":
                # Draw shield effect
                shield_size = 100 * event.intensity
                shield = plt.Circle(event.source_pos, shield_size, 
                                  fill=False, color=event.color, linewidth=3, alpha=0.7)
                ax.add_patch(shield)
                
                # Add rotating shield lines
                angle = event.animation_phase * 180
                for i in range(4):
                    line_angle = math.radians(angle + i * 90)
                    x1 = event.source_pos[0] + shield_size * 0.7 * math.cos(line_angle)
                    y1 = event.source_pos[1] + shield_size * 0.7 * math.sin(line_angle)
                    x2 = event.source_pos[0] + shield_size * 1.2 * math.cos(line_angle)
                    y2 = event.source_pos[1] + shield_size * 1.2 * math.sin(line_angle)
                    ax.plot([x1, x2], [y1, y2], color=event.color, linewidth=3, alpha=0.7)
            
            elif event.event_type == "honeypot_capture":
                # Draw honeypot capture effect
                capture_radius = 80 * (1.0 + event.animation_phase)
                capture_circle = plt.Circle(event.target_pos, capture_radius,
                                          fill=False, color=event.color, linewidth=2, 
                                          alpha=min(1.0, max(0.0, event.intensity)))
                ax.add_patch(capture_circle)
        
        # 6. Draw honeypot attraction zones
        for hviz in self.honeypot_visualizations:
            if hviz.deception_active:
                # Draw attraction zone
                attraction_alpha = 0.1 + 0.2 * hviz.glow_intensity
                attraction_zone = plt.Circle(hviz.position, hviz.attraction_radius,
                                           fill=True, color='gold', alpha=attraction_alpha)
                ax.add_patch(attraction_zone)
                
                # Draw threat level indicator
                threat_colors_viz = {'LOW': 'green', 'MEDIUM': 'yellow', 'HIGH': 'orange', 'CRITICAL': 'red'}
                threat_color = threat_colors_viz.get(hviz.threat_level, 'green')
                
                # Pulsing threat indicator
                pulse_radius = 20 + 10 * math.sin(frame_time * 3)
                threat_indicator = plt.Circle(hviz.position, pulse_radius,
                                            fill=True, color=threat_color, alpha=0.6)
                ax.add_patch(threat_indicator)
        
        # 7. Create comprehensive security status overlay
        self._draw_security_status_overlay(ax, frame_time)
        
        # 8. Set title and labels
        active_attacks = len(self.attack_visualizations)
        detected_threats = len([h for h in self.honeypot_visualizations if h.threat_level in ['HIGH', 'CRITICAL']])
        
        ax.set_title(f'🚨 ARPMEC SECURITY VISUALIZATION 🚨\n'
                    f'Active Attacks: {active_attacks} | Threats Detected: {detected_threats} | Time: {frame_time:.1f}s',
                    fontsize=14, weight='bold', color='white', pad=20)
        ax.set_xlabel('X Position (meters)', fontsize=10, color='white')
        ax.set_ylabel('Y Position (meters)', fontsize=10, color='white')
    
    def _draw_security_status_overlay(self, ax, frame_time: float):
        """Draw security status information overlay"""
        
        # Security statistics
        total_attacks = len(self.attack_visualizations)
        honeypot_captures = sum(h.captured_attacks for h in self.honeypot_visualizations)
        high_threats = len([h for h in self.honeypot_visualizations if h.threat_level in ['HIGH', 'CRITICAL']])
        countermeasures_active = len([e for e in self.security_events if e.event_type == "countermeasure"])
        
        # Left panel - Attack Statistics
        attack_stats = f"🔴 ATTACK STATISTICS\n"
        attack_stats += f"Active Attacks: {total_attacks}\n"
        attack_stats += f"Total Events: {len(self.security_events)}\n"
        attack_stats += f"High Threats: {high_threats}"
        
        ax.text(0.02, 0.98, attack_stats, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='red', alpha=0.8),
                color='white', weight='bold', zorder=200)
        
        # Right panel - Defense Status
        defense_stats = f"🛡️ DEFENSE STATUS\n"
        defense_stats += f"Honeypot Captures: {honeypot_captures}\n"
        defense_stats += f"Active Countermeasures: {countermeasures_active}\n"
        defense_stats += f"Network Status: {'UNDER ATTACK' if total_attacks > 3 else 'SECURE'}"
        
        ax.text(0.98, 0.98, defense_stats, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='green', alpha=0.8),
                color='white', weight='bold', zorder=200)
        
        # Bottom panel - Recent Events
        recent_events = [e for e in self.security_events if frame_time - e.timestamp < 5.0]
        if recent_events:
            events_text = "📡 RECENT SECURITY EVENTS:\n"
            for event in recent_events[-3:]:  # Show last 3 events
                events_text += f"• {event.description}\n"
            
            ax.text(0.5, 0.02, events_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='navy', alpha=0.9),
                    color='white', weight='bold', zorder=200)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                      markersize=8, label='Cluster Head'),
            plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', 
                      markersize=8, label='Attacker Node'),
            plt.Line2D([0], [0], marker='h', color='w', markerfacecolor='gold', 
                      markersize=8, label='Honeypot'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
                      markersize=8, label='MEC Server'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                      markersize=10, label='Active Attack'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=8, label='Countermeasure')
        ]
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5),
                 fontsize=9, fancybox=True, shadow=True, framealpha=0.9)
    
    def create_security_animation(self, filename: str = "arpmec_security_demo.mp4", duration: int = 60):
        """Create animated security demonstration"""
        print(f"Creating ARPMEC Security Animation...")
        
        # Initialize honeypot visualizations
        honeypots = [n for n in self.protocol.nodes.values() 
                    if hasattr(n, 'role') and n.role == NodeRole.HONEYPOT]
        for honeypot in honeypots:
            self.create_honeypot_visualization(honeypot)
        
        # Create figure with dark theme
        fig, ax = plt.subplots(figsize=(20, 14))
        fig.patch.set_facecolor('black')
        
        def animate(frame):
            frame_time = frame * 0.2  # 5 FPS effective animation speed
            self.draw_security_visualization(ax, frame_time)
            return []
        
        # Calculate frame count
        fps = 10  # 10 FPS for smooth security visualization
        total_frames = duration * fps
        
        print(f"Generating {total_frames} frames for {duration}s security demo...")
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                     interval=100, blit=False, repeat=True)
        
        # Save animation
        print(f"Saving security animation to {filename}...")
        anim.save(filename, writer='ffmpeg', fps=fps, 
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        
        print(f"✅ Security animation saved successfully!")
        plt.close()
        
        return anim
    
    def run_security_tests(self):
        """Run security tests without visualization"""
        print("🧪 Running comprehensive security tests...")
        
        # Run the comprehensive security scenario
        self.run_comprehensive_security_scenario()
        
        print("\n✅ Security tests completed successfully!")
        print("Key metrics:")
        print(f"  - Total attackers: {self.num_attackers}")
        print(f"  - Total honeypots: {self.num_honeypots}")
        print(f"  - Security alerts generated: {len(self.security_alerts)}")
        print(f"  - Threat intelligence entries: {len(self.threat_intelligence_db)}")
        print(f"  - Forensic samples collected: {len(self.forensics_data)}")
        print(f"  - Countermeasures deployed: {len(self.countermeasures_deployed)}")

    def show_live_security_demo(self):
        """Show live interactive security demonstration"""
        print("🚨 Starting LIVE Security Demonstration...")
        print("Press Ctrl+C to stop the demo")
        
        # Initialize honeypot visualizations
        honeypots = [n for n in self.protocol.nodes.values() 
                    if hasattr(n, 'role') and n.role == NodeRole.HONEYPOT]
        for honeypot in honeypots:
            self.create_honeypot_visualization(honeypot)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(20, 14))
        fig.patch.set_facecolor('black')
        
        def animate(frame):
            frame_time = frame * 0.2
            self.draw_security_visualization(ax, frame_time)
            return []
        
        # Create live animation
        anim = animation.FuncAnimation(fig, animate, interval=200, blit=False, repeat=True)
        
        plt.tight_layout()
        plt.show()
        
        return anim

def main():
    """Main function to run the security demonstration"""
    print("🚨 ARPMEC Advanced Security Demonstration 🚨")
    print("=" * 50)
    
    # Create demo instance
    demo = ARPMECAdvancedSecurityDemo(
        num_nodes=20,
        num_attackers=4,
        num_honeypots=3,
        area_size=1000
    )
    
    # Create network
    demo.create_secure_network()
    
    print("\nSelect demonstration mode:")
    print("1. Live Interactive Demo (real-time visualization)")
    print("2. Generate Security Animation Video")
    print("3. Run Security Tests (no visualization)")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🎬 Starting Live Security Demo...")
            print("Watch for:")
            print("- Red X's: Attacker nodes launching attacks")
            print("- Gold hexagons: Honeypots attracting and capturing attacks")
            print("- Pulsing attack packets moving across the network")
            print("- Impact explosions when attacks hit targets")
            print("- Shield effects showing active countermeasures")
            print("- Real-time security statistics in corner panels")
            print("\nPress Ctrl+C to stop...")
            
            demo.show_live_security_demo()
            
        elif choice == "2":
            duration = 60  # 60 second video
            filename = "arpmec_security_demo.mp4"
            print(f"\n🎥 Creating {duration}s security animation video...")
            print("This will show a complete security scenario with:")
            print("- Multiple coordinated attacks")
            print("- Honeypot deception and capture")
            print("- Automated countermeasures")
            print("- Real-time threat analysis")
            
            demo.create_security_animation(filename, duration)
            print(f"✅ Video saved as {filename}")
            
        elif choice == "3":
            print("\n🧪 Running Security Tests...")
            demo.run_security_tests()
            print("✅ Security tests completed!")
            
        else:
            print("Invalid choice. Running live demo by default...")
            demo.show_live_security_demo()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have all required packages: matplotlib, numpy")

if __name__ == "__main__":
    main()
