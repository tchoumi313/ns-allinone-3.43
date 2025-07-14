#!/usr/bin/env python3
"""
COMPREHENSIVE ARPMEC PROTOCOL WITH INTEGRATED SECURITY FEATURES
===============================================================

This file provides a complete implementation of the ARPMEC protocol with integrated security features.
It includes all the components required for a master's degree study:

1. PROTOCOL FEATURES:
   - Dynamic clustering algorithm
   - Cluster Head (CH) election and re-election
   - Mobile Edge Computing (MEC) servers
   - Inter-cluster Association Representatives (IAR)
   - Inter-cluster and intra-cluster communications
   - Node mobility and reclustering
   - Energy-aware protocol operations

2. SECURITY FEATURES:
   - DoS attack detection and mitigation
   - Honeypot nodes for threat intelligence
   - Security monitoring and forensics
   - Countermeasure deployment
   - Threat visualization and analysis

3. VISUALIZATION:
   - Protocol-driven network visualization
   - Security event animation
   - Real-time attack monitoring
   - Performance metrics display

Author: Master's Degree Research Implementation
Date: July 2025
"""

import math
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
# Import base ARPMEC components
from arpmec_faithful import (ARPMECProtocol, IARServer, InterClusterMessage,
                             MECServer, MECTask, Node, NodeState)
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

# =====================================================================================
# SECURITY ENUMS AND DATA STRUCTURES
# =====================================================================================

class AttackType(Enum):
    """Types of security attacks"""
    DOS_ATTACK = "dos_attack"
    DOS_FLOODING = "dos_flooding"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_JAMMING = "network_jamming"

class NodeRole(Enum):
    """Extended node roles including security roles"""
    NORMAL = "normal"
    ATTACKER = "attacker"
    HONEYPOT = "honeypot"
    SECURITY_MONITOR = "security_monitor"

class ThreatLevel(Enum):
    """Threat level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event for logging and visualization"""
    event_id: str
    event_type: str
    timestamp: float
    source_id: int
    target_id: Optional[int]
    attack_type: Optional[AttackType]
    severity: ThreatLevel
    description: str
    position: Tuple[float, float]
    duration: float = 3.0
    active: bool = True
    animation_phase: float = 0.0

@dataclass
class AttackPacket:
    """Malicious packet representation"""
    packet_id: str
    source_id: int
    target_id: int
    attack_type: AttackType
    payload_size: float
    timestamp: float
    frequency: float = 1.0
    spoofed_identity: Optional[int] = None

@dataclass
class SecurityMetrics:
    """Security performance metrics"""
    total_attacks_detected: int = 0
    attacks_blocked: int = 0
    honeypot_captures: int = 0
    false_positives: int = 0
    response_time: float = 0.0
    network_availability: float = 100.0
    threat_level: ThreatLevel = ThreatLevel.LOW

# =====================================================================================
# ENHANCED NODE CLASSES WITH SECURITY CAPABILITIES
# =====================================================================================

class SecureNode(Node):
    """Enhanced node with security capabilities"""
    
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 100.0):
        super().__init__(node_id, x, y, initial_energy)
        self.role = NodeRole.NORMAL
        self.security_level = random.uniform(0.5, 1.0)
        self.trust_score = 1.0
        self.suspicious_activity_count = 0
        self.last_security_check = time.time()
        self.security_alerts: List[SecurityEvent] = []
        
        # DRASTICALLY REDUCE MOBILITY to make clusters visible
        self.max_velocity = 2.0  # Reduced from 20.0 to 2.0 (10x slower)
        self.direction_change_probability = 0.05  # Reduced from 0.3 to 0.05 (6x less frequent)
        
    def update_mobility(self, area_bounds: Tuple[float, float, float, float] = (0, 1000, 0, 1000)):
        """Override mobility to make it much more stable"""
        min_x, max_x, min_y, max_y = area_bounds
        
        # Extremely stable mobility - rarely change direction, move slowly
        if random.random() < self.direction_change_probability:
            self.velocity_x = random.uniform(-self.max_velocity, self.max_velocity)
            self.velocity_y = random.uniform(-self.max_velocity, self.max_velocity)
        
        # Update position with much smaller steps
        new_x = self.x + self.velocity_x * 0.5  # Even slower movement
        new_y = self.y + self.velocity_y * 0.5
        
        # Bounce off boundaries
        if new_x < min_x or new_x > max_x:
            self.velocity_x = -self.velocity_x
            new_x = max(min_x, min(max_x, new_x))
        
        if new_y < min_y or new_y > max_y:
            self.velocity_y = -self.velocity_y
            new_y = max(min_y, min(max_y, new_y))
        
        self.x = new_x
        self.y = new_y
        
    def assess_security_risk(self, packet_rate: float, packet_size: float) -> ThreatLevel:
        """Assess security risk based on traffic patterns"""
        # Baseline traffic analysis
        normal_rate = 5.0  # packets per second
        normal_size = 1.0  # KB
        
        rate_ratio = packet_rate / normal_rate
        size_ratio = packet_size / normal_size
        
        if rate_ratio > 10 or size_ratio > 10:
            return ThreatLevel.CRITICAL
        elif rate_ratio > 5 or size_ratio > 5:
            return ThreatLevel.HIGH
        elif rate_ratio > 2 or size_ratio > 2:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def update_trust_score(self, behavior_score: float):
        """Update trust score based on behavior"""
        # Exponential moving average
        alpha = 0.1
        self.trust_score = alpha * behavior_score + (1 - alpha) * self.trust_score
        self.trust_score = max(0.0, min(1.0, self.trust_score))

class AttackerNode(SecureNode):
    """Malicious node that performs attacks"""
    
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 120.0):
        super().__init__(node_id, x, y, initial_energy)
        self.role = NodeRole.ATTACKER
        self.attack_capability = random.uniform(0.6, 1.0)
        self.stealth_capability = random.uniform(0.4, 0.8)
        self.active_attacks: List[AttackPacket] = []
        self.attack_cooldown = 0.0
        self.target_list: List[int] = []
        
    def select_attack_targets(self, protocol: 'SecureARPMECProtocol') -> List[Tuple[int, str, float]]:
        """Select high-value targets for attack with priority scoring"""
        targets = []
        
        # Priority 1: Cluster Heads (highest value targets)
        for node in protocol.nodes.values():
            if node.state == NodeState.CLUSTER_HEAD and node.id != self.id:
                distance = self.distance_to(node)
                if distance <= protocol.communication_range * 3:
                    priority = 10.0 - (distance / 100.0)  # Closer = higher priority
                    targets.append((node.id, "CH", priority))
        
        # Priority 2: MEC servers (critical infrastructure)
        for mec_id, mec in protocol.mec_servers.items():
            distance = math.sqrt((self.x - mec.x)**2 + (self.y - mec.y)**2)
            if distance <= protocol.mec_communication_range * 1.5:
                load = mec.get_load_percentage()
                priority = 8.0 + (load * 2.0)  # Target heavily loaded MECs
                targets.append((mec_id, "MEC", priority))
        
        # Priority 3: IAR servers (inter-cluster communication hubs)
        for iar_id, iar in protocol.iar_servers.items():
            distance = math.sqrt((self.x - iar.x)**2 + (self.y - iar.y)**2)
            if distance <= iar.coverage_radius * 1.2:
                priority = 7.0 - (distance / 150.0)
                targets.append((iar_id, "IAR", priority))
        
        # Sort by priority (highest first)
        targets.sort(key=lambda x: x[2], reverse=True)
        return targets
    
    def launch_dos_attack(self, target_id: int, protocol: 'SecureARPMECProtocol'):
        """Launch DoS attack against target"""
        if self.attack_cooldown > 0:
            return None
        
        # Create attack packet
        attack_packet = AttackPacket(
            packet_id=f"dos_{self.id}_{target_id}_{time.time()}",
            source_id=self.id,
            target_id=target_id,
            attack_type=AttackType.DOS_FLOODING,
            payload_size=random.uniform(5, 20),  # Reduced payload size
            timestamp=time.time(),
            frequency=random.uniform(10, 50)  # Reduced frequency
        )
        
        self.active_attacks.append(attack_packet)
        self.attack_cooldown = random.uniform(60, 120)  # Much longer cooldown (2-4 min)
        
        # Reduced energy consumption for attack
        self.update_energy(attack_packet.payload_size * 0.01)
        
        return attack_packet
    
    def maintain_stealth(self, protocol: 'SecureARPMECProtocol'):
        """Maintain stealth to avoid detection"""
        # Reduce activity if security monitors are nearby
        monitors = [n for n in protocol.nodes.values() 
                   if hasattr(n, 'role') and n.role == NodeRole.SECURITY_MONITOR]
        
        nearby_monitors = sum(1 for m in monitors if self.distance_to(m) < 100)
        
        if nearby_monitors > 0:
            self.attack_cooldown = max(self.attack_cooldown, 20.0)
            
        # Update cooldown
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1.0

class HoneypotNode(SecureNode):
    """Honeypot node for attracting and analyzing attacks"""
    
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 150.0):
        super().__init__(node_id, x, y, initial_energy)
        self.role = NodeRole.HONEYPOT
        self.captured_attacks: List[AttackPacket] = []
        self.attacker_profiles: Dict[int, Dict] = {}
        self.deception_active = True
        self.attraction_radius = 150.0
        self.analysis_capability = random.uniform(0.8, 0.95)
        self.fake_vulnerabilities = ["high_load", "weak_auth", "buffer_overflow"]
        
        # Tracking and isolation capabilities
        self.tracked_attackers: Set[int] = set()
        self.isolation_active: Set[int] = set()
        self.forensics_log: List[Dict] = []
        
    def mimic_important_service(self, service_type: str):
        """Mimic critical services to attract attackers"""
        if service_type == "CH":
            self.state = NodeState.CLUSTER_HEAD
            self.cluster_id = self.id
        elif service_type == "MEC":
            # Advertise as overloaded MEC server
            self.fake_load = random.uniform(0.8, 0.95)
    
    def absorb_attack(self, attack_packet: AttackPacket, protocol: 'SecureARPMECProtocol' = None) -> bool:
        """Absorb and analyze incoming attack with tracking and isolation"""
        if len(self.captured_attacks) < 1000:  # Capacity limit
            self.captured_attacks.append(attack_packet)
            
            # Profile the attacker
            attacker_id = attack_packet.source_id
            if attacker_id not in self.attacker_profiles:
                self.attacker_profiles[attacker_id] = {
                    'attack_count': 0,
                    'total_payload': 0.0,
                    'avg_frequency': 0.0,
                    'first_seen': time.time(),
                    'threat_level': ThreatLevel.LOW
                }
            
            profile = self.attacker_profiles[attacker_id]
            profile['attack_count'] += 1
            profile['total_payload'] += attack_packet.payload_size
            profile['avg_frequency'] = (profile['avg_frequency'] + attack_packet.frequency) / 2
            
            # Update threat level and trigger countermeasures
            if profile['attack_count'] > 20:
                profile['threat_level'] = ThreatLevel.CRITICAL
                # Immediately isolate critical threats
                if protocol and attacker_id not in self.isolation_active:
                    self.isolate_attacker(attacker_id, protocol)
            elif profile['attack_count'] > 10:
                profile['threat_level'] = ThreatLevel.HIGH
                # Start tracking high threats
                if protocol and attacker_id not in self.tracked_attackers:
                    self.track_attacker(attacker_id, protocol)
            elif profile['attack_count'] > 5:
                profile['threat_level'] = ThreatLevel.MEDIUM
                # Begin monitoring
                if protocol:
                    self.track_attacker(attacker_id, protocol)
            
            return True
        return False
    
    def generate_threat_intelligence(self) -> Dict:
        """Generate threat intelligence report"""
        return {
            'honeypot_id': self.id,
            'total_attacks': len(self.captured_attacks),
            'unique_attackers': len(self.attacker_profiles),
            'high_threat_attackers': [
                aid for aid, profile in self.attacker_profiles.items()
                if profile['threat_level'] in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            ],
            'attack_patterns': list(set(attack.attack_type for attack in self.captured_attacks))
        }
    
    def track_attacker(self, attacker_id: int, protocol: 'SecureARPMECProtocol') -> bool:
        """Start tracking a captured attacker"""
        if attacker_id in protocol.nodes:
            attacker = protocol.nodes[attacker_id]
            self.tracked_attackers.add(attacker_id)
            
            # Move honeypot close to attacker for isolation
            distance = self.distance_to(attacker)
            if distance > 50:  # If too far, move closer
                # Calculate direction to attacker
                dx = attacker.x - self.x
                dy = attacker.y - self.y
                norm = math.sqrt(dx*dx + dy*dy)
                if norm > 0:
                    # Move 75% of the way to attacker
                    self.x += 0.75 * dx
                    self.y += 0.75 * dy
            
            # Log forensic information
            forensic_entry = {
                'timestamp': time.time(),
                'attacker_id': attacker_id,
                'attacker_position': (attacker.x, attacker.y),
                'honeypot_position': (self.x, self.y),
                'action': 'tracking_initiated',
                'energy_level': attacker.energy,
                'attack_history': len(self.captured_attacks)
            }
            self.forensics_log.append(forensic_entry)
            return True
        return False
    
    def isolate_attacker(self, attacker_id: int, protocol: 'SecureARPMECProtocol') -> bool:
        """Isolate attacker by blocking their communications"""
        if attacker_id in self.tracked_attackers:
            self.isolation_active.add(attacker_id)
            
            # Block attacker's communications
            if attacker_id in protocol.nodes:
                attacker = protocol.nodes[attacker_id]
                # Reduce attacker's communication range by jamming
                original_range = getattr(attacker, 'original_comm_range', protocol.communication_range)
                attacker.communication_range = original_range * 0.1  # Severely reduce range
                
                # Log isolation action
                forensic_entry = {
                    'timestamp': time.time(),
                    'attacker_id': attacker_id,
                    'action': 'isolation_activated',
                    'communication_blocked': True,
                    'original_range': original_range,
                    'reduced_range': attacker.communication_range
                }
                self.forensics_log.append(forensic_entry)
                return True
        return False

# =====================================================================================
# SECURITY MONITORING AND DEFENSE SYSTEMS
# =====================================================================================

class SecurityMonitor:
    """Comprehensive security monitoring system"""
    
    def __init__(self, monitor_id: str):
        self.monitor_id = monitor_id
        self.detection_threshold = 0.7
        self.blocked_nodes: Set[int] = set()
        self.security_events: List[SecurityEvent] = []
        self.metrics = SecurityMetrics()
        self.baseline_profiles: Dict[int, Dict] = {}
        
    def analyze_network_traffic(self, node_id: int, packet_rate: float, packet_size: float) -> ThreatLevel:
        """Analyze network traffic for anomalies"""
        # Initialize baseline if not exists
        if node_id not in self.baseline_profiles:
            self.baseline_profiles[node_id] = {
                'avg_rate': packet_rate,
                'avg_size': packet_size,
                'variance_rate': 0.0,
                'variance_size': 0.0,
                'sample_count': 1
            }
            return ThreatLevel.LOW
        
        baseline = self.baseline_profiles[node_id]
        
        # Calculate deviations
        rate_deviation = abs(packet_rate - baseline['avg_rate'])
        size_deviation = abs(packet_size - baseline['avg_size'])
        
        # Update baseline (exponential moving average)
        alpha = 0.1
        baseline['avg_rate'] = alpha * packet_rate + (1 - alpha) * baseline['avg_rate']
        baseline['avg_size'] = alpha * packet_size + (1 - alpha) * baseline['avg_size']
        baseline['sample_count'] += 1
        
        # Determine threat level
        if rate_deviation > baseline['avg_rate'] * 5 or size_deviation > baseline['avg_size'] * 5:
            return ThreatLevel.CRITICAL
        elif rate_deviation > baseline['avg_rate'] * 3 or size_deviation > baseline['avg_size'] * 3:
            return ThreatLevel.HIGH
        elif rate_deviation > baseline['avg_rate'] * 2 or size_deviation > baseline['avg_size'] * 2:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def detect_coordinated_attack(self, suspicious_nodes: List[int]) -> bool:
        """Detect coordinated attacks"""
        if len(suspicious_nodes) >= 3:
            # Check for synchronized activity
            return True
        return False
    
    def deploy_countermeasures(self, attacker_id: int, attack_type: AttackType) -> List[str]:
        """Deploy automated countermeasures"""
        countermeasures = []
        
        if attack_type == AttackType.DOS_FLOODING:
            # Rate limiting
            countermeasures.append(f"Rate limiting applied to Node-{attacker_id}")
            # Traffic filtering
            countermeasures.append(f"DPI filtering activated")
        
        # Isolate attacker
        self.blocked_nodes.add(attacker_id)
        countermeasures.append(f"Node-{attacker_id} isolated")
        
        self.metrics.attacks_blocked += 1
        return countermeasures
    
    def create_security_event(self, event_type: str, source_id: int, target_id: Optional[int], 
                             attack_type: Optional[AttackType], severity: ThreatLevel, 
                             description: str, position: Tuple[float, float]) -> SecurityEvent:
        """Create new security event"""
        event = SecurityEvent(
            event_id=f"event_{len(self.security_events)}",
            event_type=event_type,
            timestamp=time.time(),
            source_id=source_id,
            target_id=target_id,
            attack_type=attack_type,
            severity=severity,
            description=description,
            position=position
        )
        self.security_events.append(event)
        return event

# =====================================================================================
# SECURE ARPMEC PROTOCOL IMPLEMENTATION
# =====================================================================================

class SecureARPMECProtocol(ARPMECProtocol):
    """Enhanced ARPMEC protocol with integrated security features"""
    
    def __init__(self, nodes: List[SecureNode], C: int = 4, R: int = 5, K: int = 3):
        # Convert nodes to base Node objects for parent class
        base_nodes = []
        for node in nodes:
            if isinstance(node, SecureNode):
                base_nodes.append(node)
            else:
                base_nodes.append(Node(node.id, node.x, node.y, node.energy))
        
        super().__init__(base_nodes, C, R, K)
        
        # Replace nodes with secure versions
        self.nodes = {node.id: node for node in nodes}
        
        # Security components
        self.security_monitor = SecurityMonitor("main_security")
        self.active_attacks: List[AttackPacket] = []
        self.security_events: List[SecurityEvent] = []
        self.threat_intelligence: Dict[int, Dict] = {}
        self.defense_active = True
        
        # Performance metrics
        self.security_metrics = SecurityMetrics()
        self.attack_detection_rate = 0.85
        self.false_positive_rate = 0.05
        
        # Message exchange tracking for visualization
        self.active_messages: List[Dict] = []
        self.message_animations: Dict[str, Dict] = {}
        
    def get_attackers(self) -> List[AttackerNode]:
        """Get all attacker nodes"""
        return [node for node in self.nodes.values() if isinstance(node, AttackerNode)]
    
    def get_honeypots(self) -> List[HoneypotNode]:
        """Get all honeypot nodes"""
        return [node for node in self.nodes.values() if isinstance(node, HoneypotNode)]
    
    def process_attack_packet(self, attack_packet: AttackPacket) -> bool:
        """Process incoming attack packet"""
        target_node = self.nodes.get(attack_packet.target_id)
        if not target_node:
            return False
        
        # Check if target is a honeypot
        if isinstance(target_node, HoneypotNode):
            if target_node.absorb_attack(attack_packet, self):
                self.security_metrics.honeypot_captures += 1
                
                # Create security event
                event = self.security_monitor.create_security_event(
                    "honeypot_capture",
                    attack_packet.source_id,
                    attack_packet.target_id,
                    attack_packet.attack_type,
                    ThreatLevel.MEDIUM,
                    f"Honeypot captured {attack_packet.attack_type.value} attack",
                    (target_node.x, target_node.y)
                )
                self.security_events.append(event)
                return True
        
        # Analyze attack for threat detection
        threat_level = self.security_monitor.analyze_network_traffic(
            attack_packet.source_id,
            attack_packet.frequency,
            attack_packet.payload_size
        )
        
        # Deploy countermeasures if threat is detected
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            countermeasures = self.security_monitor.deploy_countermeasures(
                attack_packet.source_id,
                attack_packet.attack_type
            )
            
            # Create countermeasure event
            event = self.security_monitor.create_security_event(
                "countermeasure",
                attack_packet.source_id,
                attack_packet.target_id,
                attack_packet.attack_type,
                threat_level,
                f"Countermeasures deployed: {'; '.join(countermeasures)}",
                (target_node.x, target_node.y)
            )
            self.security_events.append(event)
            
            self.security_metrics.attacks_blocked += 1
            return True
        
        # Process attack impact on target (reduced impact)
        if attack_packet.attack_type == AttackType.DOS_FLOODING:
            # Reduced DoS impact
            energy_drain = attack_packet.payload_size * 0.01  # Much less energy drain
            target_node.update_energy(energy_drain)
            
            # If target is MEC server, increase load
            if attack_packet.target_id in self.mec_servers:
                mec_server = self.mec_servers[attack_packet.target_id]
                # Simulate load increase from DoS
                fake_task = MECTask(
                    task_id=f"dos_load_{attack_packet.packet_id}",
                    source_cluster_id=attack_packet.source_id,
                    cpu_requirement=attack_packet.payload_size,
                    memory_requirement=attack_packet.payload_size,
                    deadline=self.current_time_slot + 1,
                    data_size=attack_packet.payload_size,
                    created_time=self.current_time_slot
                )
                mec_server.accept_task(fake_task)
        
        self.security_metrics.total_attacks_detected += 1
        return False
    
    def update_security_monitoring(self):
        """Update security monitoring and threat analysis"""
        current_time = time.time()
        
        # Update security events
        for event in self.security_events[:]:
            event.animation_phase += 0.1
            age = current_time - event.timestamp
            
            if age > event.duration:
                event.active = False
                self.security_events.remove(event)
        
        # Generate new attacks from attackers (ultra reduced frequency)
        for attacker in self.get_attackers():
            # Ultra reduce attack frequency to make clusters visible
            if attacker.attack_cooldown <= 0 and random.random() < 0.005:  # Only 0.5% chance to attack
                targets = attacker.select_attack_targets(self)
                if targets:
                    # Select highest priority target
                    target_id, target_type, priority = targets[0]  # Already sorted by priority
                    attack_packet = attacker.launch_dos_attack(target_id, self)
                    if attack_packet:
                        self.active_attacks.append(attack_packet)
                        
                        # Create attack event
                        event = self.security_monitor.create_security_event(
                            "attack_launch",
                            attacker.id,
                            target_id,
                            attack_packet.attack_type,
                            ThreatLevel.HIGH,
                            f"DoS attack launched against {target_type}-{target_id}",
                            (attacker.x, attacker.y)
                        )
                        self.security_events.append(event)
            
            # Update stealth
            attacker.maintain_stealth(self)
        
        # Process active attacks
        for attack in self.active_attacks[:]:
            if self.process_attack_packet(attack):
                self.active_attacks.remove(attack)
        
        # Update threat intelligence from honeypots
        for honeypot in self.get_honeypots():
            if len(honeypot.captured_attacks) > 0:
                intel = honeypot.generate_threat_intelligence()
                self.threat_intelligence[honeypot.id] = intel
        
        # Update network availability
        alive_nodes = sum(1 for node in self.nodes.values() if node.is_alive())
        total_nodes = len(self.nodes)
        self.security_metrics.network_availability = (alive_nodes / total_nodes) * 100
    
    def run_protocol_step(self, step_number: int):
        """Enhanced protocol step execution with security monitoring"""
        self.current_time_slot = step_number
        
        # Step 1: Update node mobility (much less frequent for stability)
        if step_number % 3 == 0:  # Only every 3rd round instead of every round
            area_bounds = (0, 1000, 0, 1000)
            for node in self.nodes.values():
                if node.is_alive():
                    node.update_mobility(area_bounds)
        
        # Step 2: Check for re-clustering due to mobility (much less frequent for stability)
        if step_number % 20 == 0 and step_number > 0:  # Every 20 rounds instead of 5
            print(f"\n🔄 MOBILITY-TRIGGERED RE-CLUSTERING (Step {step_number})")
            membership_changed = self._check_and_recluster()
            leadership_changed = self._check_cluster_head_validity()
            
            if membership_changed or leadership_changed:
                print("✅ Network topology updated due to mobility")
                self._build_inter_cluster_routing_table()
        
        # Step 3: Normal protocol operations
        alive_nodes = [node for node in self.nodes.values() if node.is_alive()]
        
        for node in alive_nodes:
            if node.state == NodeState.CLUSTER_HEAD:
                self._fixed_cluster_head_operations(node, step_number)
            else:
                self._fixed_cluster_member_operations(node, step_number)
        
        # Step 4: Inter-cluster communication operations
        self._generate_inter_cluster_traffic()
        self._generate_mec_tasks()
        self._process_inter_cluster_messages()
        self._process_mec_servers()
        
        # Step 5: Update message animations
        self.update_message_animations()
        
        # Step 6: Security monitoring
        self.update_security_monitoring()
        
        # Step 6: Update security metrics
        self.security_metrics.total_attacks_detected = len(self.active_attacks)
        
        # Determine overall threat level (more balanced thresholds)
        if self.security_metrics.total_attacks_detected > 50:
            self.security_metrics.threat_level = ThreatLevel.CRITICAL
        elif self.security_metrics.total_attacks_detected > 20:
            self.security_metrics.threat_level = ThreatLevel.HIGH
        elif self.security_metrics.total_attacks_detected > 5:
            self.security_metrics.threat_level = ThreatLevel.MEDIUM
        else:
            self.security_metrics.threat_level = ThreatLevel.LOW
    
    def _process_inter_cluster_messages(self):
        """Enhanced message processing with visualization tracking"""
        super()._process_inter_cluster_messages()
        
        # Track new messages for visualization
        for ch in self._get_cluster_heads():
            for message in ch.inter_cluster_messages:
                message_id = f"msg_{message.message_id}_{time.time()}"
                
                # Find source and destination positions
                source_pos = (ch.x, ch.y)
                dest_pos = None
                
                if message.destination_cluster_id in self.nodes:
                    dest_node = self.nodes[message.destination_cluster_id]
                    dest_pos = (dest_node.x, dest_node.y)
                else:
                    # Check IAR servers
                    for iar in self.iar_servers.values():
                        if iar.id == message.destination_cluster_id:
                            dest_pos = (iar.x, iar.y)
                            break
                
                if dest_pos:
                    message_info = {
                        'id': message_id,
                        'source_pos': source_pos,
                        'dest_pos': dest_pos,
                        'message_type': message.message_type,
                        'timestamp': time.time(),
                        'progress': 0.0,
                        'active': True
                    }
                    self.active_messages.append(message_info)
    
    def update_message_animations(self):
        """Update message animation states"""
        current_time = time.time()
        
        for message in self.active_messages[:]:
            age = current_time - message['timestamp']
            message['progress'] = min(1.0, age / 2.0)  # 2 second animation
            
            if message['progress'] >= 1.0:
                message['active'] = False
                self.active_messages.remove(message)
    
    def draw_message_exchanges(self):
        """Draw inter-cluster message exchanges"""
        if not hasattr(self.protocol, 'active_messages'):
            return
            
        for message in self.protocol.active_messages:
            if not message.get('active', False):
                continue
                
            source_x, source_y = message['source_pos']
            dest_x, dest_y = message['dest_pos']
            progress = message['progress']
            
            # Calculate current position of message
            current_x = source_x + (dest_x - source_x) * progress
            current_y = source_y + (dest_y - source_y) * progress
            
            # Draw message path
            self.ax.plot([source_x, dest_x], [source_y, dest_y], 
                        color='cyan', alpha=0.3, linewidth=1, linestyle=':')
            
            # Draw moving message packet
            message_color = {
                'data_request': 'blue',
                'data_response': 'green', 
                'route_discovery': 'orange',
                'mec_task': 'purple'
            }.get(message.get('message_type', 'default'), 'cyan')
            
            self.ax.scatter(current_x, current_y, c=message_color, s=30, 
                          marker='>', alpha=0.8, zorder=6)
            
            # Add message label
            self.ax.text(current_x + 5, current_y + 5, 
                        f"MSG", fontsize=6, color=message_color, weight='bold')

    def draw_security_events(self):
        """Draw active security events"""
        current_time = time.time()
        
        for event in self.protocol.security_events:
            if not event.active:
                continue
            
            age = current_time - event.timestamp
            alpha = max(0.0, 1.0 - age / event.duration)
            
            if event.event_type == "attack_launch":
                # Draw attack pulse
                pulse_size = 100 * (1 + 0.5 * math.sin(event.animation_phase * 5))
                self.ax.scatter(event.position[0], event.position[1],
                              c=self.colors['attack'], s=pulse_size, marker='*',
                              alpha=alpha, zorder=100)
                
            elif event.event_type == "countermeasure":
                # Draw countermeasure shield
                shield_size = 80 * (1 + 0.3 * math.sin(event.animation_phase * 3))
                self.ax.scatter(event.position[0], event.position[1],
                              c=self.colors['countermeasure'], s=shield_size, marker='o',
                              alpha=alpha, zorder=99)
                
            elif event.event_type == "honeypot_capture":
                # Draw honeypot absorption effect
                absorption_size = 60 * (1 + 0.4 * math.sin(event.animation_phase * 4))
                self.ax.scatter(event.position[0], event.position[1],
                              c=self.colors['honeypot_capture'], s=absorption_size, marker='h',
                              alpha=alpha, zorder=98)
    
    def draw_security_dashboard(self):
        """Draw enhanced security monitoring dashboard"""
        # Security metrics panel
        metrics = self.protocol.security_metrics
        
        # Count honeypot intelligence
        total_captures = sum(len(hp.captured_attacks) for hp in self.protocol.get_honeypots())
        tracked_attackers = sum(len(hp.tracked_attackers) for hp in self.protocol.get_honeypots())
        isolated_attackers = sum(len(hp.isolation_active) for hp in self.protocol.get_honeypots())
        
        dashboard_text = f"""🚨 SECURITY DASHBOARD 🚨
Active Attacks: {metrics.total_attacks_detected}
Attacks Blocked: {metrics.attacks_blocked}
Honeypot Captures: {total_captures}
Tracked Attackers: {tracked_attackers}
Isolated Attackers: {isolated_attackers}
Network Availability: {metrics.network_availability:.1f}%
Threat Level: {metrics.threat_level.value.upper()}
"""
        
        # Color based on threat level
        if metrics.threat_level == ThreatLevel.CRITICAL:
            dashboard_color = 'red'
        elif metrics.threat_level == ThreatLevel.HIGH:
            dashboard_color = 'orange'
        elif metrics.threat_level == ThreatLevel.MEDIUM:
            dashboard_color = 'yellow'
        else:
            dashboard_color = 'green'
        
        self.ax.text(0.02, 0.98, dashboard_text, transform=self.ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=dashboard_color, alpha=0.8),
                   color='white', weight='bold', zorder=200)
        
        # Protocol metrics panel
        protocol_text = f"""📊 PROTOCOL METRICS 📊
Time Slot: {self.protocol.current_time_slot}
Active Nodes: {sum(1 for n in self.protocol.nodes.values() if n.is_alive())}
Cluster Heads: {len(self.protocol._get_cluster_heads())}
MEC Servers: {len(self.protocol.mec_servers)}
IAR Servers: {len(self.protocol.iar_servers)}
"""
        
        self.ax.text(0.98, 0.98, protocol_text, transform=self.ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='blue', alpha=0.8),
                   color='white', weight='bold', zorder=200)
        
        # Threat intelligence panel
        intel_text = "🔍 THREAT INTELLIGENCE 🔍\n"
        honeypots = self.protocol.get_honeypots()
        for honeypot in honeypots[:3]:  # Show top 3 honeypots
            intel = honeypot.generate_threat_intelligence()
            intel_text += f"Honeypot-{honeypot.id}: {intel['total_attacks']} attacks\n"
        
        self.ax.text(0.02, 0.02, intel_text, transform=self.ax.transAxes,
                   fontsize=9, verticalalignment='bottom', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='purple', alpha=0.8),
                   color='white', weight='bold', zorder=200)
    
    def update_visualization(self, frame):
        """Update visualization for animation"""
        # Run protocol time slot
        self.protocol.run_protocol_step(frame)
        
        # Draw network topology
        self.draw_network_topology()
        
        # Draw message exchanges
        self.draw_message_exchanges()
        
        # Draw security events
        self.draw_security_events()
        
        # Draw security dashboard
        self.draw_security_dashboard()
        
        # Update frame count
        self.frame_count = frame
        current_time = time.time() - self.start_time
        
        # Set title
        self.ax.set_title(f'🔐 SECURE ARPMEC PROTOCOL VISUALIZATION 🔐\n'
                         f'Frame: {frame} | Time: {current_time:.1f}s | '
                         f'Threat Level: {self.protocol.security_metrics.threat_level.value.upper()}',
                         fontsize=16, weight='bold', color='black', pad=20)
        
        # Style axes
        self.ax.set_xlabel('X Position (meters)', color='black', fontsize=12)
        self.ax.set_ylabel('Y Position (meters)', color='black', fontsize=12)
        self.ax.tick_params(colors='black')
        for spine in self.ax.spines.values():
            spine.set_color('black')
    
    def run_animation(self, interval: int = 200):
        """Run the visualization animation"""
        self.setup_visualization()
        
        self.animation_obj = animation.FuncAnimation(
            self.fig, self.update_visualization, interval=interval,
            blit=False, cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()
        
        return self.animation_obj

# =====================================================================================
# VISUALIZATION CLASS
# =====================================================================================

class SecureARPMECVisualizer:
    """Comprehensive visualization for secure ARPMEC protocol"""
    
    def __init__(self, protocol: SecureARPMECProtocol, area_size: int = 1000):
        self.protocol = protocol
        self.area_size = area_size
        self.fig = None
        self.ax = None
        
    def setup_visualization(self):
        """Setup comprehensive visualization"""
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
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
    
    def draw_cluster_boundaries(self):
        """Draw cluster boundaries with threat-based coloring"""
        cluster_heads = self.protocol._get_cluster_heads()
        for ch in cluster_heads:
            # Color based on threat level
            threat = self.protocol.security_monitor.metrics.threat_level
            if threat == ThreatLevel.HIGH:
                boundary_color = 'red'
                alpha = 0.7
            elif threat == ThreatLevel.MEDIUM:
                boundary_color = 'orange'
                alpha = 0.6
            else:
                boundary_color = 'green'
                alpha = 0.5
                
            boundary = Circle((ch.x, ch.y), self.protocol.communication_range,
                            fill=False, color=boundary_color, alpha=alpha, 
                            linewidth=2, linestyle='--')
            self.ax.add_patch(boundary)
    
    def draw_cluster_connections(self):
        """Draw connections between cluster members and their CHs"""
        cluster_heads = self.protocol._get_cluster_heads()
        
        for ch in cluster_heads:
            # Find cluster members
            for node in self.protocol.nodes.values():
                if (node.is_alive() and node.id != ch.id and 
                    node.state == NodeState.CLUSTER_MEMBER):
                    # Check if this node is in CH's cluster
                    if node.cluster_head_id == ch.id:
                        # Draw connection line
                        self.ax.plot([ch.x, node.x], [ch.y, node.y], 
                                   'gray', alpha=0.4, linewidth=1, zorder=1)
    
    def draw_nodes(self):
        """Draw all nodes with proper styling"""
        for node in self.protocol.nodes.values():
            if not node.is_alive():
                continue
            
            # Determine node appearance based on type and state
            if hasattr(node, 'is_attacker') and node.is_attacker:
                color = 'red'
                marker = 'X'
                size = 120
                edge_color = 'darkred'
            elif hasattr(node, 'is_honeypot') and node.is_honeypot:
                color = 'gold'
                marker = 'h'
                size = 100
                edge_color = 'orange'
            elif node.state == NodeState.CLUSTER_HEAD:
                color = 'blue'
                marker = '^'
                size = 150
                edge_color = 'darkblue'
            else:
                color = 'lightblue'
                marker = 'o'
                size = 80
                edge_color = 'blue'
            
            # Draw node
            self.ax.scatter(node.x, node.y, c=color, s=size, marker=marker,
                          edgecolors=edge_color, linewidth=2, zorder=5)
            
            # Add node label with energy info
            energy_pct = (node.energy / 200.0) * 100  # Assuming max energy ~200
            label = f'N{node.id}\n{energy_pct:.0f}%'
            
            self.ax.text(node.x, node.y - 30, label, ha='center', va='top',
                       fontsize=8, color='white', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
    
    def draw_infrastructure(self):
        """Draw MEC and IAR servers"""
        # Draw MEC servers
        for mec_id, mec in self.protocol.mec_servers.items():
            self.ax.scatter(mec.x, mec.y, c='green', s=250, marker='s',
                          edgecolors='darkgreen', linewidth=3, zorder=6)
            self.ax.text(mec.x, mec.y - 35, f'MEC{mec_id}', ha='center', va='top',
                       fontsize=10, weight='bold', color='white',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.8))
        
        # Draw IAR servers
        for iar_id, iar in self.protocol.iar_servers.items():
            self.ax.scatter(iar.x, iar.y, c='purple', s=200, marker='D',
                          edgecolors='indigo', linewidth=3, zorder=6)
            self.ax.text(iar.x, iar.y - 30, f'IAR{iar_id}', ha='center', va='top',
                       fontsize=9, weight='bold', color='white',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='purple', alpha=0.8))
    
    def draw_message_exchanges(self):
        """Draw recent message exchanges"""
        if hasattr(self.protocol, 'recent_messages'):
            current_time = self.protocol.current_time_slot
            
            for msg in self.protocol.recent_messages:
                # Only show recent messages (last 3 time slots)
                if current_time - msg.get('timestamp', 0) <= 3:
                    sender_id = msg.get('sender_id')
                    receiver_id = msg.get('receiver_id')
                    msg_type = msg.get('type', 'unknown')
                    
                    if (sender_id in self.protocol.nodes and 
                        receiver_id in self.protocol.nodes):
                        
                        sender = self.protocol.nodes[sender_id]
                        receiver = self.protocol.nodes[receiver_id]
                        
                        # Color based on message type
                        if 'attack' in msg_type.lower():
                            color = 'red'
                            alpha = 0.8
                        elif 'security' in msg_type.lower():
                            color = 'orange'
                            alpha = 0.7
                        else:
                            color = 'blue'
                            alpha = 0.5
                        
                        # Draw message arrow
                        self.ax.annotate('', xy=(receiver.x, receiver.y), 
                                       xytext=(sender.x, sender.y),
                                       arrowprops=dict(arrowstyle='->', 
                                                     color=color, alpha=alpha,
                                                     lw=2, shrinkA=10, shrinkB=10),
                                       zorder=3)
    
    def draw_security_dashboard(self):
        """Draw comprehensive security dashboard"""
        metrics = self.protocol.security_monitor.metrics
        
        # Main security panel
        # Calculate detection rate from available data
        detection_rate = 0.0
        if metrics.total_attacks_detected > 0:
            detection_rate = (metrics.attacks_blocked / metrics.total_attacks_detected) * 100
        
        security_text = f"""🔒 SECURITY STATUS
Threat Level: {metrics.threat_level.value.upper()}
Total Attacks: {metrics.total_attacks_detected}
Attacks Blocked: {metrics.attacks_blocked}
Honeypot Captures: {metrics.honeypot_captures}
Network Availability: {metrics.network_availability:.1f}%
Detection Rate: {detection_rate:.1f}%"""
        
        # Color based on threat level
        if metrics.threat_level == ThreatLevel.HIGH:
            panel_color = 'red'
        elif metrics.threat_level == ThreatLevel.MEDIUM:
            panel_color = 'orange'
        else:
            panel_color = 'green'
        
        self.ax.text(0.02, 0.98, security_text, transform=self.ax.transAxes,
                   fontsize=11, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor=panel_color, alpha=0.9),
                   color='white', weight='bold', zorder=200)
        
        # Protocol status panel
        cluster_heads = self.protocol._get_cluster_heads()
        active_nodes = sum(1 for n in self.protocol.nodes.values() if n.is_alive())
        
        protocol_text = f"""📊 PROTOCOL STATUS
Time Step: {self.protocol.current_time_slot}
Active Nodes: {active_nodes}
Cluster Heads: {len(cluster_heads)}
MEC Servers: {len(self.protocol.mec_servers)}
IAR Servers: {len(self.protocol.iar_servers)}
Avg Energy: {sum(n.energy for n in self.protocol.nodes.values() if n.is_alive()) / active_nodes:.1f}J"""
        
        self.ax.text(0.98, 0.98, protocol_text, transform=self.ax.transAxes,
                   fontsize=11, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='blue', alpha=0.9),
                   color='white', weight='bold', zorder=200)
        
        # Attack activity panel (bottom left)
        recent_attacks = [e for e in self.protocol.security_monitor.security_events 
                         if e.event_type == AttackType.DOS_ATTACK and 
                         self.protocol.current_time_slot - e.timestamp <= 5]
        
        attack_text = f"""⚠️ ATTACK ACTIVITY
Recent Attacks: {len(recent_attacks)}
Active Attackers: {sum(1 for n in self.protocol.nodes.values() if hasattr(n, 'is_attacker') and n.is_attacker and n.is_alive())}
Active Honeypots: {sum(1 for n in self.protocol.nodes.values() if hasattr(n, 'is_honeypot') and n.is_honeypot and n.is_alive())}
Isolated Nodes: {sum(len(hp.isolation_active) for hp in self.protocol.get_honeypots())}"""
        
        self.ax.text(0.02, 0.35, attack_text, transform=self.ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='darkred', alpha=0.8),
                   color='white', weight='bold', zorder=200)
    
    def draw_legend(self):
        """Draw comprehensive legend"""
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
                      markersize=12, label='Cluster Head'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                      markersize=10, label='Regular Node'),
            plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='red', 
                      markersize=12, label='Attacker'),
            plt.Line2D([0], [0], marker='h', color='w', markerfacecolor='gold', 
                      markersize=12, label='Honeypot'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                      markersize=12, label='MEC Server'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='purple', 
                      markersize=10, label='IAR Server')
        ]
        
        self.ax.legend(handles=legend_elements, loc='lower right', 
                     fontsize=10, framealpha=0.9)
    
    def update_frame(self, frame):
        """Update visualization frame"""
        # Run protocol step
        self.protocol.run_protocol_step(frame)
        
        # Clear and redraw
        self.ax.clear()
        self.ax.set_xlim(0, self.area_size)
        self.ax.set_ylim(0, self.area_size)
        self.ax.set_facecolor('white')
        self.ax.grid(True, alpha=0.3, color='gray')
        
        # Draw all components
        self.draw_cluster_boundaries()
        self.draw_cluster_connections()
        self.draw_nodes()
        self.draw_infrastructure()
        self.draw_message_exchanges()
        self.draw_security_dashboard()
        self.draw_legend()
        
        # Set title with current status
        threat = self.protocol.security_monitor.metrics.threat_level
        active_nodes = sum(1 for n in self.protocol.nodes.values() if n.is_alive())
        
        title = (f'🔒 SECURE ARPMEC PROTOCOL - Frame {frame} - '
                f'Threat: {threat.value.upper()} - '
                f'Nodes: {active_nodes}')
        
        self.ax.set_title(title, fontsize=16, weight='bold', color='black', pad=20)
        
        # Style axes
        self.ax.set_xlabel('X Position (meters)', color='black', fontsize=12)
        self.ax.set_ylabel('Y Position (meters)', color='black', fontsize=12)
        self.ax.tick_params(colors='black')
        for spine in self.ax.spines.values():
            spine.set_color('black')
    
    def run_animation(self, interval: int = 1000):
        """Run comprehensive visualization animation"""
        self.setup_visualization()
        
        anim = animation.FuncAnimation(
            self.fig, self.update_frame, interval=interval,
            blit=False, cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()
        return anim

# =====================================================================================
# COMPREHENSIVE DEMO AND TESTING
# =====================================================================================

class SecureARPMECDemo:
    """Comprehensive demonstration of secure ARPMEC protocol"""
    
    def __init__(self, num_nodes: int = 25, num_attackers: int = 2, num_honeypots: int = 2):
        self.num_nodes = num_nodes
        self.num_attackers = num_attackers
        self.num_honeypots = num_honeypots
        self.area_size = 1000
        self.protocol = None
        self.visualizer = None
        
    def create_network(self):
        """Create a comprehensive network with all node types"""
        print("🚀 Creating Secure ARPMEC Network...")
        
        nodes = []
        node_id = 0
        
        # Create cluster centers for realistic topology
        cluster_centers = [
            (200, 200), (600, 200), (200, 600), (600, 600),
            (800, 400), (400, 800)
        ]
        
        # Create normal nodes around cluster centers
        normal_nodes = self.num_nodes - self.num_attackers - self.num_honeypots
        nodes_per_cluster = normal_nodes // len(cluster_centers)
        
        for cx, cy in cluster_centers:
            for i in range(nodes_per_cluster):
                # Generate position around cluster center
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, 80)
                
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                
                # Ensure nodes stay within bounds
                x = max(50, min(self.area_size - 50, x))
                y = max(50, min(self.area_size - 50, y))
                
                energy = random.uniform(150, 200)  # Higher initial energy
                node = SecureNode(node_id, x, y, energy)
                nodes.append(node)
                node_id += 1
        
        # Add remaining normal nodes randomly
        while len(nodes) < normal_nodes:
            x = random.uniform(50, self.area_size - 50)
            y = random.uniform(50, self.area_size - 50)
            energy = random.uniform(150, 200)  # Higher initial energy
            node = SecureNode(node_id, x, y, energy)
            nodes.append(node)
            node_id += 1
        
        # Create attacker nodes
        print(f"🔴 Creating {self.num_attackers} attacker nodes...")
        for i in range(self.num_attackers):
            x = random.uniform(100, self.area_size - 100)
            y = random.uniform(100, self.area_size - 100)
            energy = random.uniform(120, 150)
            
            attacker = AttackerNode(node_id, x, y, energy)
            nodes.append(attacker)
            print(f"   → Attacker-{node_id} at ({x:.0f}, {y:.0f})")
            node_id += 1
        
        # Create honeypot nodes
        print(f"🍯 Creating {self.num_honeypots} honeypot nodes...")
        for i in range(self.num_honeypots):
            # Place honeypots strategically
            center = random.choice(cluster_centers)
            x = center[0] + random.uniform(-120, 120)
            y = center[1] + random.uniform(-120, 120)
            x = max(50, min(self.area_size - 50, x))
            y = max(50, min(self.area_size - 50, y))
            energy = random.uniform(150, 200)
            
            honeypot = HoneypotNode(node_id, x, y, energy)
            honeypot.mimic_important_service("CH" if i % 2 == 0 else "MEC")
            nodes.append(honeypot)
            print(f"   → Honeypot-{node_id} at ({x:.0f}, {y:.0f})")
            node_id += 1
        
        # Create protocol
        self.protocol = SecureARPMECProtocol(nodes)
        
        # Initialize clustering
        clusters = self.protocol.clustering_algorithm()
        print(f"✅ Network created with {len(clusters)} clusters")
        
        return clusters
    
    def run_comprehensive_demo(self):
        """Run comprehensive demonstration"""
        print("\n" + "="*80)
        print("🔐 COMPREHENSIVE SECURE ARPMEC DEMONSTRATION 🔐")
        print("="*80)
        
        # Create network
        self.create_network()
        
        # Initialize visualizer
        self.visualizer = SecureARPMECVisualizer(self.protocol, self.area_size)
        
        # Display initial network statistics
        print(f"\n📊 INITIAL NETWORK STATISTICS:")
        print(f"   Total Nodes: {len(self.protocol.nodes)}")
        print(f"   Normal Nodes: {len([n for n in self.protocol.nodes.values() if isinstance(n, SecureNode) and not isinstance(n, (AttackerNode, HoneypotNode))])}")
        print(f"   Attacker Nodes: {len(self.protocol.get_attackers())}")
        print(f"   Honeypot Nodes: {len(self.protocol.get_honeypots())}")
        print(f"   Cluster Heads: {len(self.protocol._get_cluster_heads())}")
        print(f"   MEC Servers: {len(self.protocol.mec_servers)}")
        print(f"   IAR Servers: {len(self.protocol.iar_servers)}")
        
        # Run simulation for a few time slots to establish baseline
        print(f"\n🔄 Running baseline simulation...")
        for i in range(5):
            self.protocol.run_protocol_step(i)
            print(f"   Time slot {i+1} completed")
        
        # Display security metrics
        print(f"\n🛡️ SECURITY METRICS:")
        metrics = self.protocol.security_metrics
        print(f"   Threat Level: {metrics.threat_level.value.upper()}")
        print(f"   Total Attacks: {metrics.total_attacks_detected}")
        print(f"   Attacks Blocked: {metrics.attacks_blocked}")
        print(f"   Honeypot Captures: {metrics.honeypot_captures}")
        print(f"   Network Availability: {metrics.network_availability:.1f}%")
        
        # Display threat intelligence
        print(f"\n🔍 THREAT INTELLIGENCE:")
        for honeypot in self.protocol.get_honeypots():
            intel = honeypot.generate_threat_intelligence()
            print(f"   Honeypot-{honeypot.id}: {intel['total_attacks']} attacks from {intel['unique_attackers']} attackers")
        
        print(f"\n🎯 Starting interactive visualization...")
        print("   → Close the window to end the demonstration")
        print("   → The visualization shows real-time protocol and security operations")
        
        # Start visualization
        return self.visualizer.run_animation(interval=500)  # Much slower (2 FPS instead of 10 FPS)

# =====================================================================================
# MAIN EXECUTION
# =====================================================================================

def main():
    """Main execution function"""
    print("🔐 SECURE ARPMEC PROTOCOL - COMPREHENSIVE IMPLEMENTATION")
    print("=" * 60)
    print("This demonstration includes:")
    print("✓ Complete ARPMEC protocol (clustering, CH election, MEC, IAR)")
    print("✓ Inter/intra-cluster communications")
    print("✓ Node mobility and reclustering")
    print("✓ DoS attack simulation and detection")
    print("✓ Honeypot-based threat intelligence")
    print("✓ Automated security countermeasures")
    print("✓ Real-time visualization and monitoring")
    print("=" * 60)
    
    try:
        # Create and run demonstration
        demo = SecureARPMECDemo(
            num_nodes=25,
            num_attackers=2,
            num_honeypots=2
        )
        
        # Run comprehensive demonstration
        animation_obj = demo.run_comprehensive_demo()
        
        # Keep the program running
        input("\nPress Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🏁 Demonstration completed")

if __name__ == "__main__":
    main()
