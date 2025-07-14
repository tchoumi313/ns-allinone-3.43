#!/usr/bin/env python3
"""
SIMPLIFIED SECURE ARPMEC PROTOCOL FOR MASTER'S RESEARCH
=====================================================

This implementation provides:
1. Complete faithful ARPMEC protocol (working base)
2. Clean security layer (non-intrusive)
3. Data collection for performance metrics
4. Security evaluation metrics

Focus: Get working results by Sunday with all required metrics
"""

import json
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set
from enum import Enum
from typing import List, Dict, Optional, Set
"""
SIMPLIFIED SECURE ARPMEC PROTOCOL FOR MASTER'S RESEARCH
=====================================================

This implementation provides:
1. Complete faithful ARPMEC protocol (working base)
2. Clean security layer (non-intrusive)
3. Data collection for performance metrics
4. Security evaluation metrics

Focus: Get working results by Sunday with all required metrics
"""

import json
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

# Import the working faithful ARPMEC
from arpmec_faithful import (ARPMECProtocol, IARServer, MECServer, MECTask,
                             Node, NodeState)

# =====================================================================================
# SECURITY DATA STRUCTURES (MINIMAL, NON-INTRUSIVE)
# =====================================================================================

class AttackType(Enum):
    DOS = "dos"
    JAMMING = "jamming"
    BLACKHOLE = "blackhole"

@dataclass
class Attack:
    attack_id: str
    attacker_id: int
    target_id: int
    attack_type: AttackType
    timestamp: float
    detected: bool = False
    blocked: bool = False

@dataclass
class SecurityMetrics:
    """Security evaluation metrics for research"""
    total_attacks: int = 0
    detected_attacks: int = 0
    blocked_attacks: int = 0
    false_positives: int = 0
    detection_time: List[float] = field(default_factory=list)
    attack_types: Dict[str, int] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Network performance metrics for research"""
    latency_measurements: List[float] = field(default_factory=list)
    energy_consumption: List[float] = field(default_factory=list)
    packet_delivery_ratio: List[float] = field(default_factory=list)
    bandwidth_utilization: List[float] = field(default_factory=list)
    network_lifetime: float = 0.0
    throughput: List[float] = field(default_factory=list)

# =====================================================================================
# SIMPLE SECURITY EXTENSIONS (MINIMAL INTRUSION)
# =====================================================================================

class AttackerNode(Node):
    """Simple attacker node that launches attacks"""
    
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 100.0):
        super().__init__(node_id, x, y, initial_energy)
        self.is_attacker = True
        self.attack_frequency = 0.4  # Increased to 40% chance per round (was 0.1)
        self.active_attacks: List[Attack] = []
    
    def launch_attack(self, target_id: int, attack_type: AttackType) -> Attack:
        """Launch an attack against target"""
        attack = Attack(
            attack_id=f"attack_{self.id}_{target_id}_{time.time()}",
            attacker_id=self.id,
            target_id=target_id,
            attack_type=attack_type,
            timestamp=time.time()
        )
        self.active_attacks.append(attack)
        print(f"🚨 ATTACK LAUNCHED: Attacker-{self.id} → Target-{target_id} ({attack_type.value})")
        return attack

class HoneypotNode(Node):
    """Enhanced honeypot for attack detection and attacker isolation"""
    
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 150.0):
        super().__init__(node_id, x, y, initial_energy)
        self.is_honeypot = True
        self.captured_attacks: List[Attack] = []
        self.captured_attackers: Set[int] = set()  # Track isolated attackers
        self.detection_rate = 0.85  # 85% detection rate
        self.detection_range = 200  # Detection/capture range
        self.threat_detected = False  # Flag for visualization
        self.isolation_duration = 30.0  # seconds to hold attacker
        self.isolation_timestamps: Dict[int, float] = {}  # Track when isolation started
    
    def detect_attack(self, attack: Attack) -> bool:
        """Detect incoming attack"""
        if random.random() < self.detection_rate:
            attack.detected = True
            attack.blocked = True
            self.captured_attacks.append(attack)
            self.threat_detected = True
            return True
        return False
    
    def capture_attacker(self, attacker_id: int) -> bool:
        """Capture and isolate an attacker"""
        if attacker_id not in self.captured_attackers:
            self.captured_attackers.add(attacker_id)
            self.isolation_timestamps[attacker_id] = time.time()
            print(f"🍯 Honeypot-{self.id} CAPTURED Attacker-{attacker_id}!")
            return True
        return False
    
    def release_expired_attackers(self) -> List[int]:
        """Release attackers after isolation period expires"""
        current_time = time.time()
        released = []
        
        for attacker_id in list(self.captured_attackers):
            if (current_time - self.isolation_timestamps.get(attacker_id, 0)) > self.isolation_duration:
                self.captured_attackers.remove(attacker_id)
                if attacker_id in self.isolation_timestamps:
                    del self.isolation_timestamps[attacker_id]
                released.append(attacker_id)
                print(f"🔓 Honeypot-{self.id} RELEASED Attacker-{attacker_id} (isolation expired)")
        
        # Reset threat detection if no active captures
        if not self.captured_attackers:
            self.threat_detected = False
            
        return released

# =====================================================================================
# ENHANCED ARPMEC WITH SECURITY MONITORING
# =====================================================================================

class SecureARPMECProtocol(ARPMECProtocol):
    """ARPMEC protocol with security monitoring overlay"""
    
    def __init__(self, nodes: List[Node], attackers: List[AttackerNode], 
                 honeypots: List[HoneypotNode], C: int = 4, R: int = 5, K: int = 3):
        # Use the working faithful implementation as base
        all_nodes = nodes + attackers + honeypots
        super().__init__(all_nodes, C, R, K)
        
        # Security components (overlay, non-intrusive)
        self.attackers = {a.id: a for a in attackers}
        self.honeypots = {h.id: h for h in honeypots}
        self.active_attacks: List[Attack] = []
        self.security_metrics = SecurityMetrics()
        self.performance_metrics = PerformanceMetrics()
        
        # Data collection
        self.round_data: List[Dict] = []
        self.start_time = time.time()
    
    def run_secure_protocol_step(self, round_num: int):
        """Run one protocol step with security monitoring"""
        round_start_time = time.time()
        
        # 1. Run normal ARPMEC protocol operations
        alive_nodes = [node for node in self.nodes.values() if node.is_alive()]
        
        # Normal protocol operations (clustering, CH election, MEC processing)
        for node in alive_nodes:
            if node.state == NodeState.CLUSTER_HEAD:
                self._fixed_cluster_head_operations(node, round_num)
            else:
                self._fixed_cluster_member_operations(node, round_num)
        
        # Inter-cluster communication
        self._generate_inter_cluster_traffic()
        self._generate_mec_tasks()
        self._process_inter_cluster_messages()
        self._process_mec_servers()
        
        # 2. Security layer (minimal intrusion)
        self._process_attacks(round_num)
        self._collect_performance_data(round_num, round_start_time)
        
        # 3. Periodic re-clustering (from faithful implementation)
        if round_num % 5 == 0 and round_num > 0:
            membership_changed = self._check_and_recluster()
            if membership_changed:
                self._build_inter_cluster_routing_table()
    
    def _process_attacks(self, round_num: int):
        """Process security attacks and detection with honeypot activation"""
        
        # Only print debug info occasionally
        if round_num % 5 == 0:  # Every 5 rounds
            print(f"🔍 Processing attacks for round {round_num}")
        
        # Release expired attackers from honeypots
        for honeypot in self.honeypots.values():
            if honeypot.is_alive():
                honeypot.release_expired_attackers()
        
        # Generate attacks from attackers (only if not captured)
        attacks_launched = 0
        active_attackers = 0
        for attacker in self.attackers.values():
            if not attacker.is_alive():
                continue
            
            active_attackers += 1
            
            # Skip attack generation if attacker is captured by any honeypot
            is_captured = False
            for honeypot in self.honeypots.values():
                if attacker.id in honeypot.captured_attackers:
                    is_captured = True
                    break
            
            if is_captured:
                continue  # Captured attackers cannot launch new attacks
            
            attack_roll = random.random()
                
            if attack_roll < attacker.attack_frequency:
                # REALISTIC ATTACK TARGETING: Only attack nodes within communication range
                targets = []
                attacker_range = 150  # Attacker transmission range (slightly higher than normal nodes)
                
                # Target cluster heads within range
                for node in self.nodes.values():
                    if (node.state == NodeState.CLUSTER_HEAD and node.is_alive() and 
                        attacker.distance_to(node) <= attacker_range):
                        targets.append((node.id, AttackType.DOS, attacker.distance_to(node)))
                
                # Target cluster members within range (can disrupt cluster operations)
                for node in self.nodes.values():
                    if (node.state == NodeState.CLUSTER_MEMBER and node.is_alive() and 
                        attacker.distance_to(node) <= attacker_range):
                        targets.append((node.id, AttackType.DOS, attacker.distance_to(node)))
                
                # Target MEC servers within range (direct attack on infrastructure)
                for mec_id, mec in self.mec_servers.items():
                    if attacker.distance_to(mec) <= attacker_range:
                        targets.append((mec_id, AttackType.DOS, attacker.distance_to(mec)))
                
                # Target IAR servers within range (infrastructure attack)
                for iar_id, iar in self.iar_servers.items():
                    if attacker.distance_to(iar) <= attacker_range:
                        targets.append((iar_id, AttackType.DOS, attacker.distance_to(iar)))
                
                if targets:
                    # Prefer closer targets (more realistic)
                    targets.sort(key=lambda x: x[2])  # Sort by distance
                    target_id, attack_type, distance = targets[0]  # Choose closest target
                    
                    attack = attacker.launch_attack(target_id, attack_type)
                    self.active_attacks.append(attack)
                    self.security_metrics.total_attacks += 1
                    attacks_launched += 1
                    
                    # Update attack type count
                    attack_name = attack_type.value
                    self.security_metrics.attack_types[attack_name] = \
                        self.security_metrics.attack_types.get(attack_name, 0) + 1
                    
                    print(f"🎯 REALISTIC ATTACK: Attacker-{attacker.id} → Target-{target_id} (range: {distance:.1f}m)")
                    
                    # Check if honeypots should capture this attacker
                    self._check_honeypot_capture(attacker, attack)
                else:
                    if round_num % 10 == 0:  # Only print occasionally to avoid spam
                        print(f"🚫 Attacker-{attacker.id}: No targets within range ({attacker_range}m)")
                    self._check_honeypot_capture(attacker, attack)
        
        # Only print summary if there's activity
        if attacks_launched > 0 or len(self.active_attacks) > 0:
            print(f"📊 Round {round_num}: {attacks_launched} new attacks, {len(self.active_attacks)} active")
        
        # Process attack detection by honeypots
        for attack in self.active_attacks[:]:
            # Check if any honeypot can detect this attack
            for honeypot in self.honeypots.values():
                if not honeypot.is_alive():
                    continue
                    
                # Honeypots can detect attacks within their range
                if attack.target_id in self.nodes:
                    target_node = self.nodes[attack.target_id]
                    distance = honeypot.distance_to(target_node)
                    
                    if distance <= honeypot.detection_range and honeypot.detect_attack(attack):
                        detection_time = time.time() - attack.timestamp
                        self.security_metrics.detected_attacks += 1
                        self.security_metrics.detection_time.append(detection_time)
                        break
            
            # Remove processed attacks
            if attack.detected or (time.time() - attack.timestamp) > 10:
                self.active_attacks.remove(attack)
                if attack.blocked:
                    self.security_metrics.blocked_attacks += 1
    
    def _check_honeypot_capture(self, attacker: AttackerNode, attack: Attack):
        """Check if honeypots should capture an attacking node"""
        for honeypot in self.honeypots.values():
            if not honeypot.is_alive():
                continue
                
            # Check if attacker is within honeypot's capture range
            distance = honeypot.distance_to(attacker)
            
            if distance <= honeypot.detection_range:
                # High probability of capture when attacking within honeypot range
                capture_chance = 0.7  # 70% chance to capture attacking node
                
                if random.random() < capture_chance:
                    if honeypot.capture_attacker(attacker.id):
                        # Mark attack as detected and blocked
                        attack.detected = True
                        attack.blocked = True
                        break
    
    def _collect_performance_data(self, round_num: int, round_start_time: float):
        """Collect performance metrics for research"""
        round_time = time.time() - round_start_time
        
        # Latency measurement (round processing time)
        self.performance_metrics.latency_measurements.append(round_time * 1000)  # ms
        
        # Energy consumption
        total_energy = sum(100 - node.energy for node in self.nodes.values() if node.is_alive())
        self.performance_metrics.energy_consumption.append(total_energy)
        
        # Packet delivery ratio (based on successful MEC tasks)
        total_tasks = sum(len(mec.completed_tasks) + len(mec.processing_tasks) 
                         for mec in self.mec_servers.values())
        completed_tasks = sum(len(mec.completed_tasks) for mec in self.mec_servers.values())
        pdr = (completed_tasks / max(total_tasks, 1)) * 100
        self.performance_metrics.packet_delivery_ratio.append(pdr)
        
        # Bandwidth utilization (based on MEC server loads)
        avg_utilization = sum(mec.get_load_percentage() for mec in self.mec_servers.values()) / len(self.mec_servers)
        self.performance_metrics.bandwidth_utilization.append(avg_utilization)
        
        # Throughput (completed tasks per second)
        throughput = completed_tasks / max(round_time, 0.001)
        self.performance_metrics.throughput.append(throughput)
        
        # Round data for analysis
        alive_nodes = sum(1 for node in self.nodes.values() if node.is_alive())
        round_data = {
            'round': round_num,
            'timestamp': time.time(),
            'alive_nodes': alive_nodes,
            'active_clusters': len(self._get_cluster_heads()),
            'total_energy': total_energy,
            'latency_ms': round_time * 1000,
            'pdr': pdr,
            'throughput': throughput,
            'attacks_detected': len([a for a in self.active_attacks if a.detected]),
            'active_attacks': len(self.active_attacks)
        }
        self.round_data.append(round_data)

# =====================================================================================
# RESEARCH DATA COLLECTION AND ANALYSIS
# =====================================================================================

class ResearchDataCollector:
    """Collect and analyze data for master's research"""
    
    def __init__(self, protocol: SecureARPMECProtocol):
        self.protocol = protocol
        self.results = {}
    
    def calculate_security_metrics(self) -> Dict:
        """Calculate security evaluation metrics"""
        metrics = self.protocol.security_metrics
        
        # Detection rate
        detection_rate = (metrics.detected_attacks / max(metrics.total_attacks, 1)) * 100
        
        # Precision (true positives / (true positives + false positives))
        precision = (metrics.detected_attacks / max(metrics.detected_attacks + metrics.false_positives, 1)) * 100
        
        # False positive rate
        false_positive_rate = (metrics.false_positives / max(metrics.total_attacks, 1)) * 100
        
        # Recall (same as detection rate in this simple model)
        recall = detection_rate
        
        # Average detection time
        avg_detection_time = sum(metrics.detection_time) / max(len(metrics.detection_time), 1)
        
        return {
            'total_attacks': metrics.total_attacks,
            'detected_attacks': metrics.detected_attacks,
            'detection_rate': detection_rate,
            'precision': precision,
            'recall': recall,
            'false_positive_rate': false_positive_rate,
            'avg_detection_time_ms': avg_detection_time * 1000,
            'attack_types': metrics.attack_types,
            'blocked_attacks': metrics.blocked_attacks,
            'block_rate': (metrics.blocked_attacks / max(metrics.total_attacks, 1)) * 100
        }
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate network performance metrics"""
        perf = self.protocol.performance_metrics
        
        return {
            'avg_latency_ms': sum(perf.latency_measurements) / max(len(perf.latency_measurements), 1),
            'max_latency_ms': max(perf.latency_measurements) if perf.latency_measurements else 0,
            'avg_energy_consumption': sum(perf.energy_consumption) / max(len(perf.energy_consumption), 1),
            'final_energy_consumption': perf.energy_consumption[-1] if perf.energy_consumption else 0,
            'avg_pdr': sum(perf.packet_delivery_ratio) / max(len(perf.packet_delivery_ratio), 1),
            'min_pdr': min(perf.packet_delivery_ratio) if perf.packet_delivery_ratio else 0,
            'avg_bandwidth_utilization': sum(perf.bandwidth_utilization) / max(len(perf.bandwidth_utilization), 1),
            'max_bandwidth_utilization': max(perf.bandwidth_utilization) if perf.bandwidth_utilization else 0,
            'avg_throughput': sum(perf.throughput) / max(len(perf.throughput), 1),
            'max_throughput': max(perf.throughput) if perf.throughput else 0,
            'network_lifetime': self._calculate_network_lifetime()
        }
    
    def _calculate_network_lifetime(self) -> float:
        """Calculate network lifetime (time until first node dies)"""
        for round_data in self.protocol.round_data:
            if round_data['alive_nodes'] < len(self.protocol.nodes):
                return round_data['timestamp'] - self.protocol.start_time
        return time.time() - self.protocol.start_time
    
    def export_data_for_analysis(self, filename: str):
        """Export all data for external analysis (MATLAB, Python, etc.)"""
        data = {
            'security_metrics': self.calculate_security_metrics(),
            'performance_metrics': self.calculate_performance_metrics(),
            'round_by_round_data': self.protocol.round_data,
            'timestamp': time.time(),
            'simulation_parameters': {
                'total_nodes': len(self.protocol.nodes),
                'attackers': len(self.protocol.attackers),
                'honeypots': len(self.protocol.honeypots),
                'mec_servers': len(self.protocol.mec_servers),
                'iar_servers': len(self.protocol.iar_servers)
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Research data exported to {filename}")
        return data

# =====================================================================================
# COMPREHENSIVE SIMULATION FOR RESEARCH
# =====================================================================================

def run_research_simulation(num_rounds: int = 50, num_runs: int = 5) -> Dict:
    """Run comprehensive simulation for master's research"""
    
    print("🎓 MASTER'S RESEARCH SIMULATION - SECURE ARPMEC")
    print("=" * 60)
    print(f"Configuration: {num_rounds} rounds, {num_runs} independent runs")
    print("Collecting: Latency, Energy, PDR, Bandwidth, Security metrics")
    print("=" * 60)
    
    all_results = []
    
    for run in range(num_runs):
        print(f"\n🔄 Run {run + 1}/{num_runs}")
        
        # Create network topology
        nodes = []
        for i in range(20):  # 20 normal nodes
            x = random.uniform(50, 750)
            y = random.uniform(50, 750)
            energy = random.uniform(90, 110)
            nodes.append(Node(i, x, y, energy))
        
        # Add attackers
        attackers = []
        for i in range(3):  # 3 attackers
            x = random.uniform(100, 700)
            y = random.uniform(100, 700)
            attacker = AttackerNode(20 + i, x, y, 120)
            attackers.append(attacker)
        
        # Add honeypots
        honeypots = []
        for i in range(2):  # 2 honeypots
            x = random.uniform(150, 650)
            y = random.uniform(150, 650)
            honeypot = HoneypotNode(23 + i, x, y, 150)
            honeypots.append(honeypot)
        
        # Create protocol
        protocol = SecureARPMECProtocol(nodes, attackers, honeypots)
        
        # Initialize clustering
        clusters = protocol.clustering_algorithm()
        print(f"   Initial clusters: {len(clusters)}")
        
        # Run simulation
        for round_num in range(num_rounds):
            protocol.run_secure_protocol_step(round_num)
            
            if round_num % 10 == 0:
                alive = sum(1 for n in protocol.nodes.values() if n.is_alive())
                attacks = len(protocol.active_attacks)
                detected = protocol.security_metrics.detected_attacks
                print(f"     Round {round_num}: {alive} alive nodes, {attacks} active attacks, {detected} detected")
        
        # Collect results
        collector = ResearchDataCollector(protocol)
        run_results = {
            'run_id': run,
            'security': collector.calculate_security_metrics(),
            'performance': collector.calculate_performance_metrics(),
            'final_stats': {
                'alive_nodes': sum(1 for n in protocol.nodes.values() if n.is_alive()),
                'active_clusters': len(protocol._get_cluster_heads()),
                'simulation_time': time.time() - protocol.start_time
            }
        }
        all_results.append(run_results)
        
        # Export individual run data
        collector.export_data_for_analysis(f"research_data_run_{run + 1}.json")
        
        print(f"   ✅ Run {run + 1} completed")
        print(f"     Detection Rate: {run_results['security']['detection_rate']:.1f}%")
        print(f"     Avg Latency: {run_results['performance']['avg_latency_ms']:.2f}ms")
        print(f"     Avg PDR: {run_results['performance']['avg_pdr']:.1f}%")
    
    # Aggregate results
    aggregate_results = _aggregate_simulation_results(all_results)
    
    # Export final aggregated data
    with open("research_final_results.json", 'w') as f:
        json.dump(aggregate_results, f, indent=2)
    
    print("\n📈 FINAL RESEARCH RESULTS")
    print("=" * 60)
    print("SECURITY METRICS:")
    print(f"  Average Detection Rate: {aggregate_results['avg_security']['detection_rate']:.1f}%")
    print(f"  Average Precision: {aggregate_results['avg_security']['precision']:.1f}%")
    print(f"  Average False Positive Rate: {aggregate_results['avg_security']['false_positive_rate']:.1f}%")
    print(f"  Average Detection Time: {aggregate_results['avg_security']['avg_detection_time_ms']:.2f}ms")
    
    print("\nPERFORMANCE METRICS:")
    print(f"  Average Latency: {aggregate_results['avg_performance']['avg_latency_ms']:.2f}ms")
    print(f"  Average Energy Consumption: {aggregate_results['avg_performance']['avg_energy_consumption']:.2f}J")
    print(f"  Average PDR: {aggregate_results['avg_performance']['avg_pdr']:.1f}%")
    print(f"  Average Bandwidth Utilization: {aggregate_results['avg_performance']['avg_bandwidth_utilization']:.1f}%")
    print(f"  Average Network Lifetime: {aggregate_results['avg_performance']['network_lifetime']:.2f}s")
    
    print(f"\n💾 All data exported for analysis")
    print(f"📊 Ready for curve generation and evaluation!")
    
    return aggregate_results

def _aggregate_simulation_results(all_results: List[Dict]) -> Dict:
    """Aggregate results from multiple simulation runs"""
    num_runs = len(all_results)
    
    # Aggregate security metrics
    avg_security = {}
    security_keys = all_results[0]['security'].keys()
    for key in security_keys:
        if isinstance(all_results[0]['security'][key], (int, float)):
            avg_security[key] = sum(r['security'][key] for r in all_results) / num_runs
        else:
            avg_security[key] = all_results[0]['security'][key]  # Keep non-numeric as-is
    
    # Aggregate performance metrics
    avg_performance = {}
    perf_keys = all_results[0]['performance'].keys()
    for key in perf_keys:
        avg_performance[key] = sum(r['performance'][key] for r in all_results) / num_runs
    
    return {
        'num_runs': num_runs,
        'individual_results': all_results,
        'avg_security': avg_security,
        'avg_performance': avg_performance,
        'std_deviations': _calculate_std_deviations(all_results)
    }

def _calculate_std_deviations(all_results: List[Dict]) -> Dict:
    """Calculate standard deviations for key metrics"""
    import statistics

    # Security metrics std dev
    detection_rates = [r['security']['detection_rate'] for r in all_results]
    security_std = {
        'detection_rate': statistics.stdev(detection_rates) if len(detection_rates) > 1 else 0
    }
    
    # Performance metrics std dev
    latencies = [r['performance']['avg_latency_ms'] for r in all_results]
    energies = [r['performance']['avg_energy_consumption'] for r in all_results]
    pdrs = [r['performance']['avg_pdr'] for r in all_results]
    
    performance_std = {
        'latency': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        'energy': statistics.stdev(energies) if len(energies) > 1 else 0,
        'pdr': statistics.stdev(pdrs) if len(pdrs) > 1 else 0
    }
    
    return {
        'security': security_std,
        'performance': performance_std
    }

# =====================================================================================
# MAIN EXECUTION
# =====================================================================================

def main():
    """Main execution for master's research"""
    print("🎓 SECURE ARPMEC - MASTER'S RESEARCH IMPLEMENTATION")
    print("📊 Collecting data for: Latency, Energy, PDR, Bandwidth, Security")
    print("🔒 Security metrics: Detection rate, Precision, Recall, False positives")
    print("⏱️  Ready for Sunday deadline!")
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--visual":
        print("\n🎬 VISUAL MODE: Run 'python research_visual_demo.py' for full visualization")
        print("📊 For now, running data collection with progress indicators...")
    
    try:
        # Run comprehensive research simulation
        results = run_research_simulation(num_rounds=50, num_runs=5)
        
        print("\n✅ SIMULATION COMPLETE - READY FOR ANALYSIS")
        print("📈 Generate curves from: research_final_results.json")
        print("📊 Individual run data: research_data_run_*.json")
        print("\n💡 TIP: Run 'python research_visual_demo.py' to see the simulation in action!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
