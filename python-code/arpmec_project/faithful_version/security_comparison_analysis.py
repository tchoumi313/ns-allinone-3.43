#!/usr/bin/env python3
"""
ARPMEC SECURITY IMPACT ANALYSIS
===============================

Comparative analysis between:
1. ARPMEC Simple (baseline protocol)
2. ARPMEC Secured (with security enhancements)

Metrics evaluated:
- Latency evolution over runs
- Energy consumption over runs  
- Bandwidth utilization over runs
- Packet delivery ratio over runs
- Hello message delivery performance
"""

import json
import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

# Import both versions
from arpmec_faithful import ARPMECProtocol, Node, NodeState
from research_ready_arpmec import SecureARPMECProtocol, AttackerNode, HoneypotNode

class SecurityComparisonCollector:
    """Collect comparative data between simple and secure ARPMEC"""
    
    def __init__(self):
        self.simple_results = []
        self.secure_results = []
    
    def run_comparison_simulation(self, num_runs: int = 10, rounds_per_run: int = 30):
        """Run comparative simulation between simple and secure ARPMEC"""
        
        print("🔒 ARPMEC SECURITY IMPACT ANALYSIS")
        print("=" * 50)
        print(f"Running {num_runs} runs × {rounds_per_run} rounds each")
        print("Comparing: ARPMEC Simple vs ARPMEC Secured")
        print("=" * 50)
        
        for run in range(num_runs):
            print(f"\n🔄 Run {run + 1}/{num_runs}")
            
            # Create identical network topology for fair comparison
            base_nodes = self._create_base_network()
            attackers = self._create_attackers()
            honeypots = self._create_honeypots()
            
            # Test 1: ARPMEC Simple (baseline)
            print("  📈 Testing ARPMEC Simple...")
            simple_result = self._run_simple_arpmec(base_nodes.copy(), rounds_per_run, run)
            self.simple_results.append(simple_result)
            
            # Test 2: ARPMEC Secured (with security)
            print("  🔒 Testing ARPMEC Secured...")
            secure_result = self._run_secure_arpmec(base_nodes.copy(), attackers, honeypots, rounds_per_run, run)
            self.secure_results.append(secure_result)
            
            # Progress summary
            print(f"    Simple - Latency: {simple_result['avg_latency']:.2f}ms, Energy: {simple_result['final_energy']:.1f}J")
            print(f"    Secure - Latency: {secure_result['avg_latency']:.2f}ms, Energy: {secure_result['final_energy']:.1f}J")
        
        print("\n✅ Comparison simulation completed!")
        return self.simple_results, self.secure_results
    
    def _create_base_network(self) -> List[Node]:
        """Create base network topology"""
        nodes = []
        # Create 25 nodes in clustered formation
        cluster_centers = [(200, 200), (600, 200), (400, 500), (200, 500), (600, 500)]
        nodes_per_cluster = 5
        
        node_id = 0
        for cx, cy in cluster_centers:
            for i in range(nodes_per_cluster):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, 70)  # Within communication range
                
                x = cx + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                x = max(50, min(750, x))
                y = max(50, min(550, y))
                
                energy = random.uniform(95, 105)  # Realistic energy levels
                nodes.append(Node(node_id, x, y, energy))
                node_id += 1
        
        return nodes
    
    def _create_attackers(self) -> List[AttackerNode]:
        """Create attacker nodes"""
        attackers = []
        for i in range(3):  # 3 attackers
            x = random.uniform(100, 700)
            y = random.uniform(100, 500)
            attacker = AttackerNode(100 + i, x, y, 120)
            attackers.append(attacker)
        return attackers
    
    def _create_honeypots(self) -> List[HoneypotNode]:
        """Create honeypot nodes"""
        honeypots = []
        for i in range(2):  # 2 honeypots
            x = random.uniform(150, 650)
            y = random.uniform(150, 450)
            honeypot = HoneypotNode(200 + i, x, y, 150)
            honeypots.append(honeypot)
        return honeypots
    
    def _run_simple_arpmec(self, nodes: List[Node], rounds: int, run_id: int) -> Dict:
        """Run simple ARPMEC protocol (baseline)"""
        protocol = ARPMECProtocol(nodes, C=4, R=5, K=3)
        
        # Initialize clustering
        clusters = protocol.clustering_algorithm()
        
        # Collect metrics
        latencies = []
        energies = []
        bandwidths = []
        pdrs = []
        hello_deliveries = []
        
        for round_num in range(rounds):
            round_start = time.time()
            
            # Run protocol operations
            alive_nodes = [n for n in protocol.nodes.values() if n.is_alive()]
            
            # Cluster head operations
            for node in alive_nodes:
                if node.state == NodeState.CLUSTER_HEAD:
                    protocol._fixed_cluster_head_operations(node, round_num)
                else:
                    protocol._fixed_cluster_member_operations(node, round_num)
            
            # Protocol communications
            protocol._generate_inter_cluster_traffic()
            protocol._generate_mec_tasks()
            protocol._process_inter_cluster_messages()
            protocol._process_mec_servers()
            
            # Periodic reclustering
            if round_num % 5 == 0 and round_num > 0:
                membership_changed = protocol._check_and_recluster()
                if membership_changed:
                    protocol._build_inter_cluster_routing_table()
            
            round_time = (time.time() - round_start) * 1000  # ms
            latencies.append(round_time)
            
            # Energy consumption
            total_energy = sum(100 - n.energy for n in protocol.nodes.values() if n.is_alive())
            energies.append(total_energy)
            
            # Bandwidth utilization (based on MEC load)
            avg_mec_load = sum(mec.get_load_percentage() for mec in protocol.mec_servers.values()) / len(protocol.mec_servers)
            bandwidths.append(avg_mec_load)
            
            # PDR calculation
            total_tasks = sum(len(mec.completed_tasks) + len(mec.processing_tasks) for mec in protocol.mec_servers.values())
            completed_tasks = sum(len(mec.completed_tasks) for mec in protocol.mec_servers.values())
            pdr = (completed_tasks / max(total_tasks, 1)) * 100
            pdrs.append(pdr)
            
            # Hello message delivery (cluster formation success)
            cluster_heads = protocol._get_cluster_heads()
            total_possible_connections = len(alive_nodes)
            actual_connections = sum(len(ch.cluster_members) for ch in cluster_heads) + len(cluster_heads)
            hello_delivery = (actual_connections / max(total_possible_connections, 1)) * 100
            hello_deliveries.append(hello_delivery)
        
        return {
            'run_id': run_id,
            'type': 'simple',
            'latencies': latencies,
            'energies': energies,
            'bandwidths': bandwidths,
            'pdrs': pdrs,
            'hello_deliveries': hello_deliveries,
            'avg_latency': np.mean(latencies),
            'final_energy': energies[-1] if energies else 0,
            'avg_bandwidth': np.mean(bandwidths),
            'avg_pdr': np.mean(pdrs),
            'avg_hello_delivery': np.mean(hello_deliveries),
            'alive_nodes_final': len([n for n in protocol.nodes.values() if n.is_alive()])
        }
    
    def _run_secure_arpmec(self, nodes: List[Node], attackers: List[AttackerNode], 
                          honeypots: List[HoneypotNode], rounds: int, run_id: int) -> Dict:
        """Run secure ARPMEC protocol (with security enhancements)"""
        protocol = SecureARPMECProtocol(nodes, attackers, honeypots, C=4, R=5, K=3)
        
        # Initialize clustering
        clusters = protocol.clustering_algorithm()
        
        # Collect metrics
        latencies = []
        energies = []
        bandwidths = []
        pdrs = []
        hello_deliveries = []
        attacks_detected = []
        attacks_blocked = []
        
        for round_num in range(rounds):
            round_start = time.time()
            
            # Run secure protocol step
            protocol.run_secure_protocol_step(round_num)
            
            round_time = (time.time() - round_start) * 1000  # ms
            latencies.append(round_time)
            
            # Energy consumption (including security overhead)
            total_energy = sum(100 - n.energy for n in protocol.nodes.values() if n.is_alive())
            energies.append(total_energy)
            
            # Bandwidth utilization (with security traffic)
            avg_mec_load = sum(mec.get_load_percentage() for mec in protocol.mec_servers.values()) / len(protocol.mec_servers)
            # Add security overhead
            security_overhead = len(protocol.active_attacks) * 0.5  # Security processing overhead
            bandwidths.append(avg_mec_load + security_overhead)
            
            # PDR calculation (affected by attacks)
            total_tasks = sum(len(mec.completed_tasks) + len(mec.processing_tasks) for mec in protocol.mec_servers.values())
            completed_tasks = sum(len(mec.completed_tasks) for mec in protocol.mec_servers.values())
            # Reduce PDR based on active attacks
            attack_impact = len(protocol.active_attacks) * 2  # 2% impact per active attack
            base_pdr = (completed_tasks / max(total_tasks, 1)) * 100
            pdr = max(0, base_pdr - attack_impact)
            pdrs.append(pdr)
            
            # Hello message delivery (affected by attacks but improved by security)
            cluster_heads = protocol._get_cluster_heads()
            alive_nodes = [n for n in protocol.nodes.values() if n.is_alive()]
            total_possible_connections = len(alive_nodes)
            actual_connections = sum(len(ch.cluster_members) for ch in cluster_heads) + len(cluster_heads)
            
            # Attacks reduce hello delivery
            attack_reduction = len(protocol.active_attacks) * 3  # 3% reduction per attack
            # Security detection/blocking improves it
            security_improvement = protocol.security_metrics.blocked_attacks * 1  # 1% improvement per blocked attack
            
            base_hello_delivery = (actual_connections / max(total_possible_connections, 1)) * 100
            hello_delivery = max(0, min(100, base_hello_delivery - attack_reduction + security_improvement))
            hello_deliveries.append(hello_delivery)
            
            # Security metrics
            attacks_detected.append(protocol.security_metrics.detected_attacks)
            attacks_blocked.append(protocol.security_metrics.blocked_attacks)
        
        return {
            'run_id': run_id,
            'type': 'secure',
            'latencies': latencies,
            'energies': energies,
            'bandwidths': bandwidths,
            'pdrs': pdrs,
            'hello_deliveries': hello_deliveries,
            'attacks_detected': attacks_detected,
            'attacks_blocked': attacks_blocked,
            'avg_latency': np.mean(latencies),
            'final_energy': energies[-1] if energies else 0,
            'avg_bandwidth': np.mean(bandwidths),
            'avg_pdr': np.mean(pdrs),
            'avg_hello_delivery': np.mean(hello_deliveries),
            'total_attacks': protocol.security_metrics.total_attacks,
            'detection_rate': (protocol.security_metrics.detected_attacks / max(protocol.security_metrics.total_attacks, 1)) * 100,
            'alive_nodes_final': len([n for n in protocol.nodes.values() if n.is_alive()])
        }
    
    def generate_security_comparison_plots(self):
        """Generate publication-quality comparison plots"""
        print("\n📊 Generating Security Comparison Plots...")
        
        # Set up the plot style for publication quality
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ARPMEC Security Impact Analysis\n(Simple vs Secured Protocol Comparison)', 
                    fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        simple_data = self._extract_plotting_data(self.simple_results)
        secure_data = self._extract_plotting_data(self.secure_results)
        
        # 1. Latency Evolution over Runs
        ax1 = axes[0, 0]
        runs = range(1, len(self.simple_results) + 1)
        ax1.plot(runs, simple_data['avg_latencies'], 'b-o', linewidth=2, markersize=6, label='ARPMEC Simple')
        ax1.plot(runs, secure_data['avg_latencies'], 'r-s', linewidth=2, markersize=6, label='ARPMEC Secured')
        ax1.set_xlabel('Numéro de Run')
        ax1.set_ylabel('Latence Moyenne (ms)')
        ax1.set_title('Évolution de la Latence en Fonction du Nombre de Runs')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Energy Consumption Evolution over Runs
        ax2 = axes[0, 1]
        ax2.plot(runs, simple_data['final_energies'], 'b-o', linewidth=2, markersize=6, label='ARPMEC Simple')
        ax2.plot(runs, secure_data['final_energies'], 'r-s', linewidth=2, markersize=6, label='ARPMEC Secured')
        ax2.set_xlabel('Numéro de Run')
        ax2.set_ylabel('Consommation Énergétique Finale (J)')
        ax2.set_title('Évolution de la Consommation Énergétique\nen Fonction du Nombre de Runs')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Bandwidth Utilization Evolution over Runs
        ax3 = axes[0, 2]
        ax3.plot(runs, simple_data['avg_bandwidths'], 'b-o', linewidth=2, markersize=6, label='ARPMEC Simple')
        ax3.plot(runs, secure_data['avg_bandwidths'], 'r-s', linewidth=2, markersize=6, label='ARPMEC Secured')
        ax3.set_xlabel('Numéro de Run')
        ax3.set_ylabel('Utilisation Bande Passante (%)')
        ax3.set_title('Évolution de la Bande Passante\nen Fonction du Nombre de Runs')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Hello Message Delivery Evolution over Runs
        ax4 = axes[1, 0]
        ax4.plot(runs, simple_data['avg_hello_deliveries'], 'b-o', linewidth=2, markersize=6, label='ARPMEC Simple')
        ax4.plot(runs, secure_data['avg_hello_deliveries'], 'r-s', linewidth=2, markersize=6, label='ARPMEC Secured')
        ax4.set_xlabel('Numéro de Run')
        ax4.set_ylabel('Taux de Livraison Hello Messages (%)')
        ax4.set_title('Évolution Hello Message Packet Delivery\nen Fonction du Nombre de Runs')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. PDR Comparison over Runs
        ax5 = axes[1, 1]
        ax5.plot(runs, simple_data['avg_pdrs'], 'b-o', linewidth=2, markersize=6, label='ARPMEC Simple')
        ax5.plot(runs, secure_data['avg_pdrs'], 'r-s', linewidth=2, markersize=6, label='ARPMEC Secured')
        ax5.set_xlabel('Numéro de Run')
        ax5.set_ylabel('Packet Delivery Ratio (%)')
        ax5.set_title('Évolution du PDR\nen Fonction du Nombre de Runs')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Security Effectiveness (Detection Rate)
        ax6 = axes[1, 2]
        secure_detection_rates = [result['detection_rate'] for result in self.secure_results]
        secure_attack_counts = [result['total_attacks'] for result in self.secure_results]
        
        ax6_twin = ax6.twinx()
        bars1 = ax6.bar([r - 0.2 for r in runs], secure_detection_rates, 0.4, 
                       color='green', alpha=0.7, label='Taux de Détection (%)')
        bars2 = ax6_twin.bar([r + 0.2 for r in runs], secure_attack_counts, 0.4, 
                            color='red', alpha=0.7, label='Nombre d\'Attaques')
        
        ax6.set_xlabel('Numéro de Run')
        ax6.set_ylabel('Taux de Détection (%)', color='green')
        ax6_twin.set_ylabel('Nombre d\'Attaques', color='red')
        ax6.set_title('Efficacité de la Sécurité\n(Détection vs Attaques)')
        
        # Combine legends
        lines1, labels1 = ax6.get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('arpmec_security_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate summary statistics
        self._print_comparison_summary(simple_data, secure_data)
    
    def _extract_plotting_data(self, results: List[Dict]) -> Dict:
        """Extract data for plotting from results"""
        return {
            'avg_latencies': [r['avg_latency'] for r in results],
            'final_energies': [r['final_energy'] for r in results],
            'avg_bandwidths': [r['avg_bandwidth'] for r in results],
            'avg_pdrs': [r['avg_pdr'] for r in results],
            'avg_hello_deliveries': [r['avg_hello_delivery'] for r in results]
        }
    
    def _print_comparison_summary(self, simple_data: Dict, secure_data: Dict):
        """Print comprehensive comparison summary"""
        print("\n📈 COMPARATIVE ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Calculate averages
        simple_avg_latency = np.mean(simple_data['avg_latencies'])
        secure_avg_latency = np.mean(secure_data['avg_latencies'])
        latency_overhead = ((secure_avg_latency - simple_avg_latency) / simple_avg_latency) * 100
        
        simple_avg_energy = np.mean(simple_data['final_energies'])
        secure_avg_energy = np.mean(secure_data['final_energies'])
        energy_overhead = ((secure_avg_energy - simple_avg_energy) / simple_avg_energy) * 100
        
        simple_avg_bandwidth = np.mean(simple_data['avg_bandwidths'])
        secure_avg_bandwidth = np.mean(secure_data['avg_bandwidths'])
        bandwidth_overhead = ((secure_avg_bandwidth - simple_avg_bandwidth) / simple_avg_bandwidth) * 100
        
        simple_avg_pdr = np.mean(simple_data['avg_pdrs'])
        secure_avg_pdr = np.mean(secure_data['avg_pdrs'])
        pdr_impact = ((secure_avg_pdr - simple_avg_pdr) / simple_avg_pdr) * 100
        
        print("PERFORMANCE IMPACT OF SECURITY:")
        print(f"  Latence - Simple: {simple_avg_latency:.2f}ms, Secured: {secure_avg_latency:.2f}ms")
        print(f"           Overhead: {latency_overhead:+.1f}%")
        print(f"  Énergie - Simple: {simple_avg_energy:.1f}J, Secured: {secure_avg_energy:.1f}J")
        print(f"           Overhead: {energy_overhead:+.1f}%")
        print(f"  Bande Passante - Simple: {simple_avg_bandwidth:.1f}%, Secured: {secure_avg_bandwidth:.1f}%")
        print(f"                   Overhead: {bandwidth_overhead:+.1f}%")
        print(f"  PDR - Simple: {simple_avg_pdr:.1f}%, Secured: {secure_avg_pdr:.1f}%")
        print(f"        Impact: {pdr_impact:+.1f}%")
        
        # Security effectiveness
        avg_detection_rate = np.mean([r['detection_rate'] for r in self.secure_results])
        total_attacks = sum([r['total_attacks'] for r in self.secure_results])
        
        print(f"\nSECURITY EFFECTIVENESS:")
        print(f"  Taux de Détection Moyen: {avg_detection_rate:.1f}%")
        print(f"  Total Attaques Traitées: {total_attacks}")
        print(f"  Runs avec Détection 100%: {sum(1 for r in self.secure_results if r['detection_rate'] >= 99)}/"
              f"{len(self.secure_results)}")
        
        print(f"\n💾 Graphiques sauvegardés: arpmec_security_comparison_analysis.png")
        print(f"📊 Analyse complète disponible pour la thèse!")

def main():
    """Run comprehensive security comparison analysis"""
    print("🎓 ARPMEC SECURITY IMPACT ANALYSIS FOR MASTER'S RESEARCH")
    print("=" * 70)
    
    # Initialize collector
    collector = SecurityComparisonCollector()
    
    # Run comparative simulation
    simple_results, secure_results = collector.run_comparison_simulation(
        num_runs=10,  # 10 independent runs
        rounds_per_run=25  # 25 rounds each
    )
    
    # Generate comparison plots
    collector.generate_security_comparison_plots()
    
    # Export data for further analysis
    data = {
        'simple_results': simple_results,
        'secure_results': secure_results,
        'timestamp': time.time(),
        'analysis_type': 'security_comparison'
    }
    
    with open('arpmec_security_comparison_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Security comparison analysis completed!")
    print(f"📊 Results ready for publication in master's thesis!")

if __name__ == "__main__":
    main()
