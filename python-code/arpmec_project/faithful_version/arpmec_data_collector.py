#!/usr/bin/env python3
"""
ARPMEC Data Collection and Performance Analysis
==============================================

This module collects comprehensive performance metrics and generates research-quality plots
for the ARPMEC protocol evaluation. Ready for Sunday deadline!

Metrics collected:
- Latency (ms)
- Energy consumption (J)
- Packet Delivery Ratio (%)
- Bandwidth utilization (%)
- Network lifetime
- Clustering efficiency
"""

import json
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from datetime import datetime

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. CSV export will use basic format.")

@dataclass
class PerformanceData:
    """Complete performance data collection"""
    # Time series data
    timestamps: List[float] = field(default_factory=list)
    rounds: List[int] = field(default_factory=list)
    
    # Core metrics
    latency_ms: List[float] = field(default_factory=list)
    energy_consumption: List[float] = field(default_factory=list)
    packet_delivery_ratio: List[float] = field(default_factory=list)
    bandwidth_utilization: List[float] = field(default_factory=list)
    
    # Network state
    alive_nodes: List[int] = field(default_factory=list)
    active_clusters: List[int] = field(default_factory=list)
    total_energy_remaining: List[float] = field(default_factory=list)
    
    # Protocol-specific
    mec_server_loads: List[Dict[int, float]] = field(default_factory=list)
    inter_cluster_messages: List[int] = field(default_factory=list)
    successful_tasks: List[int] = field(default_factory=list)
    failed_tasks: List[int] = field(default_factory=list)

class ARPMECDataCollector:
    """Comprehensive data collector for ARPMEC protocol"""
    
    def __init__(self, protocol, output_dir: str = "results"):
        self.protocol = protocol
        self.output_dir = output_dir
        self.data = PerformanceData()
        self.start_time = time.time()
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔬 Data Collector initialized - Output: {output_dir}/")
    
    def collect_round_data(self, round_num: int):
        """Collect all performance metrics for current round"""
        current_time = time.time()
        
        # Basic round info
        self.data.timestamps.append(current_time)
        self.data.rounds.append(round_num)
        
        # 1. LATENCY - Measure protocol processing time
        round_start = time.time()
        # Simulate protocol operations
        cluster_heads = self.protocol._get_cluster_heads()
        processing_time = (time.time() - round_start) * 1000  # Convert to ms
        self.data.latency_ms.append(processing_time)
        
        # 2. ENERGY CONSUMPTION - Total energy consumed
        total_energy_consumed = 0
        total_energy_remaining = 0
        alive_count = 0
        
        for node in self.protocol.nodes.values():
            if node.is_alive():
                alive_count += 1
                energy_consumed = 100 - node.energy  # Assuming 100J initial
                total_energy_consumed += energy_consumed
                total_energy_remaining += node.energy
        
        self.data.energy_consumption.append(total_energy_consumed)
        self.data.total_energy_remaining.append(total_energy_remaining)
        self.data.alive_nodes.append(alive_count)
        
        # 3. PACKET DELIVERY RATIO - Based on successful MEC tasks
        total_tasks = 0
        completed_tasks = 0
        failed_tasks = 0
        
        for mec in self.protocol.mec_servers.values():
            completed = len(getattr(mec, 'completed_tasks', []))
            processing = len(getattr(mec, 'processing_tasks', []))
            total_tasks += completed + processing
            completed_tasks += completed
        
        # Calculate PDR
        pdr = (completed_tasks / max(total_tasks, 1)) * 100
        self.data.packet_delivery_ratio.append(pdr)
        self.data.successful_tasks.append(completed_tasks)
        self.data.failed_tasks.append(failed_tasks)
        
        # 4. BANDWIDTH UTILIZATION - Based on MEC server loads
        mec_loads = {}
        total_utilization = 0
        
        for mec_id, mec in self.protocol.mec_servers.items():
            load_pct = mec.get_load_percentage()
            mec_loads[mec_id] = load_pct
            total_utilization += load_pct
        
        avg_utilization = total_utilization / len(self.protocol.mec_servers)
        self.data.bandwidth_utilization.append(avg_utilization)
        self.data.mec_server_loads.append(mec_loads.copy())
        
        # 5. CLUSTERING METRICS
        active_clusters = len(cluster_heads)
        self.data.active_clusters.append(active_clusters)
        
        # 6. INTER-CLUSTER COMMUNICATION
        inter_cluster_count = 0
        for node in self.protocol.nodes.values():
            if hasattr(node, 'inter_cluster_messages'):
                inter_cluster_count += len(node.inter_cluster_messages)
        self.data.inter_cluster_messages.append(inter_cluster_count)
        
        # Print progress
        if round_num % 10 == 0:
            print(f"📊 Round {round_num}: Latency={processing_time:.2f}ms, "
                  f"PDR={pdr:.1f}%, Energy={total_energy_remaining:.1f}J, "
                  f"Clusters={active_clusters}")
    
    def generate_performance_plots(self, save_plots: bool = True):
        """Generate comprehensive performance plots for research"""
        print("📈 Generating performance plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ARPMEC Protocol Performance Analysis', fontsize=16, fontweight='bold')
        
        rounds = np.array(self.data.rounds)
        
        # 1. Latency over time
        ax1 = axes[0, 0]
        ax1.plot(rounds, self.data.latency_ms, 'b-', linewidth=2, label='Processing Latency')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Network Latency Performance')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Energy consumption
        ax2 = axes[0, 1]
        ax2.plot(rounds, self.data.energy_consumption, 'r-', linewidth=2, label='Energy Consumed')
        ax2.plot(rounds, self.data.total_energy_remaining, 'g-', linewidth=2, label='Energy Remaining')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Energy (J)')
        ax2.set_title('Energy Consumption Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Packet Delivery Ratio
        ax3 = axes[0, 2]
        ax3.plot(rounds, self.data.packet_delivery_ratio, 'm-', linewidth=2, label='PDR')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('PDR (%)')
        ax3.set_title('Packet Delivery Ratio')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0, 100)
        
        # 4. Bandwidth Utilization
        ax4 = axes[1, 0]
        ax4.plot(rounds, self.data.bandwidth_utilization, 'c-', linewidth=2, label='Bandwidth Usage')
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Utilization (%)')
        ax4.set_title('Bandwidth Utilization')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. Network State
        ax5 = axes[1, 1]
        ax5.plot(rounds, self.data.alive_nodes, 'k-', linewidth=2, label='Alive Nodes')
        ax5.plot(rounds, self.data.active_clusters, 'orange', linewidth=2, label='Active Clusters')
        ax5.set_xlabel('Round')
        ax5.set_ylabel('Count')
        ax5.set_title('Network State Evolution')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. Task Success Rate
        ax6 = axes[1, 2]
        if self.data.successful_tasks and self.data.failed_tasks:
            success_rate = [s/(s+f+0.001)*100 for s, f in zip(self.data.successful_tasks, self.data.failed_tasks)]
        else:
            success_rate = [0] * len(rounds)
        ax6.plot(rounds, success_rate, 'purple', linewidth=2, label='Success Rate')
        ax6.set_xlabel('Round')
        ax6.set_ylabel('Success Rate (%)')
        ax6.set_title('MEC Task Success Rate')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        ax6.set_ylim(0, 100)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/arpmec_performance_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📈 Performance plots saved: {filename}")
        
        plt.show()
        return fig
    
    def generate_comparison_plots(self, baseline_data=None):
        """Generate comparison plots against baseline"""
        if not baseline_data:
            print("⚠️ No baseline data provided for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ARPMEC vs Baseline Comparison', fontsize=16, fontweight='bold')
        
        rounds = np.array(self.data.rounds)
        
        # Latency comparison
        ax1 = axes[0, 0]
        ax1.plot(rounds, self.data.latency_ms, 'b-', linewidth=2, label='ARPMEC')
        if baseline_data.get('latency'):
            ax1.plot(rounds[:len(baseline_data['latency'])], baseline_data['latency'], 'r--', linewidth=2, label='Baseline')
        ax1.set_title('Latency Comparison')
        ax1.set_ylabel('Latency (ms)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Energy comparison
        ax2 = axes[0, 1]
        ax2.plot(rounds, self.data.energy_consumption, 'b-', linewidth=2, label='ARPMEC')
        if baseline_data.get('energy'):
            ax2.plot(rounds[:len(baseline_data['energy'])], baseline_data['energy'], 'r--', linewidth=2, label='Baseline')
        ax2.set_title('Energy Consumption Comparison')
        ax2.set_ylabel('Energy (J)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # PDR comparison
        ax3 = axes[1, 0]
        ax3.plot(rounds, self.data.packet_delivery_ratio, 'b-', linewidth=2, label='ARPMEC')
        if baseline_data.get('pdr'):
            ax3.plot(rounds[:len(baseline_data['pdr'])], baseline_data['pdr'], 'r--', linewidth=2, label='Baseline')
        ax3.set_title('PDR Comparison')
        ax3.set_ylabel('PDR (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Bandwidth comparison
        ax4 = axes[1, 1]
        ax4.plot(rounds, self.data.bandwidth_utilization, 'b-', linewidth=2, label='ARPMEC')
        if baseline_data.get('bandwidth'):
            ax4.plot(rounds[:len(baseline_data['bandwidth'])], baseline_data['bandwidth'], 'r--', linewidth=2, label='Baseline')
        ax4.set_title('Bandwidth Utilization Comparison')
        ax4.set_ylabel('Utilization (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/arpmec_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plots saved: {filename}")
        plt.show()
    
    def export_data(self, format='json'):
        """Export collected data for further analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            # Convert to serializable format
            export_data = {
                'metadata': {
                    'timestamp': timestamp,
                    'total_rounds': len(self.data.rounds),
                    'simulation_duration': time.time() - self.start_time,
                    'protocol': 'ARPMEC',
                    'nodes': len(self.protocol.nodes),
                    'mec_servers': len(self.protocol.mec_servers),
                    'iar_servers': len(self.protocol.iar_servers)
                },
                'performance_data': {
                    'rounds': self.data.rounds,
                    'timestamps': [t - self.start_time for t in self.data.timestamps],  # Relative time
                    'latency_ms': self.data.latency_ms,
                    'energy_consumption': self.data.energy_consumption,
                    'packet_delivery_ratio': self.data.packet_delivery_ratio,
                    'bandwidth_utilization': self.data.bandwidth_utilization,
                    'alive_nodes': self.data.alive_nodes,
                    'active_clusters': self.data.active_clusters,
                    'successful_tasks': self.data.successful_tasks
                }
            }
            
            filename = f"{self.output_dir}/arpmec_data_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"📁 Data exported: {filename}")
        
        elif format == 'csv':
            if HAS_PANDAS:
                # Create DataFrame for CSV export
                df_data = {
                    'Round': self.data.rounds,
                    'Timestamp': [t - self.start_time for t in self.data.timestamps],
                    'Latency_ms': self.data.latency_ms,
                    'Energy_Consumption': self.data.energy_consumption,
                    'PDR_Percent': self.data.packet_delivery_ratio,
                    'Bandwidth_Utilization': self.data.bandwidth_utilization,
                    'Alive_Nodes': self.data.alive_nodes,
                    'Active_Clusters': self.data.active_clusters
                }
                
                df = pd.DataFrame(df_data)
                filename = f"{self.output_dir}/arpmec_data_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"📁 CSV data exported: {filename}")
            else:
                # Basic CSV export without pandas
                filename = f"{self.output_dir}/arpmec_data_{timestamp}.csv"
                with open(filename, 'w') as f:
                    # Write header
                    f.write("Round,Timestamp,Latency_ms,Energy_Consumption,PDR_Percent,Bandwidth_Utilization,Alive_Nodes,Active_Clusters\n")
                    
                    # Write data
                    for i in range(len(self.data.rounds)):
                        f.write(f"{self.data.rounds[i]},{self.data.timestamps[i] - self.start_time:.3f},")
                        f.write(f"{self.data.latency_ms[i]:.3f},{self.data.energy_consumption[i]:.3f},")
                        f.write(f"{self.data.packet_delivery_ratio[i]:.3f},{self.data.bandwidth_utilization[i]:.3f},")
                        f.write(f"{self.data.alive_nodes[i]},{self.data.active_clusters[i]}\n")
                
                print(f"📁 Basic CSV data exported: {filename}")
    
    def generate_summary_report(self):
        """Generate summary statistics report"""
        if not self.data.rounds:
            print("⚠️ No data collected yet")
            return
        
        print("\n" + "="*60)
        print("📊 ARPMEC PERFORMANCE SUMMARY REPORT")
        print("="*60)
        
        # Calculate averages and statistics
        avg_latency = np.mean(self.data.latency_ms)
        std_latency = np.std(self.data.latency_ms)
        
        avg_pdr = np.mean(self.data.packet_delivery_ratio)
        std_pdr = np.std(self.data.packet_delivery_ratio)
        
        final_energy = self.data.total_energy_remaining[-1] if self.data.total_energy_remaining else 0
        initial_energy = self.data.total_energy_remaining[0] if self.data.total_energy_remaining else 0
        energy_efficiency = (final_energy / max(initial_energy, 1)) * 100
        
        avg_bandwidth = np.mean(self.data.bandwidth_utilization)
        
        print(f"Simulation Duration: {len(self.data.rounds)} rounds")
        print(f"Network Nodes: {len(self.protocol.nodes)}")
        print(f"Infrastructure: {len(self.protocol.mec_servers)} MEC, {len(self.protocol.iar_servers)} IAR")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  Average Latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
        print(f"  Average PDR: {avg_pdr:.1f} ± {std_pdr:.1f} %")
        print(f"  Energy Efficiency: {energy_efficiency:.1f}%")
        print(f"  Bandwidth Utilization: {avg_bandwidth:.1f}%")
        print(f"  Network Lifetime: {final_energy:.1f}J remaining")
        
        if self.data.successful_tasks:
            total_tasks = sum(self.data.successful_tasks)
            print(f"  Total Successful Tasks: {total_tasks}")
        
        print("="*60)
        
        return {
            'avg_latency': avg_latency,
            'avg_pdr': avg_pdr,
            'energy_efficiency': energy_efficiency,
            'avg_bandwidth': avg_bandwidth,
            'total_rounds': len(self.data.rounds)
        }
