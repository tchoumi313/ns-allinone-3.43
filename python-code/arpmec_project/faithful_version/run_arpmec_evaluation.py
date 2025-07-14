#!/usr/bin/env python3
"""
ARPMEC COMPREHENSIVE SIMULATION RUNNER
====================================

This script runs complete ARPMEC simulations and generates all required data and plots
for research evaluation. Ready for Sunday deadline!

Features:
- Multiple simulation runs for statistical significance
- Comprehensive data collection
- Automatic plot generation
- Research-quality output
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from arpmec_faithful import ARPMECProtocol, Node
from arpmec_data_collector import ARPMECDataCollector

class ARPMECSimulationRunner:
    """Complete simulation runner for ARPMEC research"""
    
    def __init__(self, output_dir: str = "simulation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Simulation parameters
        self.num_nodes = 25
        self.area_size = 1000
        self.simulation_rounds = 100
        self.num_runs = 5  # Multiple runs for statistical significance
        
        print(f"🔬 ARPMEC Simulation Runner Initialized")
        print(f"📁 Output Directory: {output_dir}")
        print(f"📊 Configuration: {self.num_nodes} nodes, {self.simulation_rounds} rounds, {self.num_runs} runs")
    
    def create_network_scenario(self, scenario_id: int = 1):
        """Create network scenario for testing"""
        print(f"\n🌐 Creating Network Scenario {scenario_id}")
        
        # Create nodes in clustered deployment
        nodes = []
        node_id = 0
        
        # Cluster centers for realistic deployment
        cluster_centers = [
            (200, 200), (600, 200), (400, 600), (800, 400), (300, 800)
        ]
        
        nodes_per_cluster = self.num_nodes // len(cluster_centers)
        
        for cx, cy in cluster_centers:
            for i in range(nodes_per_cluster):
                # Random position within cluster
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, 80)  # 80m cluster radius
                
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
                
                # Keep within bounds
                x = max(50, min(self.area_size - 50, x))
                y = max(50, min(self.area_size - 50, y))
                
                # Varied initial energy
                energy = np.random.uniform(90, 110)
                nodes.append(Node(node_id, x, y, energy))
                node_id += 1
        
        # Add remaining nodes randomly
        while len(nodes) < self.num_nodes:
            x = np.random.uniform(50, self.area_size - 50)
            y = np.random.uniform(50, self.area_size - 50)
            energy = np.random.uniform(90, 110)
            nodes.append(Node(node_id, x, y, energy))
            node_id += 1
        
        print(f"✅ Created {len(nodes)} nodes in {len(cluster_centers)} clusters")
        return nodes
    
    def run_single_simulation(self, run_id: int):
        """Run a single complete simulation"""
        print(f"\n🚀 Starting Simulation Run {run_id + 1}/{self.num_runs}")
        
        # Create network
        nodes = self.create_network_scenario(run_id + 1)
        
        # Initialize protocol
        protocol = ARPMECProtocol(nodes, C=4, R=5, K=3)
        
        # Perform initial clustering
        clusters = protocol.clustering_algorithm()
        print(f"📊 Initial clustering: {len(clusters)} clusters formed")
        
        # Initialize data collector
        collector = ARPMECDataCollector(protocol, f"{self.output_dir}/run_{run_id + 1}")
        
        # Run simulation rounds
        print(f"⏳ Running {self.simulation_rounds} simulation rounds...")
        
        for round_num in range(self.simulation_rounds):
            round_start = time.time()
            
            # Protocol operations
            protocol.current_time_slot = round_num
            
            # Update node mobility
            self._update_mobility(protocol, round_num)
            
            # Periodic reclustering
            if round_num % 10 == 0 and round_num > 0:
                clusters = protocol.clustering_algorithm()
                protocol._build_inter_cluster_routing_table()
            
            # Protocol operations
            cluster_heads = protocol._get_cluster_heads()
            
            # Generate traffic and process
            protocol._generate_inter_cluster_traffic()
            protocol._generate_mec_tasks()
            protocol._process_inter_cluster_messages()
            protocol._process_mec_servers()
            
            # Collect data
            collector.collect_round_data(round_num)
            
            # Progress indicator
            if round_num % 20 == 0:
                elapsed = time.time() - round_start
                print(f"  Round {round_num}: {elapsed*1000:.1f}ms processing time")
        
        print(f"✅ Simulation Run {run_id + 1} completed")
        
        # Generate plots for this run
        collector.generate_performance_plots(save_plots=True)
        collector.export_data('json')
        collector.export_data('csv')
        summary = collector.generate_summary_report()
        
        return collector, summary
    
    def _update_mobility(self, protocol, round_num):
        """Update node mobility during simulation"""
        # Simple mobility model - nodes move slowly
        for node in protocol.nodes.values():
            if node.is_alive():
                # Random walk with small steps
                dx = np.random.uniform(-2, 2)  # Small movement
                dy = np.random.uniform(-2, 2)
                
                # Update position
                node.x = max(50, min(protocol.area_size - 50, node.x + dx))
                node.y = max(50, min(protocol.area_size - 50, node.y + dy))
                
                # Update energy for movement
                movement_cost = 0.1  # Small energy cost
                node.update_energy(movement_cost)
    
    def run_complete_evaluation(self):
        """Run complete evaluation with multiple runs"""
        print(f"\n🎯 STARTING COMPLETE ARPMEC EVALUATION")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        all_collectors = []
        all_summaries = []
        
        # Run multiple simulations
        for run_id in range(self.num_runs):
            collector, summary = self.run_single_simulation(run_id)
            all_collectors.append(collector)
            all_summaries.append(summary)
        
        # Generate aggregate analysis
        self._generate_aggregate_analysis(all_collectors, all_summaries)
        
        print(f"\n🎉 EVALUATION COMPLETE!")
        print(f"📁 Results saved in: {self.output_dir}/")
        print(f"📊 Generated plots and data for {self.num_runs} simulation runs")
        
        return all_collectors, all_summaries
    
    def _generate_aggregate_analysis(self, collectors, summaries):
        """Generate aggregate analysis across all runs"""
        print(f"\n📈 Generating Aggregate Analysis...")
        
        # Collect data from all runs
        all_latency = []
        all_pdr = []
        all_energy = []
        all_bandwidth = []
        
        for collector in collectors:
            all_latency.extend(collector.data.latency_ms)
            all_pdr.extend(collector.data.packet_delivery_ratio)
            all_energy.extend(collector.data.energy_consumption)
            all_bandwidth.extend(collector.data.bandwidth_utilization)
        
        # Create aggregate plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'ARPMEC Aggregate Performance Analysis ({self.num_runs} runs)', 
                     fontsize=16, fontweight='bold')
        
        # Latency distribution
        ax1 = axes[0, 0]
        ax1.hist(all_latency, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Latency (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Latency Distribution\nMean: {np.mean(all_latency):.2f}ms')
        ax1.grid(True, alpha=0.3)
        
        # PDR distribution
        ax2 = axes[0, 1]
        ax2.hist(all_pdr, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('PDR (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'PDR Distribution\nMean: {np.mean(all_pdr):.1f}%')
        ax2.grid(True, alpha=0.3)
        
        # Energy consumption distribution
        ax3 = axes[1, 0]
        ax3.hist(all_energy, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax3.set_xlabel('Energy Consumption (J)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Energy Distribution\nMean: {np.mean(all_energy):.1f}J')
        ax3.grid(True, alpha=0.3)
        
        # Bandwidth utilization distribution
        ax4 = axes[1, 1]
        ax4.hist(all_bandwidth, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Bandwidth Utilization (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Bandwidth Distribution\nMean: {np.mean(all_bandwidth):.1f}%')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save aggregate plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/arpmec_aggregate_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Aggregate analysis saved: {filename}")
        plt.show()
        
        # Generate performance comparison table
        self._generate_performance_table(summaries)
    
    def _generate_performance_table(self, summaries):
        """Generate performance comparison table"""
        print(f"\n📋 PERFORMANCE COMPARISON TABLE")
        print("="*80)
        print(f"{'Run':<5} {'Latency(ms)':<12} {'PDR(%)':<8} {'Energy Eff(%)':<12} {'Bandwidth(%)':<12}")
        print("-"*80)
        
        for i, summary in enumerate(summaries):
            print(f"{i+1:<5} {summary['avg_latency']:<12.2f} {summary['avg_pdr']:<8.1f} "
                  f"{summary['energy_efficiency']:<12.1f} {summary['avg_bandwidth']:<12.1f}")
        
        # Calculate overall averages
        avg_latency = np.mean([s['avg_latency'] for s in summaries])
        avg_pdr = np.mean([s['avg_pdr'] for s in summaries])
        avg_energy = np.mean([s['energy_efficiency'] for s in summaries])
        avg_bandwidth = np.mean([s['avg_bandwidth'] for s in summaries])
        
        print("-"*80)
        print(f"{'AVG':<5} {avg_latency:<12.2f} {avg_pdr:<8.1f} {avg_energy:<12.1f} {avg_bandwidth:<12.1f}")
        print("="*80)
        
        # Save table to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/performance_summary_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("ARPMEC PERFORMANCE SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Runs: {len(summaries)}\n")
            f.write(f"Nodes: {self.num_nodes}\n")
            f.write(f"Rounds per run: {self.simulation_rounds}\n\n")
            
            f.write("AVERAGE PERFORMANCE METRICS:\n")
            f.write(f"  Latency: {avg_latency:.2f} ms\n")
            f.write(f"  PDR: {avg_pdr:.1f} %\n")
            f.write(f"  Energy Efficiency: {avg_energy:.1f} %\n")
            f.write(f"  Bandwidth Utilization: {avg_bandwidth:.1f} %\n")
        
        print(f"📁 Performance summary saved: {filename}")

def main():
    """Main execution function"""
    print("🚀 ARPMEC COMPREHENSIVE EVALUATION SYSTEM")
    print("⏰ Targeting Sunday deadline - generating all required data!")
    print("="*60)
    
    # Check if output directory should be custom
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"arpmec_results_{timestamp}"
    
    # Create and run simulation
    runner = ARPMECSimulationRunner(output_dir)
    
    print(f"\n📋 SIMULATION CONFIGURATION:")
    print(f"  Network size: {runner.num_nodes} nodes")
    print(f"  Area: {runner.area_size}x{runner.area_size}m")
    print(f"  Rounds per run: {runner.simulation_rounds}")
    print(f"  Number of runs: {runner.num_runs}")
    print(f"  Output directory: {output_dir}")
    
    # Confirm start
    input("\n📊 Press Enter to start evaluation (or Ctrl+C to cancel)...")
    
    # Run complete evaluation
    start_time = time.time()
    collectors, summaries = runner.run_complete_evaluation()
    duration = time.time() - start_time
    
    print(f"\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"⏱️ Total time: {duration:.1f} seconds")
    print(f"📁 All results saved in: {output_dir}/")
    print(f"📊 Ready for research analysis and paper writing!")
    
    # Show final summary
    if summaries:
        avg_latency = np.mean([s['avg_latency'] for s in summaries])
        avg_pdr = np.mean([s['avg_pdr'] for s in summaries])
        print(f"\n📈 OVERALL RESULTS:")
        print(f"  Average Latency: {avg_latency:.2f} ms")
        print(f"  Average PDR: {avg_pdr:.1f} %")

if __name__ == "__main__":
    main()
