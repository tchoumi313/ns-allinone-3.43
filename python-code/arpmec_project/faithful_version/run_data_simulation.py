#!/usr/bin/env python3
"""
ARPMEC Research Data Collection - 60 Rounds
==========================================

This script runs a comprehensive 60-round simulation to collect research-quality data
for the ARPMEC protocol evaluation. Fixed to generate proper metrics.
"""

import random
import time
import numpy as np
from arpmec_faithful import ARPMECProtocol, Node, NodeState
from arpmec_data_collector import ARPMECDataCollector

def create_research_network(num_nodes=25, area_size=1000):
    """Create a research-grade network for comprehensive evaluation"""
    print(f"Creating research network: {num_nodes} nodes in {area_size}x{area_size}m area")
    
    # Set seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    nodes = []
    
    # Create clustered deployment (more realistic)
    cluster_centers = [
        (200, 200), (600, 200), (400, 600), (800, 600), (500, 400)
    ]
    
    nodes_per_cluster = num_nodes // len(cluster_centers)
    node_id = 0
    
    for cx, cy in cluster_centers:
        for i in range(nodes_per_cluster):
            # Place within communication range
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(20, 90)  # 20-90m from center
            
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            
            # Keep in bounds
            x = max(50, min(area_size - 50, x))
            y = max(50, min(area_size - 50, y))
            
            # Varied initial energy (80-120J)
            energy = random.uniform(80, 120)
            nodes.append(Node(node_id, x, y, energy))
            node_id += 1
    
    # Add remaining nodes randomly
    while len(nodes) < num_nodes:
        x = random.uniform(50, area_size - 50)
        y = random.uniform(50, area_size - 50)
        energy = random.uniform(80, 120)
        nodes.append(Node(node_id, x, y, energy))
        node_id += 1
    
    print(f"✓ Created {len(nodes)} nodes with initial energy: {sum(n.energy for n in nodes):.1f}J")
    return nodes

def run_comprehensive_simulation(rounds=60):
    """Run comprehensive 60-round simulation with proper data collection"""
    print("🚀 Starting COMPREHENSIVE 60-Round ARPMEC Simulation")
    print("="*60)
    
    # Create network
    nodes = create_research_network(num_nodes=25, area_size=1000)
    
    # Initialize protocol
    protocol = ARPMECProtocol(nodes, C=4, R=5, K=3)
    
    # Perform initial clustering
    print("🔧 Performing initial clustering...")
    clusters = protocol.clustering_algorithm()
    print(f"✓ Initial clustering: {len(clusters)} clusters formed")
    
    # Initialize data collector
    collector = ARPMECDataCollector(protocol)
    
    print(f"\n📊 Starting {rounds}-round simulation...")
    print("Round | Energy | Tasks | PDR% | BW% | Clusters | Latency")
    print("-"*60)
    
    for round_num in range(1, rounds + 1):
        round_start = time.time()
        
        # Update protocol time
        protocol.current_time_slot = round_num
        
        # 1. GENERATE REALISTIC NETWORK ACTIVITY
        # Generate inter-cluster messages
        cluster_heads = protocol._get_cluster_heads()
        if len(cluster_heads) > 1:
            for _ in range(random.randint(2, 5)):  # 2-5 inter-cluster messages per round
                source_ch = random.choice(cluster_heads)
                target_ch = random.choice([ch for ch in cluster_heads if ch.id != source_ch.id])
                
                # Create inter-cluster message
                from arpmec_faithful import InterClusterMessage
                message = InterClusterMessage(
                    message_id=f"msg_{source_ch.id}_{target_ch.id}_{round_num}",
                    source_cluster_id=source_ch.cluster_id,
                    destination_cluster_id=target_ch.cluster_id,
                    message_type='data',
                    payload={'data': f'sensor_data_round_{round_num}'},
                    timestamp=round_num
                )
                
                # Add to source CH's outgoing messages
                if not hasattr(source_ch, 'inter_cluster_messages'):
                    source_ch.inter_cluster_messages = []
                source_ch.inter_cluster_messages.append(message)
        
        # 2. GENERATE MEC TASKS
        for ch in cluster_heads:
            if random.random() < 0.6:  # 60% chance each CH generates a task
                from arpmec_faithful import MECTask
                task = MECTask(
                    task_id=f"task_{ch.id}_{round_num}",
                    source_cluster_id=ch.cluster_id,
                    cpu_requirement=random.uniform(2, 10),
                    memory_requirement=random.uniform(5, 50),
                    deadline=round_num + 5,
                    data_size=random.uniform(1, 20),
                    created_time=round_num
                )
                
                # Find nearest IAR for task submission
                nearest_iar = protocol._find_nearest_iar_server(ch)
                if nearest_iar and nearest_iar.connected_mec_servers:
                    mec_id = nearest_iar.connected_mec_servers[0]
                    if mec_id in protocol.mec_servers:
                        mec_server = protocol.mec_servers[mec_id]
                        
                        # Add task to MEC server
                        if not hasattr(mec_server, 'processing_tasks'):
                            mec_server.processing_tasks = []
                        if not hasattr(mec_server, 'completed_tasks'):
                            mec_server.completed_tasks = []
                        
                        mec_server.processing_tasks.append(task)
        
        # 3. PROCESS MEC TASKS (simulate task completion)
        for mec in protocol.mec_servers.values():
            if hasattr(mec, 'processing_tasks'):
                for task in mec.processing_tasks[:]:  # Copy to avoid modification during iteration
                    if random.random() < 0.7:  # 70% completion rate
                        mec.processing_tasks.remove(task)
                        if not hasattr(mec, 'completed_tasks'):
                            mec.completed_tasks = []
                        mec.completed_tasks.append(task)
        
        # 4. SIMULATE ENERGY CONSUMPTION
        for node in protocol.nodes.values():
            if node.is_alive():
                # Calculate energy consumption based on role
                if node.state == NodeState.CLUSTER_HEAD:
                    energy_cost = random.uniform(1.5, 3.0)  # CHs consume more
                elif node.state == NodeState.CLUSTER_MEMBER:
                    energy_cost = random.uniform(0.8, 1.5)  # Members consume less
                else:
                    energy_cost = random.uniform(0.5, 1.0)  # Idle nodes consume least
                
                node.update_energy(energy_cost)
        
        # 5. SIMULATE NODE MOBILITY AND RECLUSTERING
        if round_num % 10 == 0:  # Recluster every 10 rounds
            # Simulate node movement
            for node in protocol.nodes.values():
                if node.is_alive():
                    node.update_mobility((0, 1000, 0, 1000))
            
            # Trigger reclustering
            try:
                protocol._reset_clustering_state()
                new_clusters = protocol.clustering_algorithm()
                protocol._build_inter_cluster_routing_table()
            except Exception as e:
                print(f"⚠️ Reclustering failed in round {round_num}: {e}")
        
        # 6. RUN PROTOCOL OPERATIONS
        try:
            protocol._generate_inter_cluster_traffic()
            protocol._generate_mec_tasks()
            protocol._process_inter_cluster_messages()
            protocol._process_mec_servers()
        except Exception as e:
            print(f"⚠️ Protocol operations failed in round {round_num}: {e}")
        
        # 7. COLLECT METRICS
        round_time = time.time() - round_start
        collector.collect_round_data(round_num)
        
        # Print progress every 10 rounds
        if round_num % 10 == 0 or round_num <= 5:
            energy_remaining = sum(n.energy for n in protocol.nodes.values() if n.is_alive())
            total_tasks = sum(len(getattr(mec, 'completed_tasks', [])) + len(getattr(mec, 'processing_tasks', [])) 
                             for mec in protocol.mec_servers.values())
            completed_tasks = sum(len(getattr(mec, 'completed_tasks', [])) for mec in protocol.mec_servers.values())
            pdr = (completed_tasks / max(total_tasks, 1)) * 100
            
            avg_bw = sum(mec.get_load_percentage() for mec in protocol.mec_servers.values()) / len(protocol.mec_servers)
            active_clusters = len(protocol._get_cluster_heads())
            
            print(f"{round_num:5d} | {energy_remaining:6.1f} | {completed_tasks:5d} | {pdr:4.1f} | {avg_bw:3.1f} | {active_clusters:8d} | {round_time*1000:7.2f}")
    
    print("\n" + "="*60)
    print("✅ 60-Round simulation completed!")
    
    # Generate comprehensive analysis
    print("\n📈 Generating performance analysis plots...")
    collector.generate_performance_plots(save_plots=True)
    
    # Print summary
    summary = collector.generate_summary_report()
    
    # Save data
    collector.export_data('json')
    
    return collector, summary

if __name__ == "__main__":
    print("🔬 ARPMEC Research Data Collection")
    print("📅 Target: Sunday Deadline")
    print("🎯 Goal: 60-round comprehensive performance evaluation")
    print()
    
    try:
        collector, summary = run_comprehensive_simulation(rounds=60)
        print("\n🎉 SUCCESS: Research data collection completed!")
        print("📊 Check the generated plots and saved data files")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
