#!/usr/bin/env python3
"""
QUICK ARPMEC DATA COLLECTION TEST
================================

Quick test to verify data collection and plotting works correctly
"""

import sys
import os
sys.path.append('/home/donsoft/ns-allinone-3.43/python-code/arpmec_project/faithful_version')

from arpmec_faithful import ARPMECProtocol, Node
from arpmec_data_collector import ARPMECDataCollector
import numpy as np

def quick_test():
    print("🧪 Quick ARPMEC Data Collection Test")
    print("="*50)
    
    # Create small network for testing
    nodes = []
    for i in range(10):
        x = np.random.uniform(50, 500)
        y = np.random.uniform(50, 500)
        energy = np.random.uniform(90, 110)
        nodes.append(Node(i, x, y, energy))
    
    # Initialize protocol
    protocol = ARPMECProtocol(nodes, C=2, R=3, K=2)
    clusters = protocol.clustering_algorithm()
    
    print(f"✅ Created network: {len(nodes)} nodes, {len(clusters)} clusters")
    
    # Initialize data collector
    collector = ARPMECDataCollector(protocol, "test_results")
    
    # Run a few rounds
    print("⏳ Running test simulation...")
    for round_num in range(20):
        # Simulate some protocol activity
        protocol.current_time_slot = round_num
        protocol._generate_inter_cluster_traffic()
        protocol._generate_mec_tasks()
        protocol._process_mec_servers()
        
        # Collect data
        collector.collect_round_data(round_num)
        
        if round_num % 5 == 0:
            print(f"  Round {round_num} completed")
    
    print("📊 Generating test plots...")
    collector.generate_performance_plots(save_plots=True)
    
    print("📁 Exporting test data...")
    collector.export_data('json')
    collector.export_data('csv')
    
    print("📋 Generating summary...")
    summary = collector.generate_summary_report()
    
    print("\n✅ TEST COMPLETED SUCCESSFULLY!")
    print("📁 Check test_results/ folder for output")
    
    return True

if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
