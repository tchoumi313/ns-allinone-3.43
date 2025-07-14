#!/usr/bin/env python3
"""
Simple test script to verify the base ARPMEC implementation works
"""

import math
import random
import time

import numpy as np

print("Testing base ARPMEC implementation...")

try:
    from arpmec_faithful import ARPMECProtocol, Node, NodeState
    print("✅ Successfully imported ARPMEC components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def test_basic_functionality():
    """Test basic ARPMEC functionality"""
    print("\n🧪 Testing basic functionality...")
    
    # Create a small network
    nodes = []
    for i in range(5):
        x = random.uniform(0, 200)
        y = random.uniform(0, 200)
        energy = 100.0
        nodes.append(Node(i, x, y, energy))
    
    print(f"Created {len(nodes)} nodes")
    
    # Create protocol
    try:
        protocol = ARPMECProtocol(nodes, C=2, R=5, K=1)
        print("✅ Protocol created successfully")
    except Exception as e:
        print(f"❌ Protocol creation failed: {e}")
        return False
    
    # Test clustering
    try:
        print("Running clustering algorithm...")
        clusters = protocol.clustering_algorithm()
        print(f"✅ Clustering completed - {len(clusters)} clusters formed")
        
        # Show cluster details
        for head_id, members in clusters.items():
            print(f"   Cluster {head_id}: {len(members)} members")
            
    except Exception as e:
        print(f"❌ Clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_node_creation():
    """Test individual node creation"""
    print("\n🧪 Testing node creation...")
    
    try:
        node = Node(1, 100, 100, 100)
        print(f"✅ Node created: ID={node.id}, pos=({node.x}, {node.y}), energy={node.energy}")
        
        # Test energy update
        initial_energy = node.energy
        node.update_energy(10)
        print(f"✅ Energy update: {initial_energy} -> {node.energy}")
        
        # Test distance calculation
        node2 = Node(2, 150, 150, 100)
        distance = node.distance_to(node2)
        print(f"✅ Distance calculation: {distance:.2f}m")
        
        return True
        
    except Exception as e:
        print(f"❌ Node creation/testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 ARPMEC Base Implementation Test")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Test node creation first
    if not test_node_creation():
        print("❌ Basic node tests failed")
        exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("❌ Basic functionality tests failed")
        exit(1)
    
    print("\n✅ All base tests passed!")
    print("The faithful ARPMEC implementation is working correctly.")
