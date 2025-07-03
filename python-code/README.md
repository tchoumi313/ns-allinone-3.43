# ARPMEC Protocol Implementation

This repository contains a Python implementation of the **ARPMEC (Adaptive Mobile Edge Computing-based Routing Protocol)** for IoT networks, based on the research paper:

> "ARPMEC: an adaptive mobile edge computing-based routing protocol for IoT networks"  
> by Miguel Landry Foko Sindjoung, Mthulisi Velempini, and Vianney Kengne Tchendji (2024)  
> Published in Cluster Computing, Springer

## Overview

ARPMEC is a two-phase adaptive routing protocol designed for Mobile Edge Computing (MEC)-based IoT networks that focuses on energy efficiency and Quality of Service (QoS) provisioning.

### Key Features

- **Link Quality Prediction**: Uses RSSI and PDR metrics with machine learning prediction
- **Adaptive Clustering**: Dynamic cluster formation based on link quality estimation
- **Energy-Efficient Routing**: Optimized for IoT device energy consumption
- **Mobility Support**: Handles node movement through cluster re-election
- **MEC Integration**: Designed for Mobile Edge Computing architectures

## Implementation Structure

### Core Components

1. **`arpmec_implementation.py`** - Main protocol implementation
   - `Node` class: Represents IoT devices with energy modeling
   - `ARPMECProtocol` class: Implements clustering and routing algorithms
   - Link quality estimation and prediction functions
   - Energy consumption model (Equation 8 from paper)

2. **`arpmec_simulation.py`** - Comprehensive simulation framework
   - Performance analysis tools
   - Comparison with baseline protocols
   - Visualization and reporting capabilities
   - Parameter sensitivity analysis

3. **`arpmec_demo.py`** - Interactive demonstration script
   - Step-by-step protocol demonstration
   - Network visualization
   - Parameter sensitivity analysis

## Algorithms Implemented

### Algorithm 1: Adaptive Routing Protocol (Main Framework)
- Coordinates clustering and routing phases
- Handles protocol lifecycle management
- Implements energy monitoring and node state management

### Algorithm 2: Clustering using Link Quality Prediction
```python
def clustering_algorithm(self):
    # 1. HELLO message exchange for neighbor discovery
    # 2. Link quality estimation (RSSI + PDR)
    # 3. Machine learning-based link prediction
    # 4. JOIN decision making
    # 5. Cluster head election
    # 6. Cluster formation and cleanup
```

### Algorithm 3: Adaptive Data Routing
```python
def adaptive_routing_algorithm(self, rounds):
    # For each round:
    # 1. Check cluster head energy levels
    # 2. Perform cluster head operations or re-election
    # 3. Execute cluster member data transmission
    # 4. Update energy consumption
    # 5. Handle node mobility
```

## Energy Model

The implementation uses the energy model from Equation 8 in the paper:

```
E = Q × n(et + eamp × d²) + er × n
```

Where:
- `E`: Total energy consumed
- `et`: Transmission energy (0.03J)
- `er`: Reception energy (0.02J)  
- `eamp`: Amplification energy (0.01J)
- `d`: Distance between nodes
- `n`: Number of data items
- `Q`: Energy parameter

## Usage Examples

### Basic Usage

```python
from arpmec_implementation import Node, ARPMECProtocol, create_random_network

# Create network
nodes = create_random_network(num_nodes=50, area_size=1000)

# Initialize protocol
arpmec = ARPMECProtocol(nodes, num_channels=16, hello_messages=100)

# Run clustering
clusters = arpmec.clustering_algorithm()

# Run adaptive routing
arpmec.adaptive_routing_algorithm(rounds=200)

# Get statistics
stats = arpmec.get_network_statistics()
print(f"Network lifetime: {stats['alive_nodes']/stats['total_nodes']*100:.1f}%")
```

### Running Simulations

```python
from arpmec_simulation import ARPMECSimulation, SimulationParameters

# Configure simulation parameters
params = SimulationParameters(
    num_nodes_range=[50, 100, 200, 500],
    num_channels_options=[1, 4, 8, 16],
    hello_messages_options=[25, 50, 75, 100]
)

# Run comprehensive analysis
simulation = ARPMECSimulation(params)
energy_results = simulation.run_energy_analysis()
scalability_results = simulation.run_scalability_analysis()
```

### Quick Demonstration

```python
# Run the complete demonstration
python arpmec_demo.py
```

## Simulation Parameters (from Table 3)

| Parameter | Value |
|-----------|--------|
| Energy mitigation (n) | 2 ≤ n ≤ 4 |
| Number of items (D) | 1 ≤ D ≤ 10,000 |
| Number of channels (C) | 1, 4, 8, 16 |
| Number of nodes (N) | 1 ≤ N ≤ 500 |
| Number of rounds (T) | 1 ≤ T ≤ 200 |
| HELLO messages (R) | 25, 50, 75, 100 |
| Communication range | 1 km |
| Transmission energy (et) | 0.03J |
| Reception energy (er) | 0.02J |
| Amplification energy (eamp) | 0.01J |

## Performance Metrics

The implementation evaluates the following metrics:

- **Energy Consumption**: Total and per-node energy usage
- **Network Lifetime**: Percentage of nodes remaining alive
- **Clustering Efficiency**: Number and size of clusters formed
- **Protocol Overhead**: Energy consumed in control messages
- **Scalability**: Performance with varying network sizes

## Key Findings from Paper

1. **Energy Efficiency**: ARPMEC outperforms NESEPRIN and ABBPWHN for large data volumes
2. **Scalability**: Protocol complexity is O(N) for clustering and O(TD) for routing
3. **Adaptivity**: Successfully handles node mobility through cluster re-election
4. **Link Quality**: ML-based prediction improves routing decisions

## Differences from ns-3 Implementation

⚠️ **Important Note**: The ARPMEC module found in ns-3.43 is **NOT** the real ARPMEC protocol described in this paper. It appears to be a renamed version of AODV. This Python implementation is based on the actual research paper algorithms.

## Requirements

```python
numpy
matplotlib
pandas
seaborn
dataclasses  # Python 3.7+
```

## Installation

```bash
git clone <repository-url>
cd arpmec-implementation
pip install -r requirements.txt
```

## File Structure

```
arpmec-implementation/
├── arpmec_implementation.py    # Core protocol implementation
├── arpmec_simulation.py        # Simulation framework
├── arpmec_demo.py             # Demonstration script
├── README.md                  # This file
├── requirements.txt           # Dependencies
└── results/                   # Generated simulation results
    ├── energy_analysis.csv
    ├── scalability_analysis.csv
    ├── performance_report.txt
    └── network_topology.png
```

## Research Paper Reference

```bibtex
@article{fokosindjoung2024arpmec,
  title={ARPMEC: an adaptive mobile edge computing-based routing protocol for IoT networks},
  author={Foko Sindjoung, Miguel Landry and Velempini, Mthulisi and Tchendji, Vianney Kengne},
  journal={Cluster Computing},
  year={2024},
  publisher={Springer},
  doi={10.1007/s10586-024-04450-2}
}
```

## Contact Information

For questions about the original research:
- miguel.fokosindjoung@ul.ac.za
- mthulisi.velempini@ul.ac.za  
- vianneykengne@gmail.com

## License

This implementation is provided for research and educational purposes. Please cite the original paper when using this code in your research.

## Future Enhancements

- [ ] Integration with real MEC servers
- [ ] Security mechanisms implementation
- [ ] Fault tolerance improvements
- [ ] Federated learning for link quality prediction
- [ ] Real-time visualization dashboard
- [ ] ns-3 integration module

## Contributing

Contributions are welcome! Please ensure:
1. Code follows the paper's algorithmic specifications
2. Energy model accuracy is maintained
3. Performance metrics align with paper results
4. Documentation is comprehensive

---

**Note**: This is an independent implementation based on the research paper. For the official implementation, please contact the paper authors.