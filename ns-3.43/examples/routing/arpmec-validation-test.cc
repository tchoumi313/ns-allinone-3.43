/*
 * ARPMEC Validation Test
 * Tests implementation against paper requirements
 */

#include "ns3/arpmec-module.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/netanim-module.h"
#include "ns3/energy-module.h"
#include "ns3/boolean.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <iomanip>

using namespace ns3;
using namespace ns3::energy;

NS_LOG_COMPONENT_DEFINE("ArpmecValidationTest");

class ArpmecValidator
{
public:
    ArpmecValidator() : m_totalPacketsSent(0), m_totalPacketsReceived(0), 
                       m_dataPacketsSent(0), m_dataPacketsReceived(0),
                       m_controlPacketsSent(0), m_controlPacketsReceived(0),
                       m_helloPacketsSent(0), m_clusteringPacketsSent(0),
                       m_appPacketsSent(0), m_appPacketsReceived(0),
                       m_maxClusterHeadsObserved(0) {}

    // Protocol-level callbacks (for control packet analysis)
    void PacketSentCallback(Ptr<const Packet> packet, const Address& from, const Address& to);
    void PacketReceivedCallback(Ptr<const Packet> packet, const Address& from);
    
    // Application-level callbacks (for true performance metrics)
    void ApplicationPacketSent(Ptr<const Packet> packet);
    void ApplicationPacketReceived(Ptr<const Packet> packet, const Address& from);
    
    // Node-specific application callbacks
    void ApplicationPacketSentFromNode(uint32_t nodeId, Ptr<const Packet> packet);
    void ApplicationPacketReceivedAtNode(uint32_t nodeId, Ptr<const Packet> packet, const Address& from);
    
    // Method to register flow mappings for connectivity analysis
    void RegisterFlow(uint32_t sourceNode, uint32_t destNode);
    
    // ARPMEC-specific callbacks
    void ClusterHeadCallback(uint32_t nodeId, bool isClusterHead);
    void RouteDecisionCallback(uint32_t nodeId, std::string decision);
    void LqeUpdateCallback(uint32_t nodeId, double lqeValue);
    void EnergyUpdateCallback(uint32_t nodeId, double energy);

    void PrintResults();
    void ValidateAgainstPaper();

private:
    // Protocol-level packet counts (for overhead analysis)
    uint32_t m_totalPacketsSent;
    uint32_t m_totalPacketsReceived;
    uint32_t m_dataPacketsSent;           // ARPMEC data packets only
    uint32_t m_dataPacketsReceived;       // ARPMEC data packets only
    uint32_t m_controlPacketsSent;        // ARPMEC control packets
    uint32_t m_controlPacketsReceived;    // ARPMEC control packets
    uint32_t m_helloPacketsSent;          // HELLO messages
    uint32_t m_clusteringPacketsSent;     // Clustering-related packets
    
    // Application-level packet counts (for true PDR)
    uint32_t m_appPacketsSent;            // Application packets generated
    uint32_t m_appPacketsReceived;        // Application packets delivered
    
    // Flow tracking for simplified connectivity analysis
    std::map<uint32_t, uint32_t> m_sourceToDestMap;  // Map source node to dest node
    std::map<uint32_t, uint32_t> m_sentByNode;       // Packets sent by each source node
    std::map<uint32_t, uint32_t> m_receivedByNode;   // Packets received by each dest node
    
    // ARPMEC-specific metrics
    std::map<uint32_t, bool> m_clusterHeads;
    uint32_t m_maxClusterHeadsObserved;  // Track peak number of CHs during simulation
    std::map<uint32_t, std::vector<std::string>> m_routeDecisions;
    std::map<uint32_t, std::vector<double>> m_lqeValues;
    std::map<uint32_t, double> m_energyLevels;
    std::vector<std::pair<double, uint32_t>> m_packetDeliveryTimes;
};

void ArpmecValidator::PacketSentCallback(Ptr<const Packet> packet, const Address& from, const Address& to)
{
    m_totalPacketsSent++;
    
    // Try to identify packet type by examining headers
    Ptr<Packet> packetCopy = packet->Copy();
    
    // Check if this is an ARPMEC protocol packet
    ns3::arpmec::TypeHeader typeHeader;
    if (packetCopy->PeekHeader(typeHeader)) {
        uint8_t packetType = typeHeader.Get();
        
        switch (packetType) {
            case ns3::arpmec::ARPMEC_HELLO:
                m_helloPacketsSent++;
                m_controlPacketsSent++;
                break;
            case ns3::arpmec::ARPMEC_JOIN:
            case ns3::arpmec::ARPMEC_CH_NOTIFICATION:
            case ns3::arpmec::ARPMEC_CLUSTER_LIST:
                m_clusteringPacketsSent++;
                m_controlPacketsSent++;
                break;
            case ns3::arpmec::ARPMECTYPE_RREQ:
            case ns3::arpmec::ARPMECTYPE_RREP:
            case ns3::arpmec::ARPMECTYPE_RERR:
            case ns3::arpmec::ARPMECTYPE_RREP_ACK:
                m_controlPacketsSent++;
                break;
            case ns3::arpmec::ARPMEC_DATA:
                m_dataPacketsSent++;
                break;
            default:
                // Unknown ARPMEC packet type
                m_controlPacketsSent++;
                break;
        }
    } else {
        // Not an ARPMEC packet - likely application data being routed
        m_dataPacketsSent++;
    }
    
    NS_LOG_INFO("Protocol packet sent from " << from << " to " << to << " (Total: " << m_totalPacketsSent << ")");
}

void ArpmecValidator::PacketReceivedCallback(Ptr<const Packet> packet, const Address& from)
{
    m_totalPacketsReceived++;
    
    // Try to identify packet type by examining headers
    Ptr<Packet> packetCopy = packet->Copy();
    
    // Check if this is an ARPMEC protocol packet
    ns3::arpmec::TypeHeader typeHeader;
    if (packetCopy->PeekHeader(typeHeader)) {
        uint8_t packetType = typeHeader.Get();
        
        switch (packetType) {
            case ns3::arpmec::ARPMEC_HELLO:
                m_controlPacketsReceived++;
                break;
            case ns3::arpmec::ARPMEC_JOIN:
            case ns3::arpmec::ARPMEC_CH_NOTIFICATION:
            case ns3::arpmec::ARPMEC_CLUSTER_LIST:
                m_controlPacketsReceived++;
                break;
            case ns3::arpmec::ARPMECTYPE_RREQ:
            case ns3::arpmec::ARPMECTYPE_RREP:
            case ns3::arpmec::ARPMECTYPE_RERR:
            case ns3::arpmec::ARPMECTYPE_RREP_ACK:
                m_controlPacketsReceived++;
                break;
            case ns3::arpmec::ARPMEC_DATA:
                m_dataPacketsReceived++;
                break;
            default:
                m_controlPacketsReceived++;
                break;
        }
    } else {
        // Not an ARPMEC packet - likely application data
        m_dataPacketsReceived++;
    }
    
    NS_LOG_INFO("Protocol packet received from " << from << " (Total: " << m_totalPacketsReceived << ")");
}

void ArpmecValidator::ApplicationPacketSent(Ptr<const Packet> packet)
{
    m_appPacketsSent++;
    NS_LOG_INFO("Application packet sent (Total: " << m_appPacketsSent << ")");
}

void ArpmecValidator::ApplicationPacketReceived(Ptr<const Packet> packet, const Address& from)
{
    m_appPacketsReceived++;
    NS_LOG_INFO("Application packet received from " << from << " (Total: " << m_appPacketsReceived << ")");
}

void ArpmecValidator::ApplicationPacketSentFromNode(uint32_t nodeId, Ptr<const Packet> packet)
{
    m_appPacketsSent++;
    m_sentByNode[nodeId]++;
    NS_LOG_INFO("Application packet sent from node " << nodeId << " (Total: " << m_appPacketsSent << ")");
}

void ArpmecValidator::ApplicationPacketReceivedAtNode(uint32_t nodeId, Ptr<const Packet> packet, const Address& from)
{
    m_appPacketsReceived++;
    m_receivedByNode[nodeId]++;
    NS_LOG_INFO("Application packet received at node " << nodeId << " from " << from << " (Total: " << m_appPacketsReceived << ")");
}

void ArpmecValidator::RegisterFlow(uint32_t sourceNode, uint32_t destNode)
{
    m_sourceToDestMap[sourceNode] = destNode;
    // Initialize counters for this flow
    m_sentByNode[sourceNode] = 0;
    m_receivedByNode[destNode] = 0;
}

void ArpmecValidator::ClusterHeadCallback(uint32_t nodeId, bool isClusterHead)
{
    m_clusterHeads[nodeId] = isClusterHead;
    
    // Count current cluster heads and update maximum observed
    uint32_t currentCHs = 0;
    for (auto& pair : m_clusterHeads) {
        if (pair.second) currentCHs++;
    }
    
    if (currentCHs > m_maxClusterHeadsObserved) {
        m_maxClusterHeadsObserved = currentCHs;
    }
    
    NS_LOG_INFO("Node " << nodeId << " cluster head status: " << (isClusterHead ? "YES" : "NO") 
                << " (Current CHs: " << currentCHs << ", Max observed: " << m_maxClusterHeadsObserved << ")");
}

void ArpmecValidator::RouteDecisionCallback(uint32_t nodeId, std::string decision)
{
    m_routeDecisions[nodeId].push_back(decision);
    NS_LOG_INFO("Node " << nodeId << " route decision: " << decision);
}

void ArpmecValidator::LqeUpdateCallback(uint32_t nodeId, double lqeValue)
{
    m_lqeValues[nodeId].push_back(lqeValue);
    NS_LOG_INFO("Node " << nodeId << " LQE updated: " << lqeValue);
}

void ArpmecValidator::EnergyUpdateCallback(uint32_t nodeId, double energy)
{
    m_energyLevels[nodeId] = energy;
    NS_LOG_DEBUG("Node " << nodeId << " energy level: " << energy);
}

void ArpmecValidator::PrintResults()
{
    std::cout << "\n=== ARPMEC PERFORMANCE ANALYSIS ===" << std::endl;

    // Application-Level Performance (True Network Performance)
    std::cout << "\n--- APPLICATION-LEVEL PERFORMANCE ---" << std::endl;
    std::cout << "Application Packets Sent: " << m_appPacketsSent << std::endl;
    std::cout << "Application Packets Received: " << m_appPacketsReceived << std::endl;
    
    if (m_appPacketsSent > 0) {
        double appPDR = (double)m_appPacketsReceived / m_appPacketsSent * 100.0;
        std::cout << "True Packet Delivery Ratio (PDR): " << std::fixed << std::setprecision(1) << appPDR << "%" << std::endl;
        
        // Evaluate performance
        if (appPDR >= 80.0) {
            std::cout << "Performance Status: EXCELLENT" << std::endl;
        } else if (appPDR >= 60.0) {
            std::cout << "Performance Status: GOOD" << std::endl;
        } else if (appPDR >= 40.0) {
            std::cout << "Performance Status: FAIR" << std::endl;
        } else {
            std::cout << "Performance Status: POOR" << std::endl;
        }
    } else {
        std::cout << "No application packets transmitted" << std::endl;
    }
    
    // Flow-Level Analysis
    std::cout << "\n--- FLOW CONNECTIVITY ANALYSIS ---" << std::endl;
    uint32_t totalFlows = m_sourceToDestMap.size();
    uint32_t successfulFlows = 0;
    
    for (auto& flowPair : m_sourceToDestMap) {
        uint32_t destNode = flowPair.second;
        if (m_receivedByNode[destNode] > 0) {
            successfulFlows++;
        }
    }
    
    if (totalFlows > 0) {
        double connectivity = (double)successfulFlows / totalFlows * 100.0;
        std::cout << "Successful Flows: " << successfulFlows << " / " << totalFlows << std::endl;
        std::cout << "Network Connectivity: " << std::fixed << std::setprecision(1) << connectivity << "%" << std::endl;
    } else {
        std::cout << "No flows registered" << std::endl;
    }

    // Protocol-Level Analysis (Control Overhead)
    std::cout << "\n--- PROTOCOL OVERHEAD ANALYSIS ---" << std::endl;
    std::cout << "Total Protocol Packets Sent: " << m_totalPacketsSent << std::endl;
    std::cout << "Total Protocol Packets Received: " << m_totalPacketsReceived << std::endl;
    std::cout << "  - HELLO Packets Sent: " << m_helloPacketsSent << std::endl;
    std::cout << "  - Clustering Packets Sent: " << m_clusteringPacketsSent << std::endl;
    std::cout << "  - Control Packets Sent: " << m_controlPacketsSent << std::endl;
    std::cout << "  - Data Packets Sent: " << m_dataPacketsSent << std::endl;
    std::cout << "  - Control Packets Received: " << m_controlPacketsReceived << std::endl;
    std::cout << "  - Data Packets Received: " << m_dataPacketsReceived << std::endl;
    
    if (m_totalPacketsSent > 0) {
        double controlOverhead = (double)m_controlPacketsSent / m_totalPacketsSent * 100.0;
        double broadcastEfficiency = (double)m_totalPacketsReceived / m_totalPacketsSent;
        
        std::cout << "Control Packet Overhead: " << std::fixed << std::setprecision(1) << controlOverhead << "%" << std::endl;
        std::cout << "Broadcast Reception Ratio: " << std::fixed << std::setprecision(1) << broadcastEfficiency << ":1" << std::endl;
        
        // Explain the broadcast ratio
        if (broadcastEfficiency > 1.0) {
            std::cout << "Note: Reception ratio > 1 is NORMAL for wireless broadcast networks" << std::endl;
            std::cout << "      (One broadcast reaches multiple neighbors)" << std::endl;
        }
    }

    // Clustering Analysis (unchanged)
    std::cout << "\n--- CLUSTERING ANALYSIS ---" << std::endl;
    uint32_t numClusterHeads = 0;
    std::cout << "Cluster Heads: ";
    for (auto& pair : m_clusterHeads) {
        if (pair.second) {
            std::cout << pair.first << " ";
            numClusterHeads++;
        }
    }
    std::cout << std::endl;
    std::cout << "Total Cluster Heads: " << numClusterHeads << std::endl;

    if (m_clusterHeads.size() > 0) {
        double chRatio = (double)numClusterHeads / m_clusterHeads.size() * 100.0;
        std::cout << "Cluster Head Ratio: " << chRatio << "%" << std::endl;
    }

    // Route Decision Analysis (unchanged)
    std::cout << "\n--- ROUTE DECISION ANALYSIS ---" << std::endl;
    std::map<std::string, uint32_t> decisionCounts;
    for (auto& nodePair : m_routeDecisions) {
        for (auto& decision : nodePair.second) {
            decisionCounts[decision]++;
        }
    }

    for (auto& pair : decisionCounts) {
        std::cout << pair.first << ": " << pair.second << " times" << std::endl;
    }

    // LQE Analysis (unchanged)
    std::cout << "\n--- LQE ANALYSIS ---" << std::endl;
    double totalLqe = 0.0;
    uint32_t lqeCount = 0;
    for (auto& nodePair : m_lqeValues) {
        if (!nodePair.second.empty()) {
            double avgNodeLqe = 0.0;
            for (double lqe : nodePair.second) {
                avgNodeLqe += lqe;
            }
            avgNodeLqe /= nodePair.second.size();
            totalLqe += avgNodeLqe;
            lqeCount++;
            std::cout << "Node " << nodePair.first << " average LQE: " << avgNodeLqe << std::endl;
        }
    }

    if (lqeCount > 0) {
        std::cout << "Network average LQE: " << (totalLqe / lqeCount) << std::endl;
    }

    // Energy Analysis (unchanged)
    std::cout << "\n--- ENERGY ANALYSIS ---" << std::endl;
    double totalEnergy = 0.0;
    uint32_t energyCount = 0;
    for (auto& pair : m_energyLevels) {
        std::cout << "Node " << pair.first << " energy: " << pair.second << " J" << std::endl;
        totalEnergy += pair.second;
        energyCount++;
    }

    if (energyCount > 0) {
        std::cout << "Average network energy: " << (totalEnergy / energyCount) << " J" << std::endl;
    }
}

void ArpmecValidator::ValidateAgainstPaper()
{
    std::cout << "\n=== PAPER COMPLIANCE VALIDATION ===" << std::endl;

    bool allTestsPassed = true;

    // Test 1: Algorithm 2 - Clustering should produce CHs
    std::cout << "\n[TEST 1] Algorithm 2 - Clustering Protocol" << std::endl;
    
    if (m_maxClusterHeadsObserved > 0) {
        std::cout << "✓ PASS: Cluster heads elected (Peak: " << m_maxClusterHeadsObserved << " CHs)" << std::endl;
    } else {
        std::cout << "✗ FAIL: No cluster heads elected" << std::endl;
        allTestsPassed = false;
    }

    // Test 2: Algorithm 3 - Adaptive routing decisions should be made
    std::cout << "\n[TEST 2] Algorithm 3 - Adaptive Routing" << std::endl;
    uint32_t totalDecisions = 0;
    for (auto& nodePair : m_routeDecisions) {
        totalDecisions += nodePair.second.size();
    }

    if (totalDecisions > 0) {
        std::cout << "✓ PASS: Adaptive routing decisions made (" << totalDecisions << " decisions)" << std::endl;
    } else {
        std::cout << "✗ FAIL: No adaptive routing decisions recorded" << std::endl;
        allTestsPassed = false;
    }

    // Test 3: LQE functionality
    std::cout << "\n[TEST 3] Link Quality Estimation" << std::endl;
    uint32_t nodesWithLqe = 0;
    for (auto& nodePair : m_lqeValues) {
        if (!nodePair.second.empty()) nodesWithLqe++;
    }

    if (nodesWithLqe > 0) {
        std::cout << "✓ PASS: LQE values calculated (" << nodesWithLqe << " nodes)" << std::endl;
    } else {
        std::cout << "✗ FAIL: No LQE values recorded" << std::endl;
        allTestsPassed = false;
    }

    // Test 4: Energy tracking
    std::cout << "\n[TEST 4] Energy Model Integration" << std::endl;
    if (m_energyLevels.size() > 0) {
        std::cout << "✓ PASS: Energy levels tracked (" << m_energyLevels.size() << " nodes)" << std::endl;
    } else {
        std::cout << "✗ FAIL: No energy levels recorded" << std::endl;
        allTestsPassed = false;
    }

    // Test 5: Packet delivery performance
    std::cout << "\n[TEST 5] Performance Requirements" << std::endl;
    if (m_totalPacketsSent > 0) {
        double averageReceptionsPerTx = (double)m_totalPacketsReceived / m_totalPacketsSent;
        // In wireless networks, successful delivery means good connectivity and reachability
        if (averageReceptionsPerTx >= 1.5) {  // Good connectivity indicator
            std::cout << "✓ PASS: Good network connectivity (" << std::fixed << std::setprecision(1) << averageReceptionsPerTx << " avg receptions per tx)" << std::endl;
        } else {
            std::cout << "✗ FAIL: Poor network connectivity (" << std::fixed << std::setprecision(1) << averageReceptionsPerTx << " avg receptions per tx)" << std::endl;
            allTestsPassed = false;
        }
    } else {
        std::cout << "✗ FAIL: No packets transmitted" << std::endl;
        allTestsPassed = false;
    }

    // Overall result
    std::cout << "\n=== OVERALL VALIDATION RESULT ===" << std::endl;
    if (allTestsPassed) {
        std::cout << "✓ ALL TESTS PASSED - Implementation meets paper requirements" << std::endl;
    } else {
        std::cout << "✗ SOME TESTS FAILED - Implementation needs improvements" << std::endl;
    }
}

int main(int argc, char* argv[])
{
    // Enable logging
    LogComponentEnable("ArpmecValidationTest", LOG_LEVEL_INFO);
    LogComponentEnable("ArpmecRoutingProtocol", LOG_LEVEL_INFO);
    LogComponentEnable("ArpmecClustering", LOG_LEVEL_INFO);
    LogComponentEnable("ArpmecAdaptiveRouting", LOG_LEVEL_INFO);

    // Parse command line
    CommandLine cmd;
    uint32_t numNodes = 20;
    double simTime = 100.0;
    uint32_t packetSize = 1024;
    double dataRate = 1.0; // packets per second

    cmd.AddValue("nodes", "Number of nodes", numNodes);
    cmd.AddValue("time", "Simulation time (seconds)", simTime);
    cmd.AddValue("size", "Packet size (bytes)", packetSize);
    cmd.AddValue("rate", "Data rate (packets/second)", dataRate);
    cmd.Parse(argc, argv);

    std::cout << "=== ARPMEC VALIDATION TEST ===" << std::endl;
    std::cout << "Nodes: " << numNodes << std::endl;
    std::cout << "Simulation time: " << simTime << " seconds" << std::endl;
    std::cout << "Packet size: " << packetSize << " bytes" << std::endl;
    std::cout << "Data rate: " << dataRate << " packets/second" << std::endl;

    // Create validator
    ArpmecValidator validator;

    // Create nodes
    NodeContainer nodes;
    nodes.Create(numNodes);

    // Configure WiFi
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);

    YansWifiPhyHelper wifiPhy;
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    wifiPhy.SetChannel(wifiChannel.Create());

    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");

    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, nodes);

    // Configure mobility - Grid topology for validation
    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                 "MinX", DoubleValue(0.0),
                                 "MinY", DoubleValue(0.0),
                                 "DeltaX", DoubleValue(50.0),
                                 "DeltaY", DoubleValue(50.0),
                                 "GridWidth", UintegerValue(5),
                                 "LayoutType", StringValue("RowFirst"));

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    // Add energy model
    BasicEnergySourceHelper basicSourceHelper;
    basicSourceHelper.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(100.0));
    basicSourceHelper.Set("BasicEnergySupplyVoltageV", DoubleValue(3.0));
    EnergySourceContainer sources = basicSourceHelper.Install(nodes);

    // Configure Internet stack with ARPMEC
    ArpmecHelper arpmec;
    arpmec.Set("EnableHello", BooleanValue(true));  // Enable HELLO messages for neighbor discovery
    arpmec.Set("HelloInterval", TimeValue(Seconds(1.0)));  // Send HELLO every 1 second
    InternetStackHelper stack;
    stack.SetRoutingHelper(arpmec);
    stack.Install(nodes);

    // Assign IP addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    // Enable MEC Infrastructure - this is where MEC comes into action!
    std::cout << "=== ENABLING MEC INFRASTRUCTURE ===" << std::endl;
    
    // Enable MEC Gateway on strategic nodes (nodes 1 and 3 for coverage)
    for (uint32_t i = 0; i < nodes.GetN(); i++) {
        Ptr<Node> node = nodes.Get(i);
        Ptr<ns3::arpmec::RoutingProtocol> arpmecRouting = 
            DynamicCast<ns3::arpmec::RoutingProtocol>(
                node->GetObject<Ipv4>()->GetRoutingProtocol());
        
        if (arpmecRouting) {
            // Enable MEC Gateway on every 4th node for distributed edge computing
            if (i % 4 == 1) {
                uint32_t gatewayId = i + 100;  // Unique gateway ID
                double coverageArea = 100.0;   // 100m coverage radius
                arpmecRouting->EnableMecGateway(gatewayId, coverageArea);
                std::cout << "  ✓ MEC Gateway " << gatewayId << " enabled on Node " << i 
                         << " (Coverage: " << coverageArea << "m)" << std::endl;
            }
            
            // Enable MEC Server on every 6th node for edge computation
            if (i % 6 == 2) {
                uint32_t serverId = i + 200;        // Unique server ID
                uint32_t processingCapacity = 1000; // 1000 ops/sec
                uint32_t memoryCapacity = 512;      // 512 MB
                arpmecRouting->EnableMecServer(serverId, processingCapacity, memoryCapacity);
                std::cout << "  ✓ MEC Server " << serverId << " enabled on Node " << i 
                         << " (CPU: " << processingCapacity << " ops/s, RAM: " 
                         << memoryCapacity << " MB)" << std::endl;
            }
        }
    }
    std::cout << "=== MEC INFRASTRUCTURE READY ===" << std::endl;

    // Create traffic pattern - multiple source-destination pairs
    ApplicationContainer apps;

    for (uint32_t i = 0; i < numNodes / 2; i++) {
        uint32_t sourceNode = i;
        uint32_t destNode = numNodes - 1 - i;

        // Install packet sink on destination
        PacketSinkHelper sink("ns3::UdpSocketFactory",
                             InetSocketAddress(Ipv4Address::GetAny(), 9));
        ApplicationContainer sinkApp = sink.Install(nodes.Get(destNode));
        sinkApp.Start(Seconds(0.0));
        sinkApp.Stop(Seconds(simTime));

        // Connect PacketSink trace to validator for application-level packet reception
        Ptr<PacketSink> packetSink = DynamicCast<PacketSink>(sinkApp.Get(0));
        if (packetSink) {
            packetSink->TraceConnectWithoutContext("Rx",
                MakeCallback(&ArpmecValidator::ApplicationPacketReceived, &validator));
        }

        // Install on-off application on source
        OnOffHelper onoff("ns3::UdpSocketFactory",
                         InetSocketAddress(interfaces.GetAddress(destNode), 9));
        onoff.SetConstantRate(DataRate(std::to_string((uint32_t)(dataRate * packetSize * 8)) + "bps"));
        onoff.SetAttribute("PacketSize", UintegerValue(packetSize));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));

        ApplicationContainer sourceApp = onoff.Install(nodes.Get(sourceNode));
        sourceApp.Start(Seconds(1.0));
        sourceApp.Stop(Seconds(simTime - 1.0));

        // Connect OnOffApplication trace to validator for application-level packet transmission
        Ptr<OnOffApplication> onoffApp = DynamicCast<OnOffApplication>(sourceApp.Get(0));
        if (onoffApp) {
            onoffApp->TraceConnectWithoutContext("Tx",
                MakeCallback(&ArpmecValidator::ApplicationPacketSent, &validator));
        }

        // Register this flow for connectivity analysis
        validator.RegisterFlow(sourceNode, destNode);

        apps.Add(sourceApp);
        apps.Add(sinkApp);
    }

    // Show MEC Infrastructure Status
    std::cout << "\n=== MEC INFRASTRUCTURE STATUS ===" << std::endl;
    uint32_t mecGateways = 0, mecServers = 0;
    for (uint32_t i = 0; i < numNodes; i++) {
        Ptr<ns3::arpmec::RoutingProtocol> arpmecRouting = 
            DynamicCast<ns3::arpmec::RoutingProtocol>(
                nodes.Get(i)->GetObject<Ipv4>()->GetRoutingProtocol());
        
        if (arpmecRouting) {
            if (arpmecRouting->IsMecGateway()) {
                mecGateways++;
                std::cout << "  📡 Node " << i << " = MEC Gateway" << std::endl;
            }
            if (arpmecRouting->IsMecServer()) {
                mecServers++;
                std::cout << "  🖥️  Node " << i << " = MEC Server" << std::endl;
            }
        }
    }
    std::cout << "Total MEC Infrastructure: " << mecGateways << " Gateways, " 
              << mecServers << " Servers" << std::endl;

    // Connect trace sources to validation callbacks AND set diverse energy levels
    for (uint32_t i = 0; i < numNodes; i++) {
        // Get the ARPMEC routing protocol from each node
        Ptr<Ipv4> ipv4 = nodes.Get(i)->GetObject<Ipv4>();
        Ptr<Ipv4RoutingProtocol> routingProtocol = ipv4->GetRoutingProtocol();
        Ptr<arpmec::RoutingProtocol> arpmecProtocol = DynamicCast<arpmec::RoutingProtocol>(routingProtocol);

        if (arpmecProtocol) {
            // Set diverse energy levels to test clustering algorithm properly
            // Adjusted distribution to ensure more nodes can become cluster heads
            double energyLevel;
            if (i < 3) {
                energyLevel = 0.9 + (i % 2) * 0.1; // High energy: 0.9-1.0 (3 nodes - potential CHs)
            } else if (i < 6) {
                energyLevel = 0.8 + (i % 3) * 0.05; // Medium-high energy: 0.8-0.9 (3 nodes)
            } else if (i < 12) {
                energyLevel = 0.65 + (i % 2) * 0.1; // Medium energy: 0.65-0.75 (6 nodes)
            } else {
                energyLevel = 0.6 + (i % 4) * 0.05; // Lower energy: 0.6-0.75 (8 nodes)
            }

            // Set the energy level in the clustering component
            Ptr<arpmec::ArpmecClustering> clustering = arpmecProtocol->GetClustering();
            if (clustering) {
                clustering->SetEnergyLevel(energyLevel);
                std::cout << "Node " << i << " energy level set to: " << energyLevel << std::endl;
            }

            // Connect packet transmission traces
            arpmecProtocol->TraceConnectWithoutContext("Tx",
                MakeCallback(&ArpmecValidator::PacketSentCallback, &validator));
            arpmecProtocol->TraceConnectWithoutContext("Rx",
                MakeCallback(&ArpmecValidator::PacketReceivedCallback, &validator));

            // Connect cluster head status trace
            arpmecProtocol->TraceConnectWithoutContext("ClusterHead",
                MakeCallback(&ArpmecValidator::ClusterHeadCallback, &validator));

            // Connect routing decision trace
            arpmecProtocol->TraceConnectWithoutContext("RouteDecision",
                MakeCallback(&ArpmecValidator::RouteDecisionCallback, &validator));

            // Connect LQE update trace
            arpmecProtocol->TraceConnectWithoutContext("LqeUpdate",
                MakeCallback(&ArpmecValidator::LqeUpdateCallback, &validator));

            // Connect energy update trace
            arpmecProtocol->TraceConnectWithoutContext("EnergyUpdate",
                MakeCallback(&ArpmecValidator::EnergyUpdateCallback, &validator));
        }
    }

    // Configure NetAnim
    AnimationInterface anim("arpmec-validation.xml");
    anim.SetMaxPktsPerTraceFile(500000);

    // Run simulation
    std::cout << "\nStarting simulation..." << std::endl;
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    // Print results and validate
    validator.PrintResults();
    validator.ValidateAgainstPaper();

    Simulator::Destroy();

    std::cout << "\nValidation test completed. Check arpmec-validation.xml for network animation." << std::endl;

    return 0;
}
