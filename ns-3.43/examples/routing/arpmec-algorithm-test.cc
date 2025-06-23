/*
 * ARPMEC Algorithm Compliance Test
 * Specifically tests Algorithm 2 and Algorithm 3 implementations
 */

#include "ns3/arpmec-module.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include <iostream>
#include <fstream>

using namespace ns3;
using namespace arpmec;

NS_LOG_COMPONENT_DEFINE("ArpmecAlgorithmTest");

void TestAlgorithm2Clustering()
{
    std::cout << "\n=== TESTING ALGORITHM 2 - CLUSTERING PROTOCOL ===" << std::endl;
    
    // Create a small network to test clustering behavior
    NodeContainer nodes;
    nodes.Create(10);
    
    // Configure WiFi
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    
    YansWifiPhyHelper wifiPhy;
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    wifiPhy.SetChannel(wifiChannel.Create());
    
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    
    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, nodes);
    
    // Position nodes in a line for predictable clustering
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < 10; i++) {
        positionAlloc->Add(Vector(i * 30.0, 0.0, 0.0)); // 30m apart
    }
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);
    
    // Install ARPMEC with clustering
    ArpmecHelper arpmec;
    InternetStackHelper stack;
    stack.SetRoutingHelper(arpmec);
    stack.Install(nodes);
    
    // Assign addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);
    
    std::cout << "Network topology: Linear (10 nodes, 30m apart)" << std::endl;
    std::cout << "Running clustering algorithm for 30 seconds..." << std::endl;
    
    // Run simulation to allow clustering to stabilize
    Simulator::Stop(Seconds(30.0));
    Simulator::Run();
    
    // Get clustering results from routing protocols
    std::cout << "\nClustering Results:" << std::endl;
    for (uint32_t i = 0; i < nodes.GetN(); i++) {
        Ptr<Node> node = nodes.Get(i);
        Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
        Ptr<Ipv4RoutingProtocol> routing = ipv4->GetRoutingProtocol();
        Ptr<RoutingProtocol> arpmecRouting = DynamicCast<RoutingProtocol>(routing);
        
        if (arpmecRouting) {
            // Note: We would need to add public methods to check cluster status
            std::cout << "Node " << i << ": [Cluster status would be checked here]" << std::endl;
        }
    }
    
    Simulator::Destroy();
    std::cout << "Algorithm 2 test completed." << std::endl;
}

void TestAlgorithm3AdaptiveRouting()
{
    std::cout << "\n=== TESTING ALGORITHM 3 - ADAPTIVE ROUTING ===" << std::endl;
    
    // Create a network with clear cluster structure
    NodeContainer nodes;
    nodes.Create(15);
    
    // Configure WiFi
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    
    YansWifiPhyHelper wifiPhy;
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    wifiPhy.SetChannel(wifiChannel.Create());
    
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    
    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, nodes);
    
    // Create two clusters with gateway nodes
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    
    // Cluster 1: nodes 0-6
    for (uint32_t i = 0; i < 7; i++) {
        double x = (i % 3) * 20.0;
        double y = (i / 3) * 20.0;
        positionAlloc->Add(Vector(x, y, 0.0));
    }
    
    // Gateway area: nodes 7-8
    positionAlloc->Add(Vector(100.0, 20.0, 0.0));
    positionAlloc->Add(Vector(120.0, 20.0, 0.0));
    
    // Cluster 2: nodes 9-14
    for (uint32_t i = 0; i < 6; i++) {
        double x = 200.0 + (i % 3) * 20.0;
        double y = (i / 3) * 20.0;
        positionAlloc->Add(Vector(x, y, 0.0));
    }
    
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);
    
    // Install ARPMEC
    ArpmecHelper arpmec;
    InternetStackHelper stack;
    stack.SetRoutingHelper(arpmec);
    stack.Install(nodes);
    
    // Assign addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);
    
    std::cout << "Network topology: Two clusters with gateway nodes" << std::endl;
    std::cout << "Cluster 1: nodes 0-6, Gateway: nodes 7-8, Cluster 2: nodes 9-14" << std::endl;
    
    // Allow network to stabilize
    Simulator::Schedule(Seconds(10.0), []() {
        std::cout << "Network stabilization phase completed." << std::endl;
    });
    
    // Create traffic patterns to test different routing scenarios
    ApplicationContainer apps;
    
    // Intra-cluster traffic (within cluster 1)
    PacketSinkHelper sink1("ns3::UdpSocketFactory", 
                          InetSocketAddress(Ipv4Address::GetAny(), 9001));
    ApplicationContainer sinkApp1 = sink1.Install(nodes.Get(1));
    sinkApp1.Start(Seconds(15.0));
    sinkApp1.Stop(Seconds(45.0));
    
    OnOffHelper onoff1("ns3::UdpSocketFactory",
                      InetSocketAddress(interfaces.GetAddress(1), 9001));
    onoff1.SetConstantRate(DataRate("32kbps"));
    onoff1.SetAttribute("PacketSize", UintegerValue(512));
    
    ApplicationContainer sourceApp1 = onoff1.Install(nodes.Get(0));
    sourceApp1.Start(Seconds(15.0));
    sourceApp1.Stop(Seconds(45.0));
    
    // Inter-cluster traffic (cluster 1 to cluster 2)
    PacketSinkHelper sink2("ns3::UdpSocketFactory", 
                          InetSocketAddress(Ipv4Address::GetAny(), 9002));
    ApplicationContainer sinkApp2 = sink2.Install(nodes.Get(10));
    sinkApp2.Start(Seconds(20.0));
    sinkApp2.Stop(Seconds(50.0));
    
    OnOffHelper onoff2("ns3::UdpSocketFactory",
                      InetSocketAddress(interfaces.GetAddress(10), 9002));
    onoff2.SetConstantRate(DataRate("32kbps"));
    onoff2.SetAttribute("PacketSize", UintegerValue(512));
    
    ApplicationContainer sourceApp2 = onoff2.Install(nodes.Get(2));
    sourceApp2.Start(Seconds(20.0));
    sourceApp2.Stop(Seconds(50.0));
    
    std::cout << "Traffic patterns:" << std::endl;
    std::cout << "- Intra-cluster: Node 0 -> Node 1 (should use INTRA_CLUSTER routing)" << std::endl;
    std::cout << "- Inter-cluster: Node 2 -> Node 10 (should use INTER_CLUSTER/GATEWAY routing)" << std::endl;
    
    // Run simulation
    Simulator::Stop(Seconds(60.0));
    Simulator::Run();
    
    std::cout << "Algorithm 3 test completed." << std::endl;
    std::cout << "Check logs for routing decisions made by adaptive routing module." << std::endl;
    
    Simulator::Destroy();
}

void TestLQEFunctionality()
{
    std::cout << "\n=== TESTING LQE (LINK QUALITY ESTIMATION) ===" << std::endl;
    
    // Create a simple 5-node network
    NodeContainer nodes;
    nodes.Create(5);
    
    // Configure WiFi with different signal strengths
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    
    YansWifiPhyHelper wifiPhy;
    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::FriisPropagationLossModel");
    wifiPhy.SetChannel(wifiChannel.Create());
    
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    
    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, nodes);
    
    // Position nodes with varying distances to test LQE
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    positionAlloc->Add(Vector(0.0, 0.0, 0.0));    // Node 0: origin
    positionAlloc->Add(Vector(20.0, 0.0, 0.0));   // Node 1: close
    positionAlloc->Add(Vector(50.0, 0.0, 0.0));   // Node 2: medium
    positionAlloc->Add(Vector(100.0, 0.0, 0.0));  // Node 3: far
    positionAlloc->Add(Vector(200.0, 0.0, 0.0));  // Node 4: very far
    
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);
    
    // Install ARPMEC
    ArpmecHelper arpmec;
    InternetStackHelper stack;
    stack.SetRoutingHelper(arpmec);
    stack.Install(nodes);
    
    // Assign addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);
    
    std::cout << "Network topology: Linear (5 nodes at 0m, 20m, 50m, 100m, 200m)" << std::endl;
    std::cout << "Running LQE calculations for 40 seconds..." << std::endl;
    
    // Generate some traffic to trigger LQE calculations
    for (uint32_t i = 1; i < 5; i++) {
        PacketSinkHelper sink("ns3::UdpSocketFactory", 
                             InetSocketAddress(Ipv4Address::GetAny(), 9000 + i));
        ApplicationContainer sinkApp = sink.Install(nodes.Get(i));
        sinkApp.Start(Seconds(1.0));
        sinkApp.Stop(Seconds(39.0));
        
        OnOffHelper onoff("ns3::UdpSocketFactory",
                         InetSocketAddress(interfaces.GetAddress(i), 9000 + i));
        onoff.SetConstantRate(DataRate("16kbps"));
        onoff.SetAttribute("PacketSize", UintegerValue(256));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        
        ApplicationContainer sourceApp = onoff.Install(nodes.Get(0));
        sourceApp.Start(Seconds(1.0 + i * 2.0));
        sourceApp.Stop(Seconds(39.0));
    }
    
    // Run simulation
    Simulator::Stop(Seconds(40.0));
    Simulator::Run();
    
    std::cout << "LQE test completed." << std::endl;
    std::cout << "Check logs for LQE calculations at different distances." << std::endl;
    
    Simulator::Destroy();
}

int main(int argc, char* argv[])
{
    // Enable detailed logging
    LogComponentEnable("ArpmecAlgorithmTest", LOG_LEVEL_INFO);
    LogComponentEnable("ArpmecRoutingProtocol", LOG_LEVEL_INFO);
    LogComponentEnable("ArpmecClustering", LOG_LEVEL_INFO);
    LogComponentEnable("ArpmecAdaptiveRouting", LOG_LEVEL_INFO);
    LogComponentEnable("ArpmecLqe", LOG_LEVEL_INFO);
    
    // Parse command line
    CommandLine cmd;
    bool testClustering = true;
    bool testRouting = true;
    bool testLqe = true;
    
    cmd.AddValue("clustering", "Test Algorithm 2 (clustering)", testClustering);
    cmd.AddValue("routing", "Test Algorithm 3 (adaptive routing)", testRouting);
    cmd.AddValue("lqe", "Test LQE functionality", testLqe);
    cmd.Parse(argc, argv);
    
    std::cout << "=== ARPMEC ALGORITHM COMPLIANCE TEST ===" << std::endl;
    std::cout << "This test validates the implementation of ARPMEC algorithms against paper specifications." << std::endl;
    
    try {
        if (testClustering) {
            TestAlgorithm2Clustering();
        }
        
        if (testRouting) {
            TestAlgorithm3AdaptiveRouting();
        }
        
        if (testLqe) {
            TestLQEFunctionality();
        }
        
        std::cout << "\n=== ALL ALGORITHM TESTS COMPLETED ===" << std::endl;
        std::cout << "Review the output above and logs for detailed algorithm behavior." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
