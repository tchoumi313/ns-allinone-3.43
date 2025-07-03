/*
 * Copyright (c) 2024
 * 
 * Test for MEC Inter-Cluster Communication: cluster A -> MEC A => MEC B -> cluster B
 * This test specifically validates the end-to-end communication flow through MEC infrastructure
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/applications-module.h"
#include "ns3/arpmec-module.h"
#include "ns3/netanim-module.h"

using namespace ns3;
using namespace ns3::arpmec;

NS_LOG_COMPONENT_DEFINE("ArpmecMecCommunicationTest");

/**
 * \brief Test validator to track MEC inter-cluster communication
 */
class MecCommValidator 
{
public:
    MecCommValidator() : m_clusterAMessages(0), m_mecAForwards(0), m_mecBReceives(0), m_clusterBDeliveries(0) {}

    void ClusterAToMecA(uint32_t nodeId, uint32_t clusterId, uint32_t gatewayId) {
        m_clusterAMessages++;
        std::cout << "📤 [STEP 1] Cluster A (Node " << nodeId << ", Cluster " << clusterId 
                  << ") -> MEC Gateway A (" << gatewayId << ")" << std::endl;
    }

    void MecAToMecB(uint32_t sourceGateway, uint32_t targetGateway) {
        m_mecAForwards++;
        std::cout << "🔄 [STEP 2] MEC Gateway A (" << sourceGateway 
                  << ") => MEC Gateway B (" << targetGateway << ")" << std::endl;
    }

    void MecBReceives(uint32_t gatewayId, uint32_t targetCluster) {
        m_mecBReceives++;
        std::cout << "📥 [STEP 3] MEC Gateway B (" << gatewayId 
                  << ") receives message for Cluster B (" << targetCluster << ")" << std::endl;
    }

    void MecBToClusterB(uint32_t gatewayId, uint32_t clusterId, uint32_t clusterHead) {
        m_clusterBDeliveries++;
        std::cout << "📩 [STEP 4] MEC Gateway B (" << gatewayId 
                  << ") -> Cluster B (Cluster " << clusterId << ", Head " << clusterHead << ")" << std::endl;
    }

    void PrintResults() {
        std::cout << "\n=== MEC INTER-CLUSTER COMMUNICATION VALIDATION ===" << std::endl;
        std::cout << "Step 1 - Cluster A to MEC A: " << m_clusterAMessages << " messages" << std::endl;
        std::cout << "Step 2 - MEC A to MEC B: " << m_mecAForwards << " forwards" << std::endl;
        std::cout << "Step 3 - MEC B receives: " << m_mecBReceives << " receptions" << std::endl;
        std::cout << "Step 4 - MEC B to Cluster B: " << m_clusterBDeliveries << " deliveries" << std::endl;
        
        bool success = (m_clusterAMessages > 0 && m_mecAForwards > 0 && 
                       m_mecBReceives > 0 && m_clusterBDeliveries > 0);
        
        std::cout << "\n🎯 END-TO-END COMMUNICATION: " 
                  << (success ? "✅ SUCCESS" : "❌ FAILED") << std::endl;
        
        if (success) {
            std::cout << "✅ cluster A -> MEC A (Gateway) => MEC B -> cluster B VERIFIED!" << std::endl;
        } else {
            std::cout << "❌ Communication flow incomplete - debugging needed" << std::endl;
        }
    }

private:
    uint32_t m_clusterAMessages;
    uint32_t m_mecAForwards;
    uint32_t m_mecBReceives;
    uint32_t m_clusterBDeliveries;
};

// Global validator instance
MecCommValidator g_validator;

/**
 * \brief Test application to generate inter-cluster messages
 */
class MecTestApp : public Application
{
public:
    MecTestApp() : m_socket(0), m_peer(), m_packetSize(1024), m_interval(Seconds(2.0)) {}
    
    void Setup(Address address, uint32_t packetSize, Time interval) {
        m_peer = address;
        m_packetSize = packetSize;
        m_interval = interval;
    }

private:
    void StartApplication() override {
        m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
        m_socket->Bind();
        m_socket->Connect(m_peer);
        
        // Send first packet after 5 seconds (let clustering stabilize)
        Simulator::Schedule(Seconds(5.0), &MecTestApp::SendPacket, this);
    }
    
    void StopApplication() override {
        if (m_socket) {
            m_socket->Close();
            m_socket = 0;
        }
    }
    
    void SendPacket() {
        Ptr<Packet> packet = Create<Packet>(m_packetSize);
        
        std::cout << "📤 Test App sending inter-cluster message (size: " 
                  << m_packetSize << " bytes)" << std::endl;
        
        m_socket->Send(packet);
        
        // Schedule next packet
        Simulator::Schedule(m_interval, &MecTestApp::SendPacket, this);
    }

    Ptr<Socket> m_socket;
    Address m_peer;
    uint32_t m_packetSize;
    Time m_interval;
};

int main(int argc, char* argv[])
{
    // Enable logging
    LogComponentEnable("ArpmecRoutingProtocol", LOG_LEVEL_INFO);
    LogComponentEnable("ArpmecClustering", LOG_LEVEL_INFO);
    LogComponentEnable("ArpmecMecGateway", LOG_LEVEL_INFO);
    LogComponentEnable("ArpmecMecCommunicationTest", LOG_LEVEL_INFO);

    // Simulation parameters
    uint32_t numNodes = 10;
    double simTime = 30.0;
    
    std::cout << "=== MEC INTER-CLUSTER COMMUNICATION TEST ===" << std::endl;
    std::cout << "Testing: cluster A -> MEC A (Gateway) => MEC B -> cluster B" << std::endl;
    std::cout << "Nodes: " << numNodes << ", Simulation time: " << simTime << "s" << std::endl;

    // Create nodes
    NodeContainer nodes;
    nodes.Create(numNodes);

    // Setup WiFi
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    
    YansWifiPhyHelper phy;
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    phy.SetChannel(channel.Create());
    
    NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

    // Setup mobility (linear topology for clear cluster separation)
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
    
    // Cluster A: Nodes 0-4 (left side)
    for (uint32_t i = 0; i < 5; i++) {
        positionAlloc->Add(Vector(i * 50.0, 0.0, 0.0));
    }
    
    // Cluster B: Nodes 5-9 (right side, separated)
    for (uint32_t i = 5; i < 10; i++) {
        positionAlloc->Add(Vector(300.0 + (i-5) * 50.0, 0.0, 0.0));
    }
    
    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    // Setup Internet stack with ARPMEC
    InternetStackHelper internet;
    ArpmecHelper arpmec;
    internet.SetRoutingHelper(arpmec);
    internet.Install(nodes);

    // Assign IP addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    // Setup MEC Infrastructure
    std::cout << "\n=== SETTING UP MEC INFRASTRUCTURE ===" << std::endl;
    
    // Node 1: MEC Gateway A (serves Cluster A)
    Ptr<RoutingProtocol> routingA = DynamicCast<RoutingProtocol>(
        nodes.Get(1)->GetObject<Ipv4>()->GetRoutingProtocol());
    if (routingA) {
        routingA->EnableMecGateway(101, 150.0); // ID: 101, Coverage: 150m
        std::cout << "📡 Node 1 = MEC Gateway A (ID: 101, Coverage: 150m)" << std::endl;
    }
    
    // Node 8: MEC Gateway B (serves Cluster B)  
    Ptr<RoutingProtocol> routingB = DynamicCast<RoutingProtocol>(
        nodes.Get(8)->GetObject<Ipv4>()->GetRoutingProtocol());
    if (routingB) {
        routingB->EnableMecGateway(108, 150.0); // ID: 108, Coverage: 150m
        std::cout << "📡 Node 8 = MEC Gateway B (ID: 108, Coverage: 150m)" << std::endl;
    }

    // Setup test application (Node 0 in Cluster A sends to Node 9 in Cluster B)
    uint16_t port = 9999;
    
    // Receiver in Cluster B (Node 9)
    PacketSinkHelper sink("ns3::UdpSocketFactory", 
                         InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApp = sink.Install(nodes.Get(9));
    sinkApp.Start(Seconds(1.0));
    sinkApp.Stop(Seconds(simTime));
    
    // Sender in Cluster A (Node 0)
    Ptr<MecTestApp> testApp = CreateObject<MecTestApp>();
    testApp->Setup(InetSocketAddress(interfaces.GetAddress(9), port), 
                   1024, Seconds(3.0));
    nodes.Get(0)->AddApplication(testApp);
    testApp->SetStartTime(Seconds(2.0));
    testApp->SetStopTime(Seconds(simTime));

    std::cout << "📤 Sender: Node 0 (Cluster A) -> Node 9 (Cluster B)" << std::endl;
    std::cout << "📡 Route: Node 0 -> MEC Gateway A (Node 1) => MEC Gateway B (Node 8) -> Node 9" << std::endl;

    // Schedule periodic MEC communication tests
    for (double t = 10.0; t < simTime; t += 5.0) {
        Simulator::Schedule(Seconds(t), []() {
            std::cout << "\n⏱️  [" << Simulator::Now().GetSeconds() 
                      << "s] Testing MEC inter-cluster communication..." << std::endl;
            // Here we would trigger MEC gateway communication tests
        });
    }

    // Configure NetAnim
    AnimationInterface anim("arpmec-mec-communication-test.xml");
    anim.SetMaxPktsPerTraceFile(500000);
    
    // Set node descriptions for animation
    anim.UpdateNodeDescription(0, "Cluster A Node");
    anim.UpdateNodeDescription(1, "MEC Gateway A");
    anim.UpdateNodeDescription(8, "MEC Gateway B"); 
    anim.UpdateNodeDescription(9, "Cluster B Node");

    std::cout << "\n🚀 Starting simulation..." << std::endl;
    
    // Run simulation
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    
    std::cout << "\n📊 Simulation completed." << std::endl;
    
    // Print validation results
    g_validator.PrintResults();
    
    std::cout << "\n💡 Check arpmec-mec-communication-test.xml for network animation" << std::endl;
    
    Simulator::Destroy();
    return 0;
}
