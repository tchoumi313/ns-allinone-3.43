/*
 * Quick ARPMEC Performance Test
 * This test runs for a shorter duration to get performance metrics quickly
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/arpmec-helper.h"
#include <iomanip>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ArpmecQuickTest");

int main(int argc, char* argv[])
{
    // Test parameters
    uint32_t nNodes = 20;
    double distance = 40.0; // meters
    double simTime = 15.0;  // Shorter simulation time
    
    std::cout << "=== ARPMEC Quick Performance Test ===" << std::endl;
    std::cout << "Nodes: " << nNodes << std::endl;
    std::cout << "Distance: " << distance << " meters" << std::endl;
    std::cout << "Simulation Time: " << simTime << " seconds" << std::endl;
    std::cout << "======================================" << std::endl;

    // Create nodes
    NodeContainer nodes;
    nodes.Create(nNodes);
    
    // Set up grid topology
    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(0.0),
                                  "MinY", DoubleValue(0.0),
                                  "DeltaX", DoubleValue(distance),
                                  "DeltaY", DoubleValue(distance),
                                  "GridWidth", UintegerValue(10),
                                  "LayoutType", StringValue("RowFirst"));
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);

    // WiFi setup
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211b);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode", StringValue("DsssRate11Mbps"),
                                 "ControlMode", StringValue("DsssRate1Mbps"));

    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());

    NetDeviceContainer devices = wifi.Install(phy, mac, nodes);

    // Install Internet stack with ARPMEC routing
    ArpmecHelper arpmec;
    InternetStackHelper internet;
    internet.SetRoutingHelper(arpmec);
    internet.Install(nodes);

    // Assign IP addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = address.Assign(devices);

    // Install test applications (Node 0 -> Node 19)
    uint16_t port = 9;
    
    // Sink on destination node
    PacketSinkHelper sink("ns3::UdpSocketFactory", 
                         Address(InetSocketAddress(Ipv4Address::GetAny(), port)));
    ApplicationContainer sinkApp = sink.Install(nodes.Get(19));
    sinkApp.Start(Seconds(1.0));
    sinkApp.Stop(Seconds(simTime));

    // Source on sender node
    OnOffHelper source("ns3::UdpSocketFactory",
                      Address(InetSocketAddress(interfaces.GetAddress(19), port)));
    source.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    source.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    source.SetAttribute("DataRate", StringValue("250kbps"));
    source.SetAttribute("PacketSize", UintegerValue(512));

    ApplicationContainer sourceApp = source.Install(nodes.Get(0));
    sourceApp.Start(Seconds(5.0));
    sourceApp.Stop(Seconds(simTime - 1.0));

    std::cout << "Starting simulation..." << std::endl;
    
    // Run simulation
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();

    // Calculate performance metrics
    Ptr<PacketSink> sinkPtr = DynamicCast<PacketSink>(sinkApp.Get(0));
    uint32_t totalRx = sinkPtr->GetTotalRx();
    uint32_t packetsReceived = totalRx / 512; // Assuming 512 byte packets
    
    // Estimate packets sent (rough calculation)
    double dataTime = simTime - 6.0; // 5s start + 1s stop margin
    double dataRate = 250000.0; // 250kbps in bps
    uint32_t estimatedPacketsSent = (uint32_t)((dataRate * dataTime) / (512 * 8));
    
    double pdr = (estimatedPacketsSent > 0) ? (double)packetsReceived / estimatedPacketsSent * 100.0 : 0.0;

    std::cout << "\n=== ARPMEC Performance Results ===" << std::endl;
    std::cout << "Estimated packets sent: " << estimatedPacketsSent << std::endl;
    std::cout << "Packets received: " << packetsReceived << std::endl;
    std::cout << "Bytes received: " << totalRx << std::endl;
    std::cout << "Packet Delivery Ratio: " << std::fixed << std::setprecision(2) << pdr << "%" << std::endl;
    std::cout << "=====================================" << std::endl;

    Simulator::Destroy();
    return 0;
}
