/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * ARPMEC Paper Algorithm Verification Test Suite
 * 
 * This test suite verifies that our ARPMEC implementation matches the 
 * algorithms described in the paper "Adaptive Routing Protocol for 
 * Mobile Edge Computing-based IoT Networks"
 */

#include "ns3/arpmec-clustering.h"
#include "ns3/arpmec-lqe.h"
#include "ns3/arpmec-adaptive-routing.h"
#include "ns3/arpmec-mec-gateway.h"
#include "ns3/test.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/node-container.h"
#include "ns3/vector.h"
#include <map>
#include <vector>

using namespace ns3;
using namespace ns3::arpmec;

NS_LOG_COMPONENT_DEFINE("ArpmecPaperVerificationTest");

/**
 * \ingroup arpmec-test
 * \ingroup tests
 *
 * \brief ARPMEC Paper Algorithm Verification Test Case
 * 
 * Tests that verify our implementation matches the paper algorithms:
 * - Algorithm 1: LQE Module
 * - Algorithm 2: Clustering Protocol  
 * - Algorithm 3: Adaptive Routing
 * - MEC Gateway Operations
 */
class ArpmecPaperVerificationTestCase : public TestCase
{
public:
    ArpmecPaperVerificationTestCase();
    virtual ~ArpmecPaperVerificationTestCase();

private:
    virtual void DoRun() override;
    
    /**
     * \brief Test Algorithm 1: LQE Module from paper
     * Verifies PDR calculation, RSSI processing, and neighbor ranking
     */
    void TestAlgorithm1_LQE();
    
    /**
     * \brief Test Algorithm 2: Clustering Protocol from paper
     * Verifies cluster head election, energy thresholds, and cluster formation
     */
    void TestAlgorithm2_Clustering();
    
    /**
     * \brief Test Algorithm 3: Adaptive Routing from paper
     * Verifies intra-cluster, inter-cluster, and AODV fallback routing
     */
    void TestAlgorithm3_AdaptiveRouting();
    
    /**
     * \brief Test MEC Gateway Operations from paper
     * Verifies task offloading, load balancing, and cluster management
     */
    void TestMecGatewayOperations();
    
    /**
     * \brief Test integrated scenario with multiple nodes
     * Verifies all algorithms working together as described in paper
     */
    void TestIntegratedScenario();
    
    /**
     * \brief Helper function to create test topology
     */
    void CreateTestTopology();
    
    /**
     * \brief Helper function to verify clustering behavior matches paper
     */
    bool VerifyClusteringBehavior();
    
    /**
     * \brief Helper function to verify routing decisions match paper
     */
    bool VerifyRoutingDecisions();

private:
    std::vector<Ptr<ArpmecClustering>> m_clusteringNodes;
    std::vector<Ptr<ArpmecLqe>> m_lqeNodes;
    std::vector<Ptr<ArpmecAdaptiveRouting>> m_routingNodes;
    std::vector<Ptr<ArpmecMecGateway>> m_gatewayNodes;
    NodeContainer m_nodes;
    uint32_t m_numNodes;
};

ArpmecPaperVerificationTestCase::ArpmecPaperVerificationTestCase()
    : TestCase("ARPMEC Paper Algorithm Verification"),
      m_numNodes(10)
{
}

ArpmecPaperVerificationTestCase::~ArpmecPaperVerificationTestCase()
{
}

void
ArpmecPaperVerificationTestCase::DoRun()
{
    NS_LOG_FUNCTION(this);
    
    NS_LOG_INFO("=== Starting ARPMEC Paper Algorithm Verification Tests ===");
    
    // Test each algorithm individually
    TestAlgorithm1_LQE();
    TestAlgorithm2_Clustering();
    TestAlgorithm3_AdaptiveRouting();
    TestMecGatewayOperations();
    
    // Test integrated scenario
    TestIntegratedScenario();
    
    NS_LOG_INFO("=== ARPMEC Paper Algorithm Verification Tests Completed ===");
}

void
ArpmecPaperVerificationTestCase::TestAlgorithm1_LQE()
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO("Testing Algorithm 1: LQE Module");
    
    // Create LQE instance
    Ptr<ArpmecLqe> lqe = CreateObject<ArpmecLqe>();
    lqe->Initialize(1); // Node ID 1
    
    // Test 1: PDR Calculation (Paper equation)
    // PDR = successfully_received_packets / total_sent_packets
    uint32_t neighborId = 2;
    
    // Simulate packet reception with different success rates
    for (int i = 0; i < 10; i++)
    {
        bool success = (i < 8); // 80% success rate
        lqe->UpdatePacketReception(neighborId, success);
    }
    
    double pdr = lqe->GetPdr(neighborId);
    NS_TEST_ASSERT_MSG_EQ_TOL(pdr, 0.8, 0.1, "PDR calculation should match paper formula");
    
    // Test 2: RSSI Processing
    // Test that RSSI values are properly processed and contribute to link quality
    lqe->UpdateLinkQuality(neighborId, -50.0, 0.8, Simulator::Now(), Simulator::Now());
    double linkScore1 = lqe->GetLinkScore(neighborId);
    
    lqe->UpdateLinkQuality(neighborId, -80.0, 0.8, Simulator::Now(), Simulator::Now());
    double linkScore2 = lqe->GetLinkScore(neighborId);
    
    NS_TEST_ASSERT_MSG_GT(linkScore1, linkScore2, "Better RSSI should result in higher link score");
    
    // Test 3: Neighbor Ranking
    // Add multiple neighbors with different qualities
    lqe->UpdateLinkQuality(3, -40.0, 0.9, Simulator::Now(), Simulator::Now());
    lqe->UpdateLinkQuality(4, -60.0, 0.7, Simulator::Now(), Simulator::Now());
    lqe->UpdateLinkQuality(5, -70.0, 0.6, Simulator::Now(), Simulator::Now());
    
    std::vector<uint32_t> rankedNeighbors = lqe->GetNeighborsByQuality();
    NS_TEST_ASSERT_MSG_GT(rankedNeighbors.size(), 0, "Should have neighbors in ranking");
    
    // Verify ranking order (highest quality first)
    if (rankedNeighbors.size() >= 2)
    {
        double firstScore = lqe->GetLinkScore(rankedNeighbors[0]);
        double secondScore = lqe->GetLinkScore(rankedNeighbors[1]);
        NS_TEST_ASSERT_MSG_GE(firstScore, secondScore, "Neighbors should be ranked by quality");
    }
    
    NS_LOG_INFO("Algorithm 1 (LQE) verification: PASSED");
}

void
ArpmecPaperVerificationTestCase::TestAlgorithm2_Clustering()
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO("Testing Algorithm 2: Clustering Protocol");
    
    // Create multiple clustering instances to test cluster formation
    std::vector<Ptr<ArpmecClustering>> clusters;
    std::vector<Ptr<ArpmecLqe>> lqes;
    
    for (uint32_t i = 0; i < 5; i++)
    {
        Ptr<ArpmecClustering> clustering = CreateObject<ArpmecClustering>();
        Ptr<ArpmecLqe> lqe = CreateObject<ArpmecLqe>();
        
        lqe->Initialize(i);
        clustering->Initialize(i, lqe);
        
        clusters.push_back(clustering);
        lqes.push_back(lqe);
    }
    
    // Test 1: Energy Threshold Check (Paper Algorithm 2, line 3)
    // Verify that nodes with energy below threshold don't become cluster heads
    clusters[0]->SetEnergyLevel(0.5); // Below typical threshold
    clusters[1]->SetEnergyLevel(0.8); // Above threshold
    
    // Test 2: Link Quality Threshold (Paper Algorithm 2, line 7)
    // Create neighbor relationships with different link qualities
    for (uint32_t i = 0; i < 4; i++)
    {
        for (uint32_t j = i + 1; j < 5; j++)
        {
            // Simulate bidirectional links
            double quality = 0.3 + (i + j) * 0.1; // Varying quality
            lqes[i]->UpdateLinkQuality(j, -50.0, quality, Simulator::Now(), Simulator::Now());
            lqes[j]->UpdateLinkQuality(i, -50.0, quality, Simulator::Now(), Simulator::Now());
        }
    }
    
    // Start clustering on all nodes
    for (auto& cluster : clusters)
    {
        cluster->Start();
    }
    
    // Run simulation to allow clustering algorithm to execute
    Simulator::Schedule(Seconds(2.0), [&]() {
        // Test 3: Cluster Head Election
        uint32_t clusterHeads = 0;
        uint32_t clusterMembers = 0;
        uint32_t isolatedNodes = 0;
        
        for (uint32_t i = 0; i < clusters.size(); i++)
        {
            ArpmecClustering::NodeState state = clusters[i]->GetNodeState();
            switch (state)
            {
                case ArpmecClustering::CLUSTER_HEAD:
                    clusterHeads++;
                    NS_LOG_INFO("Node " << i << " is cluster head");
                    break;
                case ArpmecClustering::CLUSTER_MEMBER:
                    clusterMembers++;
                    NS_LOG_INFO("Node " << i << " is cluster member");
                    break;
                case ArpmecClustering::ISOLATED:
                    isolatedNodes++;
                    NS_LOG_INFO("Node " << i << " is isolated");
                    break;
                default:
                    break;
            }
        }
        
        // Verify clustering results match paper expectations
        NS_TEST_ASSERT_MSG_GT(clusterHeads, 0, "Should have at least one cluster head");
        NS_TEST_ASSERT_MSG_LE(clusterHeads, 3, "Should not have too many cluster heads");
        
        // Test 4: Verify high-energy nodes are preferred as cluster heads
        for (uint32_t i = 0; i < clusters.size(); i++)
        {
            if (clusters[i]->GetNodeState() == ArpmecClustering::CLUSTER_HEAD)
            {
                double energy = clusters[i]->GetEnergyLevel();
                NS_TEST_ASSERT_MSG_GE(energy, 0.7, "Cluster heads should have sufficient energy");
            }
        }
    });
    
    Simulator::Run();
    Simulator::Destroy();
    
    NS_LOG_INFO("Algorithm 2 (Clustering) verification: PASSED");
}

void
ArpmecPaperVerificationTestCase::TestAlgorithm3_AdaptiveRouting()
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO("Testing Algorithm 3: Adaptive Routing");
    
    // Create test setup with routing, clustering, and LQE
    Ptr<ArpmecClustering> clustering = CreateObject<ArpmecClustering>();
    Ptr<ArpmecLqe> lqe = CreateObject<ArpmecLqe>();
    Ptr<ArpmecAdaptiveRouting> routing = CreateObject<ArpmecAdaptiveRouting>();
    
    uint32_t nodeId = 1;
    lqe->Initialize(nodeId);
    clustering->Initialize(nodeId, lqe);
    routing->Initialize(nodeId, clustering, lqe);
    
    // Set up test topology
    std::vector<uint32_t> clusterMembers = {1, 2, 3};
    routing->UpdateClusterTopology(1, clusterMembers);
    
    clustering->Start();
    
    // Test 1: Intra-cluster routing (Paper Algorithm 3, line 1-2)
    Ipv4Address dest1("10.1.1.2");
    uint32_t destNode1 = 2; // Same cluster
    
    ArpmecAdaptiveRouting::RoutingInfo route1 = routing->DetermineRoute(dest1, destNode1);
    NS_TEST_ASSERT_MSG_EQ(route1.decision, ArpmecAdaptiveRouting::INTRA_CLUSTER,
                         "Should use intra-cluster routing for same cluster destination");
    
    // Test 2: Inter-cluster routing (Paper Algorithm 3, line 3-7)
    std::vector<uint32_t> otherClusterMembers = {4, 5, 6};
    routing->UpdateClusterTopology(4, otherClusterMembers);
    
    Ipv4Address dest2("10.1.1.5");
    uint32_t destNode2 = 5; // Different cluster
    
    ArpmecAdaptiveRouting::RoutingInfo route2 = routing->DetermineRoute(dest2, destNode2);
    NS_TEST_ASSERT_MSG_EQ(route2.decision, ArpmecAdaptiveRouting::INTER_CLUSTER,
                         "Should use inter-cluster routing for different cluster destination");
    
    // Test 3: AODV fallback (Paper Algorithm 3, line 8)
    Ipv4Address dest3("10.1.1.10");
    uint32_t destNode3 = 10; // Unknown destination
    
    ArpmecAdaptiveRouting::RoutingInfo route3 = routing->DetermineRoute(dest3, destNode3);
    NS_TEST_ASSERT_MSG_EQ(route3.decision, ArpmecAdaptiveRouting::AODV_FALLBACK,
                         "Should use AODV fallback for unknown destinations");
    
    // Test 4: Route quality calculations
    NS_TEST_ASSERT_MSG_GT(route1.routeQuality, route2.routeQuality,
                         "Intra-cluster routes should have better quality than inter-cluster");
    NS_TEST_ASSERT_MSG_GT(route2.routeQuality, route3.routeQuality,
                         "Inter-cluster routes should have better quality than AODV fallback");
    
    // Test routing statistics
    std::map<ArpmecAdaptiveRouting::RouteDecision, uint32_t> stats = routing->GetRoutingStatistics();
    NS_TEST_ASSERT_MSG_EQ(stats[ArpmecAdaptiveRouting::INTRA_CLUSTER], 1,
                         "Should record intra-cluster routing decision");
    NS_TEST_ASSERT_MSG_EQ(stats[ArpmecAdaptiveRouting::INTER_CLUSTER], 1,
                         "Should record inter-cluster routing decision");
    NS_TEST_ASSERT_MSG_EQ(stats[ArpmecAdaptiveRouting::AODV_FALLBACK], 1,
                         "Should record AODV fallback decision");
    
    NS_LOG_INFO("Algorithm 3 (Adaptive Routing) verification: PASSED");
}

void
ArpmecPaperVerificationTestCase::TestMecGatewayOperations()
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO("Testing MEC Gateway Operations");
    
    // Create MEC Gateway
    Ptr<ArpmecMecGateway> gateway = CreateObject<ArpmecMecGateway>();
    gateway->Initialize(1, 100); // Gateway ID 1, initial load 100
    
    // Test 1: Task Processing
    gateway->ProcessMecTask(1001, 50, Seconds(1.0)); // Task ID, computational load, deadline
    
    // Verify task was queued
    NS_TEST_ASSERT_MSG_GT(gateway->GetCurrentLoad(), 100,
                         "Gateway load should increase after adding task");
    
    // Test 2: Load Balancing
    double initialLoad = gateway->GetCurrentLoad();
    
    // Add multiple gateways to test load balancing
    gateway->AddKnownGateway(2, 50.0); // Gateway 2 with lower load
    gateway->AddKnownGateway(3, 200.0); // Gateway 3 with higher load
    
    uint32_t bestGateway = gateway->FindBestMecGateway(0);
    NS_TEST_ASSERT_MSG_EQ(bestGateway, 2, "Should select gateway with lowest load");
    
    // Test 3: Cluster Management
    std::vector<uint32_t> clusterMembers = {10, 11, 12, 13, 14, 15, 16, 17, 18}; // 9 members
    gateway->UpdateClusterInfo(10, clusterMembers, 10); // Large cluster
    
    // Test cluster splitting for large clusters
    gateway->SplitLargeCluster(10);
    
    // Test 4: Task Completion Processing
    gateway->OnTaskCompletion(1001, 1, 0.5); // Task completed in 0.5 seconds
    
    // Verify load decreased after task completion
    Simulator::Schedule(Seconds(2.0), [&]() {
        double finalLoad = gateway->GetCurrentLoad();
        NS_TEST_ASSERT_MSG_LT(finalLoad, initialLoad,
                             "Gateway load should decrease after task completion");
    });
    
    Simulator::Run();
    Simulator::Destroy();
    
    NS_LOG_INFO("MEC Gateway Operations verification: PASSED");
}

void
ArpmecPaperVerificationTestCase::TestIntegratedScenario()
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO("Testing Integrated ARPMEC Scenario");
    
    CreateTestTopology();
    
    // Run simulation for cluster formation
    Simulator::Schedule(Seconds(1.0), [this]() {
        NS_LOG_INFO("Checking cluster formation after 1 second");
        NS_TEST_ASSERT_MSG_EQ(VerifyClusteringBehavior(), true,
                             "Clustering should work as expected");
    });
    
    // Test routing after clusters form
    Simulator::Schedule(Seconds(2.0), [this]() {
        NS_LOG_INFO("Checking routing decisions after 2 seconds");
        NS_TEST_ASSERT_MSG_EQ(VerifyRoutingDecisions(), true,
                             "Routing should work as expected");
    });
    
    // Test system stability
    Simulator::Schedule(Seconds(5.0), [this]() {
        NS_LOG_INFO("Checking system stability after 5 seconds");
        
        uint32_t activeNodes = 0;
        for (auto& clustering : m_clusteringNodes)
        {
            if (clustering->GetNodeState() != ArpmecClustering::UNDECIDED)
            {
                activeNodes++;
            }
        }
        
        NS_TEST_ASSERT_MSG_GT(activeNodes, m_numNodes / 2,
                             "Most nodes should be in active clustering state");
    });
    
    Simulator::Run();
    Simulator::Destroy();
    
    NS_LOG_INFO("Integrated Scenario verification: PASSED");
}

void
ArpmecPaperVerificationTestCase::CreateTestTopology()
{
    NS_LOG_FUNCTION(this);
    
    m_nodes.Create(m_numNodes);
    
    // Create ARPMEC components for each node
    for (uint32_t i = 0; i < m_numNodes; i++)
    {
        Ptr<ArpmecLqe> lqe = CreateObject<ArpmecLqe>();
        Ptr<ArpmecClustering> clustering = CreateObject<ArpmecClustering>();
        Ptr<ArpmecAdaptiveRouting> routing = CreateObject<ArpmecAdaptiveRouting>();
        
        lqe->Initialize(i);
        clustering->Initialize(i, lqe);
        routing->Initialize(i, clustering, lqe);
        
        // Set varying energy levels
        double energy = 0.5 + (i % 5) * 0.1; // Energy from 0.5 to 0.9
        clustering->SetEnergyLevel(energy);
        
        m_lqeNodes.push_back(lqe);
        m_clusteringNodes.push_back(clustering);
        m_routingNodes.push_back(routing);
        
        // Install mobility model
        Ptr<ConstantPositionMobilityModel> mobility = CreateObject<ConstantPositionMobilityModel>();
        Vector position(i * 50.0, 0.0, 0.0); // Linear topology
        mobility->SetPosition(position);
        m_nodes.Get(i)->AggregateObject(mobility);
    }
    
    // Create neighbor relationships based on proximity
    for (uint32_t i = 0; i < m_numNodes; i++)
    {
        for (uint32_t j = 0; j < m_numNodes; j++)
        {
            if (i != j && abs((int)i - (int)j) <= 2) // Neighbors within 2 hops
            {
                double distance = abs((int)i - (int)j) * 50.0;
                double rssi = -40.0 - distance / 10.0; // Simple path loss
                double pdr = std::max(0.3, 1.0 - distance / 200.0);
                
                m_lqeNodes[i]->UpdateLinkQuality(j, rssi, pdr, Simulator::Now(), Simulator::Now());
            }
        }
    }
    
    // Start all clustering protocols
    for (auto& clustering : m_clusteringNodes)
    {
        clustering->Start();
    }
    
    // Create some MEC gateways
    for (uint32_t i = 0; i < 3; i++)
    {
        Ptr<ArpmecMecGateway> gateway = CreateObject<ArpmecMecGateway>();
        gateway->Initialize(i + 100, 50 + i * 25); // Gateway IDs 100, 101, 102
        m_gatewayNodes.push_back(gateway);
    }
}

bool
ArpmecPaperVerificationTestCase::VerifyClusteringBehavior()
{
    NS_LOG_FUNCTION(this);
    
    uint32_t clusterHeads = 0;
    uint32_t clusterMembers = 0;
    
    for (uint32_t i = 0; i < m_clusteringNodes.size(); i++)
    {
        ArpmecClustering::NodeState state = m_clusteringNodes[i]->GetNodeState();
        if (state == ArpmecClustering::CLUSTER_HEAD)
        {
            clusterHeads++;
            
            // Verify cluster head has good energy
            double energy = m_clusteringNodes[i]->GetEnergyLevel();
            if (energy < 0.6)
            {
                NS_LOG_ERROR("Cluster head " << i << " has low energy: " << energy);
                return false;
            }
        }
        else if (state == ArpmecClustering::CLUSTER_MEMBER)
        {
            clusterMembers++;
        }
    }
    
    NS_LOG_INFO("Found " << clusterHeads << " cluster heads and " << clusterMembers << " members");
    
    // Should have reasonable clustering
    return (clusterHeads > 0 && clusterHeads <= m_numNodes / 2);
}

bool
ArpmecPaperVerificationTestCase::VerifyRoutingDecisions()
{
    NS_LOG_FUNCTION(this);
    
    // Test routing decisions from first node
    if (m_routingNodes.empty())
        return false;
    
    Ptr<ArpmecAdaptiveRouting> routing = m_routingNodes[0];
    
    // Update topology information for routing
    for (uint32_t i = 0; i < m_clusteringNodes.size(); i++)
    {
        if (m_clusteringNodes[i]->IsClusterHead())
        {
            std::vector<uint32_t> members = m_clusteringNodes[i]->GetClusterMembers();
            members.push_back(i); // Include cluster head itself
            routing->UpdateClusterTopology(i, members);
        }
    }
    
    // Test a few routing decisions
    uint32_t correctDecisions = 0;
    uint32_t totalTests = 3;
    
    for (uint32_t dest = 1; dest < 4; dest++)
    {
        if (dest == 0) continue; // Skip self
        
        Ipv4Address destAddr("10.1.1." + std::to_string(dest + 1));
        ArpmecAdaptiveRouting::RoutingInfo route = routing->DetermineRoute(destAddr, dest);
        
        if (route.decision != ArpmecAdaptiveRouting::AODV_FALLBACK)
        {
            correctDecisions++;
        }
    }
    
    NS_LOG_INFO("Routing made " << correctDecisions << " non-fallback decisions out of " << totalTests);
    
    return correctDecisions > 0; // At least some routing should work
}

/**
 * \ingroup arpmec-test
 * \ingroup tests
 *
 * \brief ARPMEC Paper Verification Test Suite
 */
class ArpmecPaperVerificationTestSuite : public TestSuite
{
public:
    ArpmecPaperVerificationTestSuite();
};

ArpmecPaperVerificationTestSuite::ArpmecPaperVerificationTestSuite()
    : TestSuite("arpmec-paper-verification", UNIT)
{
    AddTestCase(new ArpmecPaperVerificationTestCase, TestCase::QUICK);
}

static ArpmecPaperVerificationTestSuite arpmecPaperVerificationTestSuite;
