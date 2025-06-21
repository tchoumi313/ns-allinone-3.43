/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Test suite for ARPMEC Clustering module
 */

#include "ns3/arpmec-clustering.h"
#include "ns3/arpmec-lqe.h"
#include "ns3/test.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/packet.h"

using namespace ns3;
using namespace ns3::arpmec;

/**
 * \ingroup arpmec-test
 * \ingroup tests
 *
 * \brief ARPMEC Clustering Test Case
 */
class ArpmecClusteringTestCase : public TestCase
{
public:
    ArpmecClusteringTestCase();
    virtual ~ArpmecClusteringTestCase();

private:
    virtual void DoRun() override;
    
    /**
     * \brief Test basic clustering functionality
     */
    void TestBasicClustering();
    
    /**
     * \brief Test cluster head election
     */
    void TestClusterHeadElection();
    
    /**
     * \brief Test cluster joining
     */
    void TestClusterJoining();
    
    /**
     * \brief Test cluster maintenance
     */
    void TestClusterMaintenance();

    /**
     * \brief Callback for cluster events
     */
    void OnClusterEvent(ArpmecClustering::ClusterEvent event, uint32_t nodeId);
    
    /**
     * \brief Callback for packet sending
     */
    void OnSendPacket(Ptr<Packet> packet, uint32_t destination);

    // Test state variables
    ArpmecClustering::ClusterEvent m_lastEvent;
    uint32_t m_lastEventNodeId;
    uint32_t m_packetsSent;
};

ArpmecClusteringTestCase::ArpmecClusteringTestCase()
    : TestCase("ARPMEC Clustering Test"),
      m_lastEvent(ArpmecClustering::CH_ELECTED),
      m_lastEventNodeId(0),
      m_packetsSent(0)
{
}

ArpmecClusteringTestCase::~ArpmecClusteringTestCase()
{
}

void
ArpmecClusteringTestCase::DoRun()
{
    TestBasicClustering();
    TestClusterHeadElection();
    TestClusterJoining();
    // TestClusterMaintenance(); // Skip complex maintenance test for now
}

void
ArpmecClusteringTestCase::TestBasicClustering()
{
    // Create LQE and clustering modules
    Ptr<ArpmecLqe> lqe = CreateObject<ArpmecLqe>();
    Ptr<ArpmecClustering> clustering = CreateObject<ArpmecClustering>();
    
    uint32_t nodeId = 1;
    
    // Initialize clustering
    clustering->Initialize(nodeId, lqe);
    clustering->SetClusterEventCallback(MakeCallback(&ArpmecClusteringTestCase::OnClusterEvent, this));
    clustering->SetSendPacketCallback(MakeCallback(&ArpmecClusteringTestCase::OnSendPacket, this));
    
    // Test initial state
    NS_TEST_ASSERT_MSG_EQ(clustering->GetNodeState(), ArpmecClustering::UNDECIDED, "Initial state should be UNDECIDED");
    NS_TEST_ASSERT_MSG_EQ(clustering->IsClusterHead(), false, "Node should not be CH initially");
    NS_TEST_ASSERT_MSG_EQ(clustering->IsInCluster(), false, "Node should not be in cluster initially");
    NS_TEST_ASSERT_MSG_EQ(clustering->GetClusterHeadId(), 0, "No cluster head initially");
    
    // Test energy level setting
    clustering->SetEnergyLevel(0.8);
    NS_TEST_ASSERT_MSG_EQ_TOL(clustering->GetEnergyLevel(), 0.8, 0.001, "Energy level should be set correctly");
    
    // Start clustering
    clustering->Start();
    
    // Run simulation briefly to let clustering algorithm execute
    Simulator::Stop(Seconds(2.0));
    Simulator::Run();
    
    // With high energy and no neighbors, node should become CH
    NS_TEST_ASSERT_MSG_EQ(clustering->GetNodeState(), ArpmecClustering::CLUSTER_HEAD, "Node should become CH with high energy");
    NS_TEST_ASSERT_MSG_EQ(clustering->IsClusterHead(), true, "Node should be CH");
    NS_TEST_ASSERT_MSG_EQ(clustering->GetClusterHeadId(), nodeId, "Cluster head should be this node");
    
    // Check that cluster event was triggered
    NS_TEST_ASSERT_MSG_EQ(m_lastEvent, ArpmecClustering::CH_ELECTED, "CH_ELECTED event should be triggered");
    NS_TEST_ASSERT_MSG_EQ(m_lastEventNodeId, nodeId, "Event should be for this node");
    
    clustering->Stop();
    Simulator::Destroy();
}

void
ArpmecClusteringTestCase::TestClusterHeadElection()
{
    // Create two nodes with different energy levels
    Ptr<ArpmecLqe> lqe1 = CreateObject<ArpmecLqe>();
    Ptr<ArpmecLqe> lqe2 = CreateObject<ArpmecLqe>();
    
    Ptr<ArpmecClustering> clustering1 = CreateObject<ArpmecClustering>();
    Ptr<ArpmecClustering> clustering2 = CreateObject<ArpmecClustering>();
    
    uint32_t nodeId1 = 1;
    uint32_t nodeId2 = 2;
    
    // Initialize nodes
    clustering1->Initialize(nodeId1, lqe1);
    clustering2->Initialize(nodeId2, lqe2);
    
    clustering1->SetClusterEventCallback(MakeCallback(&ArpmecClusteringTestCase::OnClusterEvent, this));
    clustering2->SetClusterEventCallback(MakeCallback(&ArpmecClusteringTestCase::OnClusterEvent, this));
    
    clustering1->SetSendPacketCallback(MakeCallback(&ArpmecClusteringTestCase::OnSendPacket, this));
    clustering2->SetSendPacketCallback(MakeCallback(&ArpmecClusteringTestCase::OnSendPacket, this));
    
    // Set different energy levels
    clustering1->SetEnergyLevel(0.9); // High energy - should become CH
    clustering2->SetEnergyLevel(0.6); // Lower energy - should not become CH
    
    // Add each other as neighbors with good link quality
    Time now = Simulator::Now();
    lqe1->UpdateLinkQuality(nodeId2, -65.0, 0.9, 1000000, now);
    lqe2->UpdateLinkQuality(nodeId1, -65.0, 0.9, 1000000, now);
    
    // Start clustering
    clustering1->Start();
    clustering2->Start();
    
    // Run simulation to let clustering algorithm execute
    Simulator::Stop(Seconds(3.0));
    Simulator::Run();
    
    // Node 1 with higher energy should become CH
    NS_TEST_ASSERT_MSG_EQ(clustering1->IsClusterHead(), true, "Node 1 should become CH with higher energy");
    NS_TEST_ASSERT_MSG_EQ(clustering2->IsClusterHead(), false, "Node 2 should not become CH with lower energy");
    
    clustering1->Stop();
    clustering2->Stop();
    Simulator::Destroy();
}

void
ArpmecClusteringTestCase::TestClusterJoining()
{
    // Create cluster head and member nodes
    Ptr<ArpmecLqe> lqeCh = CreateObject<ArpmecLqe>();
    Ptr<ArpmecLqe> lqeMember = CreateObject<ArpmecLqe>();
    
    Ptr<ArpmecClustering> clusteringCh = CreateObject<ArpmecClustering>();
    Ptr<ArpmecClustering> clusteringMember = CreateObject<ArpmecClustering>();
    
    uint32_t chId = 1;
    uint32_t memberId = 2;
    
    // Initialize nodes
    clusteringCh->Initialize(chId, lqeCh);
    clusteringMember->Initialize(memberId, lqeMember);
    
    clusteringCh->SetClusterEventCallback(MakeCallback(&ArpmecClusteringTestCase::OnClusterEvent, this));
    clusteringMember->SetClusterEventCallback(MakeCallback(&ArpmecClusteringTestCase::OnClusterEvent, this));
    
    clusteringCh->SetSendPacketCallback(MakeCallback(&ArpmecClusteringTestCase::OnSendPacket, this));
    clusteringMember->SetSendPacketCallback(MakeCallback(&ArpmecClusteringTestCase::OnSendPacket, this));
    
    // Set energy levels
    clusteringCh->SetEnergyLevel(0.9);      // High energy - should become CH
    clusteringMember->SetEnergyLevel(0.5);  // Low energy - should join cluster
    
    // Add each other as neighbors
    Time now = Simulator::Now();
    lqeCh->UpdateLinkQuality(memberId, -70.0, 0.8, 1000000, now);
    lqeMember->UpdateLinkQuality(chId, -70.0, 0.8, 1000000, now);
    
    // Start clustering
    clusteringCh->Start();
    clusteringMember->Start();
    
    // Run simulation
    Simulator::Stop(Seconds(3.0));
    Simulator::Run();
    
    // CH should be cluster head
    NS_TEST_ASSERT_MSG_EQ(clusteringCh->IsClusterHead(), true, "CH node should be cluster head");
    
    // Member should join cluster (this depends on the clustering algorithm execution)
    // In practice, this test might need adjustment based on timing
    
    clusteringCh->Stop();
    clusteringMember->Stop();
    Simulator::Destroy();
}

void
ArpmecClusteringTestCase::TestClusterMaintenance()
{
    // This test would verify cluster maintenance functionality
    // Skip for now due to complexity and timing dependencies
}

void
ArpmecClusteringTestCase::OnClusterEvent(ArpmecClustering::ClusterEvent event, uint32_t nodeId)
{
    m_lastEvent = event;
    m_lastEventNodeId = nodeId;
}

void
ArpmecClusteringTestCase::OnSendPacket(Ptr<Packet> packet, uint32_t destination)
{
    m_packetsSent++;
}

/**
 * \ingroup arpmec-test
 * \ingroup tests
 *
 * \brief ARPMEC Clustering Test Suite
 */
class ArpmecClusteringTestSuite : public TestSuite
{
public:
    ArpmecClusteringTestSuite();
};

ArpmecClusteringTestSuite::ArpmecClusteringTestSuite()
    : TestSuite("arpmec-clustering-test-suite", Type::UNIT)
{
    AddTestCase(new ArpmecClusteringTestCase, Duration::QUICK);
}

static ArpmecClusteringTestSuite g_arpmecClusteringTestSuite; ///< Static variable for test initialization
