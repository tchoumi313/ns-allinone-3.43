/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * ARPMEC Clustering Protocol Implementation
 */

#include "arpmec-clustering.h"
#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/simulator.h"
#include "ns3/packet.h"
#include <algorithm>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ArpmecClustering");

namespace arpmec
{

// Static Time constants
const Time ArpmecClustering::DEFAULT_CLUSTERING_INTERVAL = Seconds(5.0);
const Time ArpmecClustering::DEFAULT_MEMBER_TIMEOUT = Seconds(15.0);

NS_OBJECT_ENSURE_REGISTERED(ArpmecClustering);

TypeId
ArpmecClustering::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::ArpmecClustering")
                            .SetParent<Object>()
                            .SetGroupName("Arpmec")
                            .AddConstructor<ArpmecClustering>()
                            .AddAttribute("EnergyThreshold",
                                        "Energy threshold for cluster head election",
                                        DoubleValue(DEFAULT_ENERGY_THRESHOLD),
                                        MakeDoubleAccessor(&ArpmecClustering::m_energyThreshold),
                                        MakeDoubleChecker<double>(0.0, 1.0))
                            .AddAttribute("ClusteringInterval",
                                        "Interval for clustering algorithm execution",
                                        TimeValue(DEFAULT_CLUSTERING_INTERVAL),
                                        MakeTimeAccessor(&ArpmecClustering::m_clusteringInterval),
                                        MakeTimeChecker())
                            .AddAttribute("MemberTimeout",
                                        "Timeout for cluster member inactivity",
                                        TimeValue(DEFAULT_MEMBER_TIMEOUT),
                                        MakeTimeAccessor(&ArpmecClustering::m_memberTimeout),
                                        MakeTimeChecker());
    return tid;
}

ArpmecClustering::ArpmecClustering()
    : m_nodeId(0),
      m_nodeState(UNDECIDED),
      m_energyLevel(1.0),
      m_energyThreshold(DEFAULT_ENERGY_THRESHOLD),
      m_clusteringInterval(DEFAULT_CLUSTERING_INTERVAL),
      m_memberTimeout(DEFAULT_MEMBER_TIMEOUT),
      m_isRunning(false)
{
    NS_LOG_FUNCTION(this);

    // Set up timers
    m_clusteringTimer.SetFunction(&ArpmecClustering::ExecuteClusteringAlgorithm, this);
    m_maintenanceTimer.SetFunction(&ArpmecClustering::CheckClusterMaintenance, this);
}

ArpmecClustering::~ArpmecClustering()
{
    NS_LOG_FUNCTION(this);
    Stop();
}

void
ArpmecClustering::Initialize(uint32_t nodeId, Ptr<ArpmecLqe> lqe)
{
    NS_LOG_FUNCTION(this << nodeId);

    m_nodeId = nodeId;
    m_lqe = lqe;

    NS_LOG_INFO("Clustering initialized for node " << m_nodeId);
}

void
ArpmecClustering::Start()
{
    NS_LOG_FUNCTION(this);

    if (m_isRunning)
    {
        return;
    }

    m_isRunning = true;
    m_nodeState = UNDECIDED;
    m_startTime = Simulator::Now(); // Track start time for neighbor discovery

    // Schedule first clustering algorithm execution
    m_clusteringTimer.Schedule(Seconds(1.0)); // Small delay to let system stabilize

    // Schedule maintenance checks
    m_maintenanceTimer.Schedule(m_memberTimeout / 2);

    NS_LOG_INFO("Clustering started for node " << m_nodeId);
}

void
ArpmecClustering::Stop()
{
    NS_LOG_FUNCTION(this);

    if (!m_isRunning)
    {
        return;
    }

    m_isRunning = false;
    m_clusteringTimer.Cancel();
    m_maintenanceTimer.Cancel();

    // Leave cluster if we're in one
    if (IsInCluster())
    {
        LeaveCluster();
    }

    // Send abdicate if we're CH
    if (IsClusterHead())
    {
        SendAbdicateMessage();
    }

    m_nodeState = UNDECIDED;
    m_clusterMembers.clear();
    m_currentCluster = ClusterInfo();

    NS_LOG_INFO("Clustering stopped for node " << m_nodeId);
}

void
ArpmecClustering::ProcessHelloMessage(uint32_t senderId, const ArpmecHelloHeader& hello)
{
    NS_LOG_FUNCTION(this << senderId);

    if (!m_isRunning || !m_lqe)
    {
        return;
    }

    // Update LQE information for this neighbor
    Time now = Simulator::Now();
    m_lqe->UpdateLinkQuality(senderId, hello.GetRssi(), hello.GetPdr(),
                            hello.GetTimestamp(), now);

    NS_LOG_DEBUG("Processed HELLO from node " << senderId <<
                 " RSSI=" << hello.GetRssi() << " PDR=" << hello.GetPdr());
}

void
ArpmecClustering::ProcessJoinMessage(uint32_t senderId, const ArpmecJoinHeader& join)
{
    NS_LOG_FUNCTION(this << senderId << join.GetChId());

    if (!m_isRunning || join.GetChId() != m_nodeId || !IsClusterHead())
    {
        return;
    }

    // Add the node as a cluster member
    UpdateClusterMember(senderId, true);

    // Send notification about cluster membership
    SendChNotificationMessage();

    NS_LOG_INFO("Node " << senderId << " joined cluster headed by " << m_nodeId);

    // Notify about cluster event
    if (!m_clusterEventCallback.IsNull())
    {
        m_clusterEventCallback(JOINED_CLUSTER, senderId);
    }
}

void
ArpmecClustering::ProcessChNotificationMessage(uint32_t senderId, const ArpmecChNotificationHeader& notification)
{
    NS_LOG_FUNCTION(this << senderId);

    if (!m_isRunning)
    {
        return;
    }

    // Update cluster information if we're a member of this cluster
    if (m_nodeState == CLUSTER_MEMBER && m_currentCluster.headId == senderId)
    {
        m_currentCluster.lastUpdate = Simulator::Now();

        // Update cluster member list
        const std::vector<uint32_t>& members = notification.GetClusterMembers();
        m_currentCluster.members.clear();
        for (uint32_t member : members)
        {
            m_currentCluster.members.insert(member);
        }

        NS_LOG_DEBUG("Updated cluster info from CH " << senderId);
    }
}

void
ArpmecClustering::ProcessAbdicateMessage(uint32_t senderId, const ArpmecAbdicateHeader& abdicate)
{
    NS_LOG_FUNCTION(this << senderId);

    if (!m_isRunning)
    {
        return;
    }

    // If our cluster head is abdicating, we need to find a new cluster
    if (m_nodeState == CLUSTER_MEMBER && m_currentCluster.headId == senderId)
    {
        NS_LOG_INFO("Cluster head " << senderId << " is abdicating, leaving cluster");
        LeaveCluster();

        // Trigger clustering algorithm to find new cluster
        m_clusteringTimer.Schedule(Seconds(0.1));
    }
}

ArpmecClustering::NodeState
ArpmecClustering::GetNodeState() const
{
    return m_nodeState;
}

uint32_t
ArpmecClustering::GetClusterHeadId() const
{
    if (IsClusterHead())
    {
        return m_nodeId;
    }
    else if (IsInCluster())
    {
        return m_currentCluster.headId;
    }
    return 0;
}

std::vector<uint32_t>
ArpmecClustering::GetClusterMembers() const
{
    std::vector<uint32_t> members;
    if (IsClusterHead())
    {
        for (uint32_t member : m_clusterMembers)
        {
            members.push_back(member);
        }
    }
    return members;
}

bool
ArpmecClustering::IsClusterHead() const
{
    return m_nodeState == CLUSTER_HEAD;
}

bool
ArpmecClustering::IsInCluster() const
{
    return m_nodeState == CLUSTER_MEMBER || m_nodeState == CLUSTER_HEAD;
}

void
ArpmecClustering::SetEnergyLevel(double energy)
{
    m_energyLevel = std::max(0.0, std::min(1.0, energy));
    NS_LOG_DEBUG("Energy level updated to " << m_energyLevel << " for node " << m_nodeId);
}

double
ArpmecClustering::GetEnergyLevel() const
{
    return m_energyLevel;
}

void
ArpmecClustering::SetEnergyThreshold(double threshold)
{
    m_energyThreshold = std::max(0.0, std::min(1.0, threshold));
}

void
ArpmecClustering::SetClusterEventCallback(Callback<void, ClusterEvent, uint32_t> callback)
{
    m_clusterEventCallback = callback;
}

void
ArpmecClustering::SetSendPacketCallback(Callback<void, Ptr<Packet>, uint32_t> callback)
{
    m_sendPacketCallback = callback;
}

// Private methods implementation

void
ArpmecClustering::ExecuteClusteringAlgorithm()
{
    NS_LOG_FUNCTION(this);

    if (!m_isRunning || !m_lqe)
    {
        return;
    }

    NS_LOG_DEBUG("Executing clustering algorithm for node " << m_nodeId);

    // Algorithm 2 from ARPMEC paper
    // Lines 7-15: Clustering algorithm

    if (ShouldBecomeClusterHead())
    {
        // Node has enough energy and good link quality - become CH
        if (m_nodeState != CLUSTER_HEAD)
        {
            BecomeClusterHead();
        }
    }
    else
    {
        // Find the best cluster head to join
        uint32_t bestCh = SelectBestClusterHead();

        if (bestCh != 0)
        {
            // Join the best cluster
            if (m_nodeState != CLUSTER_MEMBER || m_currentCluster.headId != bestCh)
            {
                JoinCluster(bestCh);
            }
        }
        else
        {
            // No suitable cluster head found - stay isolated or become CH if needed
            if (m_energyLevel > m_energyThreshold * 0.8) // Lower threshold for isolated nodes
            {
                BecomeClusterHead();
            }
            else
            {
                m_nodeState = ISOLATED;
            }
        }
    }

    // Schedule next clustering algorithm execution
    m_clusteringTimer.Schedule(m_clusteringInterval);
}

bool
ArpmecClustering::ShouldBecomeClusterHead()
{
    NS_LOG_FUNCTION(this << "Energy:" << m_energyLevel << "Threshold:" << m_energyThreshold);
    
    // Check energy requirement (Algorithm 2, line 9)
    if (m_energyLevel < m_energyThreshold)
    {
        NS_LOG_DEBUG("Node " << m_nodeId << " cannot be CH - insufficient energy (" 
                     << m_energyLevel << " < " << m_energyThreshold << ")");
        return false;
    }

    // Check if we have neighbors
    std::vector<uint32_t> neighbors = m_lqe->GetNeighborsByQuality();
    NS_LOG_DEBUG("Node " << m_nodeId << " has " << neighbors.size() << " neighbors");
    
    if (neighbors.empty())
    {
        // Give isolated nodes some time to discover neighbors before becoming CH
        Time timeSinceStart = Simulator::Now() - m_startTime;
        if (timeSinceStart > Seconds(5.0)) // Wait 5 seconds for neighbor discovery
        {
            NS_LOG_DEBUG("Node " << m_nodeId << " becoming CH - isolated after waiting");
            return true; 
        }
        else
        {
            NS_LOG_DEBUG("Node " << m_nodeId << " waiting for neighbors - time since start: " << timeSinceStart.GetSeconds() << "s");
            return false; // Wait for neighbors
        }
    }

    // If we already have enough cluster heads in the neighborhood, don't become one
    uint32_t existingCHs = 0;
    for (uint32_t neighbor : neighbors)
    {
        // In a real implementation, we'd track the CH status of neighbors
        // For now, use a simple heuristic: assume lower node IDs are more likely to be CHs
        if (neighbor < m_nodeId)
        {
            existingCHs++;
        }
    }
    
    // Limit cluster head density - no more than 1 CH per 5-7 nodes
    if (existingCHs > 0 && neighbors.size() < 7)
    {
        NS_LOG_DEBUG("Node " << m_nodeId << " - enough CHs in neighborhood (" << existingCHs << ")");
        return false;
    }

    // Check if we're the best candidate among our neighbors
    uint32_t bestNeighbor = m_lqe->GetBestNeighbor();
    if (bestNeighbor == 0)
    {
        NS_LOG_DEBUG("Node " << m_nodeId << " becoming CH - no good neighbors");
        return true; // No good neighbors, we should be CH
    }

    // Calculate our own "score" for CH election (energy + link quality)
    double ourScore = m_energyLevel + CLUSTER_HEAD_BONUS;
    double bestNeighborScore = m_lqe->PredictLinkScore(bestNeighbor) + 0.5; // Assume neighbor has decent energy
    
    NS_LOG_DEBUG("Node " << m_nodeId << " CH score: " << ourScore 
                 << " vs best neighbor " << bestNeighbor << " score: " << bestNeighborScore);

    // Become CH if our score is significantly better
    bool shouldBeCH = ourScore > bestNeighborScore + 0.2; // Increased threshold
    NS_LOG_DEBUG("Node " << m_nodeId << " CH decision: " << (shouldBeCH ? "YES" : "NO"));
    return shouldBeCH;
}

uint32_t
ArpmecClustering::SelectBestClusterHead()
{
    if (!m_lqe)
    {
        return 0;
    }

    // Get neighbors sorted by link quality
    std::vector<uint32_t> neighbors = m_lqe->GetNeighborsByQuality();

    for (uint32_t neighbor : neighbors)
    {
        // Check if this neighbor has good enough link quality
        double linkScore = m_lqe->PredictLinkScore(neighbor);
        if (linkScore > 0.5) // Minimum quality threshold
        {
            return neighbor; // Return the best quality neighbor
        }
    }

    return 0; // No suitable cluster head found
}

void
ArpmecClustering::BecomeClusterHead()
{
    NS_LOG_FUNCTION(this);

    // Leave current cluster if we're in one
    if (IsInCluster() && !IsClusterHead())
    {
        LeaveCluster();
    }

    m_nodeState = CLUSTER_HEAD;
    m_currentCluster.headId = m_nodeId;
    m_currentCluster.lastUpdate = Simulator::Now();

    // Clear existing members and start fresh
    m_clusterMembers.clear();

    // Send notification to announce ourselves as CH
    SendChNotificationMessage();

    NS_LOG_INFO("Node " << m_nodeId << " became cluster head");

    // Notify about cluster event
    if (!m_clusterEventCallback.IsNull())
    {
        m_clusterEventCallback(CH_ELECTED, m_nodeId);
    }
}

void
ArpmecClustering::JoinCluster(uint32_t headId)
{
    NS_LOG_FUNCTION(this << headId);

    // Leave current cluster if we're in a different one
    if (IsInCluster() && GetClusterHeadId() != headId)
    {
        LeaveCluster();
    }

    m_nodeState = CLUSTER_MEMBER;
    m_currentCluster.headId = headId;
    m_currentCluster.lastUpdate = Simulator::Now();
    m_currentCluster.headLinkScore = m_lqe->PredictLinkScore(headId);

    // Send JOIN message to the cluster head
    SendJoinMessage(headId);

    NS_LOG_INFO("Node " << m_nodeId << " joined cluster headed by " << headId);

    // Notify about cluster event
    if (!m_clusterEventCallback.IsNull())
    {
        m_clusterEventCallback(JOINED_CLUSTER, headId);
    }
}

void
ArpmecClustering::LeaveCluster()
{
    NS_LOG_FUNCTION(this);

    if (!IsInCluster())
    {
        return;
    }

    uint32_t oldHeadId = GetClusterHeadId();

    if (IsClusterHead())
    {
        // Send abdicate message to cluster members
        SendAbdicateMessage();
        m_clusterMembers.clear();
    }

    m_nodeState = UNDECIDED;
    m_currentCluster = ClusterInfo();

    NS_LOG_INFO("Node " << m_nodeId << " left cluster");

    // Notify about cluster event
    if (!m_clusterEventCallback.IsNull())
    {
        m_clusterEventCallback(LEFT_CLUSTER, oldHeadId);
    }
}

void
ArpmecClustering::SendJoinMessage(uint32_t headId)
{
    NS_LOG_FUNCTION(this << headId);

    if (m_sendPacketCallback.IsNull())
    {
        return;
    }

    // Create JOIN packet
    Ptr<Packet> packet = Create<Packet>();

    // Add type header
    TypeHeader typeHeader(ARPMEC_JOIN);
    packet->AddHeader(typeHeader);

    // Add JOIN header
    ArpmecJoinHeader joinHeader;
    joinHeader.SetNodeId(m_nodeId);
    joinHeader.SetChId(headId);
    packet->AddHeader(joinHeader);

    // Send packet
    m_sendPacketCallback(packet, headId);

    NS_LOG_DEBUG("Sent JOIN message from " << m_nodeId << " to CH " << headId);
}

void
ArpmecClustering::SendChNotificationMessage()
{
    NS_LOG_FUNCTION(this);

    if (!IsClusterHead() || m_sendPacketCallback.IsNull())
    {
        return;
    }

    // Create CH_NOTIFICATION packet
    Ptr<Packet> packet = Create<Packet>();

    // Add type header
    TypeHeader typeHeader(ARPMEC_CH_NOTIFICATION);
    packet->AddHeader(typeHeader);

    // Add CH_NOTIFICATION header
    ArpmecChNotificationHeader notificationHeader;
    notificationHeader.SetChId(m_nodeId);

    // Add cluster members
    std::vector<uint32_t> members;
    for (uint32_t member : m_clusterMembers)
    {
        members.push_back(member);
    }
    notificationHeader.SetClusterMembers(members);
    packet->AddHeader(notificationHeader);

    // Broadcast to all neighbors
    m_sendPacketCallback(packet, 0); // 0 means broadcast

    NS_LOG_DEBUG("Sent CH_NOTIFICATION from " << m_nodeId << " with " <<
                 m_clusterMembers.size() << " members");
}

void
ArpmecClustering::SendAbdicateMessage()
{
    NS_LOG_FUNCTION(this);

    if (!IsClusterHead() || m_sendPacketCallback.IsNull())
    {
        return;
    }

    // Create ABDICATE packet
    Ptr<Packet> packet = Create<Packet>();

    // Add type header
    TypeHeader typeHeader(ARPMEC_ABDICATE);
    packet->AddHeader(typeHeader);

    // Add ABDICATE header
    ArpmecAbdicateHeader abdicateHeader;
    abdicateHeader.SetChId(m_nodeId);
    packet->AddHeader(abdicateHeader);

    // Broadcast to all cluster members
    m_sendPacketCallback(packet, 0); // 0 means broadcast

    NS_LOG_DEBUG("Sent ABDICATE message from CH " << m_nodeId);
}

void
ArpmecClustering::CheckClusterMaintenance()
{
    NS_LOG_FUNCTION(this);

    if (!m_isRunning)
    {
        return;
    }

    Time now = Simulator::Now();

    // Check if cluster head is still reachable (for members)
    if (m_nodeState == CLUSTER_MEMBER)
    {
        if ((now - m_currentCluster.lastUpdate) > m_memberTimeout)
        {
            NS_LOG_INFO("Cluster head timeout for node " << m_nodeId);
            HandleClusterHeadTimeout();
        }
    }

    // Cleanup inactive members (for cluster heads)
    if (IsClusterHead())
    {
        CleanupInactiveMembers();
    }

    // Schedule next maintenance check
    m_maintenanceTimer.Schedule(m_memberTimeout / 2);
}

void
ArpmecClustering::HandleClusterHeadTimeout()
{
    NS_LOG_FUNCTION(this);

    NS_LOG_INFO("Cluster head timeout - leaving cluster");
    LeaveCluster();

    // Trigger clustering algorithm to find new cluster
    m_clusteringTimer.Schedule(Seconds(0.1));
}

void
ArpmecClustering::UpdateClusterMember(uint32_t memberId, bool isJoining)
{
    NS_LOG_FUNCTION(this << memberId << isJoining);

    if (!IsClusterHead())
    {
        return;
    }

    if (isJoining)
    {
        m_clusterMembers.insert(memberId);
        NS_LOG_DEBUG("Added member " << memberId << " to cluster");
    }
    else
    {
        m_clusterMembers.erase(memberId);
        NS_LOG_DEBUG("Removed member " << memberId << " from cluster");
    }
}

void
ArpmecClustering::CleanupInactiveMembers()
{
    NS_LOG_FUNCTION(this);

    if (!IsClusterHead() || !m_lqe)
    {
        return;
    }

    std::vector<uint32_t> toRemove;

    for (uint32_t member : m_clusterMembers)
    {
        if (!m_lqe->IsNeighborActive(member))
        {
            toRemove.push_back(member);
        }
    }

    for (uint32_t member : toRemove)
    {
        UpdateClusterMember(member, false);
        NS_LOG_INFO("Removed inactive member " << member << " from cluster");
    }

    // Send updated cluster notification if members were removed
    if (!toRemove.empty())
    {
        SendChNotificationMessage();
    }
}

} // namespace arpmec
} // namespace ns3
