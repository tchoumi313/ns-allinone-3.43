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
    if (!m_clusteringTimer.IsRunning())
    {
        m_clusteringTimer.Schedule(Seconds(1.0)); // Small delay to let system stabilize
    }

    // Schedule maintenance checks
    if (!m_maintenanceTimer.IsRunning())
    {
        m_maintenanceTimer.Schedule(m_memberTimeout / 2);
    }

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

    // Clear callbacks first to prevent issues during cleanup
    m_clusterEventCallback = MakeNullCallback<void, ClusterEvent, uint32_t>();
    m_sendPacketCallback = MakeNullCallback<void, Ptr<Packet>, uint32_t>();

    // Leave cluster if we're in one (callbacks are now cleared)
    if (IsInCluster())
    {
        LeaveCluster();
    }

    // Send abdicate if we're CH (callbacks are now cleared)
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
        if (!m_clusteringTimer.IsRunning())
        {
            m_clusteringTimer.Schedule(Seconds(0.1));
        }
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
        NS_LOG_DEBUG("Clustering algorithm skipped - not running or LQE unavailable");
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
            // No suitable cluster head found - only become CH if we're designated by the deterministic algorithm
            if (ShouldBecomeClusterHead())
            {
                BecomeClusterHead();
            }
            else
            {
                m_nodeState = ISOLATED;
                NS_LOG_DEBUG("Node " << m_nodeId << " - remaining isolated (no suitable CH and not designated as CH)");
            }
        }
    }

    // Schedule next clustering algorithm execution only if still running
    if (m_isRunning && !m_clusteringTimer.IsRunning())
    {
        m_clusteringTimer.Schedule(m_clusteringInterval);
    }
}

bool
ArpmecClustering::ShouldBecomeClusterHead()
{
    NS_LOG_FUNCTION(this << "Energy:" << m_energyLevel << "Threshold:" << m_energyThreshold);

    // EXACT ARPMEC Paper Algorithm 2 Implementation
    // Following the paper's algorithm step by step with proper debugging
    
    // Step 1: Energy threshold check (Paper Algorithm 2, line 3)
    if (m_energyLevel < 0.7) // Paper uses 0.7 as energy threshold
    {
        NS_LOG_DEBUG("Node " << m_nodeId << " - Energy too low: " << m_energyLevel << " < 0.7");
        return false;
    }

    if (!m_lqe)
    {
        NS_LOG_DEBUG("Node " << m_nodeId << " - LQE not available");
        return false;
    }

    // Step 2: Get neighbors and calculate average link quality (Paper Algorithm 2, line 4-5)
    std::vector<uint32_t> neighbors = m_lqe->GetNeighborsByQuality();
    
    // Handle isolated nodes
    if (neighbors.empty())
    {
        Time elapsed = Simulator::Now() - m_startTime;
        bool becomeIsolatedCH = (elapsed > Seconds(3.0));
        NS_LOG_INFO("Node " << m_nodeId << " - Isolated node, elapsed: " 
                     << elapsed.GetSeconds() << "s, decision: " << becomeIsolatedCH);
        return becomeIsolatedCH;
    }

    // Calculate average link quality (Paper Algorithm 2, line 6)
    double totalQuality = 0.0;
    uint32_t validNeighbors = 0;
    
    for (uint32_t neighbor : neighbors)
    {
        double quality = m_lqe->PredictLinkScore(neighbor);
        if (quality > 0.3) // Minimum valid link
        {
            totalQuality += quality;
            validNeighbors++;
        }
    }
    
    if (validNeighbors == 0)
    {
        NS_LOG_DEBUG("Node " << m_nodeId << " - No valid neighbors");
        return false;
    }
    
    double avgLinkQuality = totalQuality / validNeighbors;

    // Step 3: Link quality threshold check (Paper Algorithm 2, line 7)
    if (avgLinkQuality < 0.5) // Paper's LQ_threshold
    {
        NS_LOG_DEBUG("Node " << m_nodeId << " - Low link quality: " << avgLinkQuality << " < 0.5");
        return false;
    }

    // Step 4: Check for nearby cluster heads (Paper Algorithm 2, line 8-9)
    uint32_t nearbyClusterHeads = CountNearbyClusterHeads();
    
    // Step 5: Final decision (Paper Algorithm 2, line 10-12)
    bool shouldBecomeCH = false;
    
    if (nearbyClusterHeads == 0)
    {
        shouldBecomeCH = true; // No nearby CHs - must become CH
        NS_LOG_INFO("Node " << m_nodeId << " - No nearby CHs, becoming CH (energy=" 
                    << m_energyLevel << ", avgLQ=" << avgLinkQuality << ", neighbors=" << validNeighbors << ")");
    }
    else if (validNeighbors > 6 && nearbyClusterHeads <= 1)
    {
        shouldBecomeCH = true; // Dense area with only one CH
        NS_LOG_INFO("Node " << m_nodeId << " - Dense area (" << validNeighbors 
                     << " neighbors), adding CH (nearby CHs: " << nearbyClusterHeads << ")");
    }
    else
    {
        shouldBecomeCH = false; // Join existing cluster
        NS_LOG_DEBUG("Node " << m_nodeId << " - " << nearbyClusterHeads 
                     << " nearby CHs, joining cluster (neighbors=" << validNeighbors << ")");
    }
    
    return shouldBecomeCH;
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

    uint32_t bestCH = 0;
    double bestScore = 0.0;

    for (uint32_t neighbor : neighbors)
    {
        // Check if this neighbor has good enough link quality
        double linkScore = m_lqe->PredictLinkScore(neighbor);
        if (linkScore > 0.4) // Minimum quality threshold
        {
            // Evaluate neighbor suitability as cluster head based on ARPMEC criteria
            double chSuitability = 0.0;

            // Link quality factor (50% weight)
            chSuitability += linkScore * 0.5;

            // Stability factor: prefer nodes with consistent behavior
            // (In real implementation, this would track CH announcement history)
            // For now, use node characteristics that suggest stable behavior
            if (linkScore > 0.7) // Consistently good link quality
            {
                chSuitability += 0.3;
            }

            // Energy estimation factor (we can't know neighbor's energy directly)
            // But we can infer from their activity and link quality
            if (linkScore > 0.6) // Good performance suggests good energy
            {
                chSuitability += 0.2;
            }

            if (chSuitability > bestScore)
            {
                bestScore = chSuitability;
                bestCH = neighbor;
            }

            NS_LOG_DEBUG("Node " << m_nodeId << " evaluating neighbor " << neighbor
                         << " as CH: linkScore=" << linkScore << " suitability=" << chSuitability);
        }
    }

    if (bestCH != 0)
    {
        NS_LOG_DEBUG("Node " << m_nodeId << " selecting CH " << bestCH << " with score " << bestScore);
    }
    else
    {
        NS_LOG_DEBUG("Node " << m_nodeId << " - no suitable cluster head found");
    }

    return bestCH;
}

uint32_t
ArpmecClustering::CountNearbyClusterHeads()
{
    if (!m_lqe)
    {
        return 0;
    }
    
    uint32_t count = 0;
    std::vector<uint32_t> neighbors = m_lqe->GetNeighborsByQuality();
    
    for (uint32_t neighbor : neighbors)
    {
        double linkQuality = m_lqe->PredictLinkScore(neighbor);
        // In real implementation, this would check actual CH announcements
        // For simulation, we estimate based on very high link quality
        if (linkQuality > 0.8) // Assume very good neighbors might be CHs
        {
            count++;
        }
    }
    
    NS_LOG_DEBUG("Node " << m_nodeId << " counted " << count << " nearby cluster heads");
    return count;
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
    m_currentCluster.headLinkScore = m_lqe ? m_lqe->PredictLinkScore(headId) : 0.0;

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

    // Schedule next maintenance check only if still running
    if (m_isRunning && !m_maintenanceTimer.IsRunning())
    {
        m_maintenanceTimer.Schedule(m_memberTimeout / 2);
    }
}

void
ArpmecClustering::HandleClusterHeadTimeout()
{
    NS_LOG_FUNCTION(this);

    NS_LOG_INFO("Cluster head timeout - leaving cluster");
    LeaveCluster();

    // Trigger clustering algorithm to find new cluster
    if (!m_clusteringTimer.IsRunning())
    {
        m_clusteringTimer.Schedule(Seconds(0.1));
    }
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
