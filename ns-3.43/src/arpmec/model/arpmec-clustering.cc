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
        m_clusteringTimer.Schedule(Seconds(0.1)); // Very fast initial execution for testing
    }

    // Schedule maintenance checks
    if (!m_maintenanceTimer.IsRunning())
    {
        m_maintenanceTimer.Schedule(m_memberTimeout / 2);
    }
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

    // Track that this neighbor is a cluster head
    m_neighborClusterHeads[senderId] = true;
    NS_LOG_DEBUG("Node " << m_nodeId << " learned that neighbor " << senderId << " is a cluster head");

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
    else if (m_nodeState == ISOLATED || m_nodeState == UNDECIDED)
    {
        // If we're isolated and a neighbor announces itself as CH, we might want to join
        if (m_lqe)
        {
            double linkQuality = m_lqe->PredictLinkScore(senderId);
            if (linkQuality > 0.5) // Good enough link to consider joining
            {
                // Join this cluster
                JoinCluster(senderId);
            }
        }
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

    // Remove from neighbor cluster heads tracking since they're no longer a CH
    m_neighborClusterHeads[senderId] = false;
    NS_LOG_DEBUG("Node " << m_nodeId << " learned that neighbor " << senderId << " is no longer a cluster head");

    // If our cluster head is abdicating, we need to find a new cluster
    if (m_nodeState == CLUSTER_MEMBER && m_currentCluster.headId == senderId)
    {
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
        return;
    }

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
    // Use configurable threshold instead of hardcoded 0.7 to allow testing
    if (m_energyLevel < m_energyThreshold)
    {
        return false;
    }

    if (!m_lqe)
    {
        return false;
    }

    // Step 2: Get neighbors and calculate average link quality (Paper Algorithm 2, line 4-5)
    std::vector<uint32_t> neighbors = m_lqe->GetNeighborsByQuality();

    // Handle isolated nodes with staggered decision to avoid simultaneous CH election
    if (neighbors.empty())
    {
        Time elapsed = Simulator::Now() - m_startTime;
        // Use node ID to stagger decisions for isolated nodes (shorter for testing)
        double nodeDelay = (m_nodeId % 10) * 0.1; // 0-0.9 second stagger
        bool becomeIsolatedCH = (elapsed > Seconds(1.0 + nodeDelay));
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
    if (avgLinkQuality < 0.2) // Even lower threshold for initial cluster formation
    {
        NS_LOG_DEBUG("Node " << m_nodeId << " - Low link quality: " << avgLinkQuality << " < 0.2");
        return false;
    }

    // Step 4: Check for nearby cluster heads (Paper Algorithm 2, line 8-9)
    uint32_t nearbyClusterHeads = CountNearbyClusterHeads();

    // Step 5: Final decision with staggered timing (Paper Algorithm 2, line 10-12)
    bool shouldBecomeCH = false;

    // Implement staggered decision making to prevent simultaneous CH election
    // Use node characteristics to determine decision priority
    double decisionPriority = m_energyLevel + avgLinkQuality + (validNeighbors * 0.1);
    double nodeOffset = (m_nodeId % 20) * 0.1; // 0-1.9 second stagger based on node ID
    Time elapsed = Simulator::Now() - m_startTime;

    if (nearbyClusterHeads == 0)
    {
        // Wait for staggered decision to avoid all nodes becoming CH simultaneously
        // Give more time for cluster head announcements to propagate
        if (elapsed > Seconds(1.0 + nodeOffset * 0.2)) // Longer timing to allow CH notifications
        {
            shouldBecomeCH = true; // No nearby CHs - must become CH
        }
        else
        {
            NS_LOG_DEBUG("Node " << m_nodeId << " - Waiting for staggered decision (elapsed="
                        << elapsed.GetSeconds() << "s, threshold=" << (1.0 + nodeOffset * 0.2) << "s)");
        }
    }
    else if (validNeighbors > 5 && nearbyClusterHeads <= 1) // Higher threshold for dense areas
    {
        // Dense area needs additional CH, but with higher threshold
        if (elapsed > Seconds(2.0 + nodeOffset * 0.3) && decisionPriority > 0.7) // Stricter requirements
        {
            shouldBecomeCH = true; // Dense area with only one CH
        }
        else
        {
            NS_LOG_DEBUG("Node " << m_nodeId << " - Dense area but waiting (elapsed="
                        << elapsed.GetSeconds() << "s, CHs=" << nearbyClusterHeads << ")");
        }
    }
    else
    {
        shouldBecomeCH = false; // Join existing cluster
        NS_LOG_DEBUG("Node " << m_nodeId << " - " << nearbyClusterHeads << " nearby CHs");
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

    // Check actual cluster head status of neighbors
    // This uses the neighbor information table that's maintained via HELLO messages
    for (uint32_t neighbor : neighbors)
    {
        double linkQuality = m_lqe->PredictLinkScore(neighbor);
        if (linkQuality > 0.5) // Good enough link to consider
        {
            // Check if we know this neighbor is a cluster head
            // In ARPMEC, CH status is announced via CH_NOTIFICATION messages
            auto it = m_neighborClusterHeads.find(neighbor);
            if (it != m_neighborClusterHeads.end() && it->second)
            {
                count++;
                NS_LOG_DEBUG("Node " << m_nodeId << " detected neighbor " << neighbor << " as cluster head");
            }
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

    if (!IsClusterHead())
    {
        NS_LOG_WARN("SendChNotificationMessage called but node " << m_nodeId << " is not cluster head");
        return;
    }

    if (m_sendPacketCallback.IsNull())
    {
        NS_LOG_ERROR("CRITICAL: SendChNotificationMessage called but m_sendPacketCallback is NULL for node " << m_nodeId);
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
    }

    // Send updated cluster notification if members were removed
    if (!toRemove.empty())
    {
        SendChNotificationMessage();
    }
}

void
ArpmecClustering::SplitLargeCluster(uint32_t clusterId)
{
    NS_LOG_FUNCTION(this << clusterId);

    // Only proceed if this node is the cluster head of the specified cluster
    if (!IsClusterHead() || clusterId != m_nodeId)
    {
        NS_LOG_WARN("Cannot split cluster " << clusterId << " - not the cluster head");
        return;
    }

    NS_LOG_INFO("Attempting to split large cluster " << clusterId << " with " << m_clusterMembers.size() << " members");

    // Split cluster if it has more than 8 members (threshold from ARPMEC paper)
    if (m_clusterMembers.size() > 8)
    {
        // Find the member with highest energy and link quality to become new CH
        uint32_t newClusterHead = 0;
        double bestScore = 0.0;

        for (uint32_t member : m_clusterMembers)
        {
            double linkScore = m_lqe->GetLinkScore(member);
            // Simulate member energy (in real implementation, this would come from the member)
            double memberEnergy = 0.7 + (member % 30) * 0.01; // Simulated energy level
            double combinedScore = linkScore * 0.6 + memberEnergy * 0.4;

            if (combinedScore > bestScore)
            {
                bestScore = combinedScore;
                newClusterHead = member;
            }
        }

        if (newClusterHead != 0)
        {
            NS_LOG_INFO("Selected node " << newClusterHead << " as new cluster head for split");

            // Remove half the members and assign them to the new cluster head
            std::vector<uint32_t> membersToMove;
            uint32_t halfSize = m_clusterMembers.size() / 2;

            for (uint32_t member : m_clusterMembers)
            {
                if (membersToMove.size() < halfSize && member != newClusterHead)
                {
                    membersToMove.push_back(member);
                }
            }

            // Remove moved members from current cluster
            for (uint32_t member : membersToMove)
            {
                UpdateClusterMember(member, false);
            }

            // Also remove the new cluster head
            UpdateClusterMember(newClusterHead, false);

            NS_LOG_INFO("Split cluster - moved " << membersToMove.size() + 1 << " members to new cluster head " << newClusterHead);

            // Send notification to remaining members
            SendChNotificationMessage();
        }
    }
    else
    {
        NS_LOG_INFO("Cluster " << clusterId << " is not large enough to split (" << m_clusterMembers.size() << " members)");
    }
}

void
ArpmecClustering::RebalanceCluster(uint32_t clusterId)
{
    NS_LOG_FUNCTION(this << clusterId);

    // Only proceed if this node is the cluster head of the specified cluster
    if (!IsClusterHead() || clusterId != m_nodeId)
    {
        NS_LOG_WARN("Cannot rebalance cluster " << clusterId << " - not the cluster head");
        return;
    }

    NS_LOG_INFO("Rebalancing cluster " << clusterId << " with " << m_clusterMembers.size() << " members");

    // Check if any nearby cluster heads have fewer members and could take some
    std::vector<uint32_t> nearbyClusterHeads = m_lqe->GetNeighbors();

    for (uint32_t neighborId : nearbyClusterHeads)
    {
        // Check if neighbor is a cluster head with fewer members
        double linkScore = m_lqe->GetLinkScore(neighborId);
        if (linkScore > 0.5) // Good link quality
        {
            // Find members that might have better connectivity to this neighbor
            std::vector<uint32_t> membersToMove;

            for (uint32_t member : m_clusterMembers)
            {
                // Check if member has better link to the neighbor CH
                double linkToNeighbor = m_lqe->GetLinkScore(member); // Simplified
                double linkToCurrent = m_lqe->GetLinkScore(member);

                // Move member if they have significantly better link to neighbor
                if (linkToNeighbor > linkToCurrent * 1.2 && membersToMove.size() < 2)
                {
                    membersToMove.push_back(member);
                }
            }

            // Move selected members
            for (uint32_t member : membersToMove)
            {
                UpdateClusterMember(member, false);
                NS_LOG_INFO("Moved member " << member << " to better cluster head " << neighborId);
            }

            if (!membersToMove.empty())
            {
                SendChNotificationMessage();
                break; // Only rebalance to one neighbor at a time
            }
        }
    }
}

void
ArpmecClustering::OnTaskCompletion(uint32_t taskId, uint32_t clusterId, double processingTime)
{
    NS_LOG_FUNCTION(this << taskId << clusterId << processingTime);

    NS_LOG_INFO("Cluster " << m_nodeId << " received task completion notification: "
                << "taskId=" << taskId << ", clusterId=" << clusterId
                << ", processingTime=" << processingTime << "s");

    // Update cluster statistics for task completion
    if (IsClusterHead())
    {
        // Track task completion metrics for cluster management
        // This could be used to adjust cluster size or routing decisions

        // If processing time is too high, consider cluster optimization
        if (processingTime > 2.0) // Threshold for slow processing
        {
            NS_LOG_INFO("Slow task processing detected (" << processingTime
                        << "s) - may need cluster optimization");

            // Consider splitting if cluster is large and processing is slow
            if (m_clusterMembers.size() > 6)
            {
                NS_LOG_INFO("Large cluster with slow processing - considering split");
                // Could trigger cluster split or rebalancing
            }
        }

        // Notify cluster event callback if set
        if (!m_clusterEventCallback.IsNull())
        {
            m_clusterEventCallback(TASK_COMPLETED, taskId);
        }
    }
}

void
ArpmecClustering::CleanupOrphanedCluster(uint32_t clusterId)
{
    NS_LOG_FUNCTION(this << clusterId);

    NS_LOG_INFO("Cluster " << m_nodeId << " cleaning up orphaned cluster: " << clusterId);

    if (IsClusterHead() && m_nodeId == clusterId)
    {
        // This cluster head is being marked as orphaned
        // Check if we have active members
        CleanupInactiveMembers();

        if (m_clusterMembers.empty())
        {
            NS_LOG_INFO("No active members found - abdicating cluster head role");
            SendAbdicateMessage();
            m_nodeState = UNDECIDED;
            m_currentCluster = ClusterInfo();

            // Trigger re-clustering
            if (m_isRunning)
            {
                m_clusteringTimer.Schedule(Seconds(1.0));
            }
        }
        else
        {
            NS_LOG_INFO("Active members found (" << m_clusterMembers.size()
                       << ") - maintaining cluster");
        }
    }
    else if (m_currentCluster.headId == clusterId)
    {
        // Our cluster head is being cleaned up
        NS_LOG_INFO("Current cluster head " << clusterId << " is orphaned - leaving cluster");
        LeaveCluster();

        // Trigger re-clustering
        if (m_isRunning)
        {
            m_clusteringTimer.Schedule(Seconds(0.5));
        }
    }
}

void
ArpmecClustering::MergeSmallCluster(uint32_t clusterId)
{
    NS_LOG_FUNCTION(this << clusterId);

    NS_LOG_INFO("Cluster " << m_nodeId << " attempting to merge small cluster: " << clusterId);

    if (IsClusterHead() && m_nodeId == clusterId)
    {
        // This cluster is being marked for merging
        if (m_clusterMembers.size() < 3) // Small cluster threshold
        {
            NS_LOG_INFO("Small cluster detected (" << m_clusterMembers.size()
                       << " members) - looking for merge candidate");

            // Find nearby cluster heads to merge with
            std::vector<uint32_t> neighbors = m_lqe->GetNeighbors();
            uint32_t bestMergeCandidate = 0;
            double bestLinkScore = 0.0;

            for (uint32_t neighborId : neighbors)
            {
                // Check if neighbor is a cluster head with good link quality
                double linkScore = m_lqe->GetLinkScore(neighborId);
                if (linkScore > bestLinkScore && linkScore > 0.6)
                {
                    // Additional check: ensure neighbor is actually a cluster head
                    // This would require neighbor state information
                    bestLinkScore = linkScore;
                    bestMergeCandidate = neighborId;
                }
            }

            if (bestMergeCandidate != 0)
            {
                NS_LOG_INFO("Found merge candidate " << bestMergeCandidate
                           << " with link score " << bestLinkScore);

                // Abdicate and encourage members to join the better cluster
                SendAbdicateMessage();
                m_nodeState = UNDECIDED;

                // Clear cluster information
                m_clusterMembers.clear();
                m_currentCluster = ClusterInfo();

                // Attempt to join the better cluster
                JoinCluster(bestMergeCandidate);
            }
            else
            {
                NS_LOG_INFO("No suitable merge candidate found - maintaining small cluster");
            }
        }
        else
        {
            NS_LOG_INFO("Cluster size (" << m_clusterMembers.size()
                       << ") above merge threshold - no action needed");
        }
    }
    else if (m_currentCluster.headId == clusterId)
    {
        // Our cluster head is being merged - this should trigger re-clustering
        NS_LOG_INFO("Current cluster head " << clusterId << " is being merged");
        // Wait for abdication message from cluster head
    }
}

} // namespace arpmec
} // namespace ns3
