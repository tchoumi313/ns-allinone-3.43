/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * ARPMEC MEC Gateway Implementation
 */

#include "arpmec-mec-gateway.h"
#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ArpmecMecGateway");

namespace arpmec
{

// Static Time constants
const Time ArpmecMecGateway::DEFAULT_CLEANUP_INTERVAL = Seconds(10.0);
const Time ArpmecMecGateway::DEFAULT_CLUSTER_TIMEOUT = Seconds(30.0);

NS_OBJECT_ENSURE_REGISTERED(ArpmecMecGateway);

TypeId
ArpmecMecGateway::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::ArpmecMecGateway")
                            .SetParent<Object>()
                            .SetGroupName("Arpmec")
                            .AddConstructor<ArpmecMecGateway>()
                            .AddAttribute("MaxComputationCapacity",
                                        "Maximum computation capacity of the gateway",
                                        UintegerValue(DEFAULT_MAX_COMPUTATION_CAPACITY),
                                        MakeUintegerAccessor(&ArpmecMecGateway::m_maxComputationCapacity),
                                        MakeUintegerChecker<uint32_t>())
                            .AddAttribute("EnergyThreshold",
                                        "Energy threshold for cluster operations",
                                        DoubleValue(DEFAULT_ENERGY_THRESHOLD),
                                        MakeDoubleAccessor(&ArpmecMecGateway::m_energyThreshold),
                                        MakeDoubleChecker<double>(0.0, 1.0))
                            .AddAttribute("CleanupInterval",
                                        "Interval for cluster cleanup operations",
                                        TimeValue(DEFAULT_CLEANUP_INTERVAL),
                                        MakeTimeAccessor(&ArpmecMecGateway::m_cleanupInterval),
                                        MakeTimeChecker())
                            .AddAttribute("ClusterTimeout",
                                        "Timeout for inactive clusters",
                                        TimeValue(DEFAULT_CLUSTER_TIMEOUT),
                                        MakeTimeAccessor(&ArpmecMecGateway::m_clusterTimeout),
                                        MakeTimeChecker());
    return tid;
}

ArpmecMecGateway::ArpmecMecGateway()
    : m_gatewayId(0),
      m_coverageArea(0.0),
      m_state(INACTIVE),
      m_maxComputationCapacity(DEFAULT_MAX_COMPUTATION_CAPACITY),
      m_currentComputationLoad(0),
      m_energyThreshold(DEFAULT_ENERGY_THRESHOLD),
      m_cleanupInterval(DEFAULT_CLEANUP_INTERVAL),
      m_clusterTimeout(DEFAULT_CLUSTER_TIMEOUT),
      m_isRunning(false)
{
    NS_LOG_FUNCTION(this);

    // Set up timers
    m_cleanupTimer.SetFunction(&ArpmecMecGateway::PerformClusterCleanup, this);
    m_loadBalanceTimer.SetFunction(&ArpmecMecGateway::BalanceClusterLoad, this);
}

ArpmecMecGateway::~ArpmecMecGateway()
{
    NS_LOG_FUNCTION(this);
    Stop();
}

void
ArpmecMecGateway::Initialize(uint32_t gatewayId, double coverageArea)
{
    NS_LOG_FUNCTION(this << gatewayId << coverageArea);

    m_gatewayId = gatewayId;
    m_coverageArea = coverageArea;
}

void
ArpmecMecGateway::Start()
{
    NS_LOG_FUNCTION(this);

    if (m_isRunning)
    {
        return;
    }

    m_isRunning = true;
    m_state = ACTIVE;
    m_currentComputationLoad = 0;

    // Schedule first cleanup
    if (!m_cleanupTimer.IsRunning())
    {
        m_cleanupTimer.Schedule(m_cleanupInterval);
    }

    // Schedule load balancing
    if (!m_loadBalanceTimer.IsRunning())
    {
        m_loadBalanceTimer.Schedule(m_cleanupInterval * 2);
    }
}

void
ArpmecMecGateway::Stop()
{
    NS_LOG_FUNCTION(this);

    if (!m_isRunning)
    {
        return;
    }

    m_isRunning = false;
    m_state = INACTIVE;
    m_cleanupTimer.Cancel();
    m_loadBalanceTimer.Cancel();

    // Clear all managed clusters
    m_managedClusters.clear();
    m_currentComputationLoad = 0;
}

void
ArpmecMecGateway::RegisterCluster(uint32_t clusterId, uint32_t clusterHeadId, uint32_t memberCount)
{
    NS_LOG_FUNCTION(this << clusterId << clusterHeadId << memberCount);

    if (!m_isRunning)
    {
        return;
    }

    ClusterInfo info;
    info.clusterId = clusterId;
    info.clusterHeadId = clusterHeadId;
    info.memberCount = memberCount;
    info.lastUpdate = Simulator::Now();
    info.avgEnergyLevel = 1.0; // Default to full energy
    info.computationLoad = 0;

    m_managedClusters[clusterId] = info;
}

void
ArpmecMecGateway::UnregisterCluster(uint32_t clusterId)
{
    NS_LOG_FUNCTION(this << clusterId);

    auto it = m_managedClusters.find(clusterId);
    if (it != m_managedClusters.end())
    {
        // Reduce computation load
        m_currentComputationLoad -= it->second.computationLoad;
        m_managedClusters.erase(it);
    }
}

void
ArpmecMecGateway::UpdateClusterStatus(uint32_t clusterId, uint32_t memberCount, double avgEnergyLevel)
{
    NS_LOG_FUNCTION(this << clusterId << memberCount << avgEnergyLevel);

    auto it = m_managedClusters.find(clusterId);
    if (it != m_managedClusters.end())
    {
        it->second.memberCount = memberCount;
        it->second.avgEnergyLevel = avgEnergyLevel;
        it->second.lastUpdate = Simulator::Now();

        NS_LOG_DEBUG("Gateway " << m_gatewayId << " updated cluster " << clusterId
                     << " status: " << memberCount << " members, " << avgEnergyLevel << " avg energy");
    }
}

bool
ArpmecMecGateway::ProcessComputationRequest(uint32_t clusterId, uint32_t requestSize, double deadline)
{
    NS_LOG_FUNCTION(this << clusterId << requestSize << deadline);

    if (!m_isRunning || m_state == INACTIVE)
    {
        return false;
    }

    // Check if we have capacity to handle this request locally
    if (m_currentComputationLoad + requestSize <= m_maxComputationCapacity)
    {
        // Handle locally
        m_currentComputationLoad += requestSize;
        
        auto it = m_managedClusters.find(clusterId);
        if (it != m_managedClusters.end())
        {
            it->second.computationLoad += requestSize;
        }

        NS_LOG_INFO("Gateway " << m_gatewayId << " processing request of size " << requestSize 
                    << " locally for cluster " << clusterId);
        
        // Schedule completion (simplified - just reduce load after deadline)
        Simulator::Schedule(Seconds(deadline * 0.8), [this, clusterId, requestSize]() {
            m_currentComputationLoad -= requestSize;
            auto it = m_managedClusters.find(clusterId);
            if (it != m_managedClusters.end())
            {
                it->second.computationLoad -= requestSize;
            }
        });
        
        return true;
    }
    else
    {
        // Forward to cloud
        return ForwardToCloud(clusterId, requestSize);
    }
}

ArpmecMecGateway::GatewayState
ArpmecMecGateway::GetGatewayState() const
{
    return m_state;
}

uint32_t
ArpmecMecGateway::GetManagedClusterCount() const
{
    return m_managedClusters.size();
}

void
ArpmecMecGateway::SetClusterManagementCallback(Callback<void, ClusterOperation, uint32_t> callback)
{
    m_clusterMgmtCallback = callback;
}

void
ArpmecMecGateway::SetCloudCommunicationCallback(Callback<bool, uint32_t, uint32_t> callback)
{
    m_cloudCommCallback = callback;
}

void
ArpmecMecGateway::PerformClusterCleanup()
{
    NS_LOG_FUNCTION(this);

    if (!m_isRunning)
    {
        return;
    }

    CheckOrphanedClusters();
    
    // Schedule next cleanup
    if (m_isRunning)
    {
        m_cleanupTimer.Schedule(m_cleanupInterval);
    }
}

void
ArpmecMecGateway::CheckOrphanedClusters()
{
    NS_LOG_FUNCTION(this);

    Time now = Simulator::Now();
    std::vector<uint32_t> toRemove;

    for (auto& pair : m_managedClusters)
    {
        ClusterInfo& info = pair.second;
        
        // Check if cluster hasn't been updated recently
        if ((now - info.lastUpdate) > m_clusterTimeout)
        {
            toRemove.push_back(info.clusterId);
            NS_LOG_INFO("Gateway " << m_gatewayId << " detected orphaned cluster " << info.clusterId);
            
            // Trigger cleanup callback if available
            if (!m_clusterMgmtCallback.IsNull())
            {
                m_clusterMgmtCallback(CLEANUP_ORPHANED, info.clusterId);
            }
        }
        // Check if cluster has very low energy
        else if (info.avgEnergyLevel < m_energyThreshold)
        {
            NS_LOG_INFO("Gateway " << m_gatewayId << " cluster " << info.clusterId 
                        << " has low energy (" << info.avgEnergyLevel << ")");
            
            if (!m_clusterMgmtCallback.IsNull())
            {
                m_clusterMgmtCallback(REBALANCE, info.clusterId);
            }
        }
    }

    // Remove orphaned clusters
    for (uint32_t clusterId : toRemove)
    {
        UnregisterCluster(clusterId);
    }
}

void
ArpmecMecGateway::BalanceClusterLoad()
{
    NS_LOG_FUNCTION(this);

    if (!m_isRunning)
    {
        return;
    }

    // Update gateway state based on load
    double loadRatio = static_cast<double>(m_currentComputationLoad) / m_maxComputationCapacity;
    
    if (loadRatio > 0.9)
    {
        m_state = OVERLOADED;
        NS_LOG_INFO("Gateway " << m_gatewayId << " is overloaded (" << (loadRatio * 100) << "%)");
    }
    else if (loadRatio < 0.7)
    {
        m_state = ACTIVE;
    }

    // Check for clusters that might need merging or splitting
    for (auto& pair : m_managedClusters)
    {
        ClusterInfo& info = pair.second;
        
        if (info.memberCount < 2)
        {
            // Very small cluster - might need merging
            if (!m_clusterMgmtCallback.IsNull())
            {
                m_clusterMgmtCallback(MERGE_SMALL, info.clusterId);
            }
        }
        else if (info.memberCount > 10)
        {
            // Large cluster - might need splitting
            if (!m_clusterMgmtCallback.IsNull())
            {
                m_clusterMgmtCallback(SPLIT_LARGE, info.clusterId);
            }
        }
    }

    // Schedule next load balancing
    if (m_isRunning)
    {
        m_loadBalanceTimer.Schedule(m_cleanupInterval * 2);
    }
}

bool
ArpmecMecGateway::ForwardToCloud(uint32_t clusterId, uint32_t requestSize)
{
    NS_LOG_FUNCTION(this << clusterId << requestSize);

    if (!m_cloudCommCallback.IsNull())
    {
        bool cloudAccepted = m_cloudCommCallback(clusterId, requestSize);
        
        if (cloudAccepted)
        {
            NS_LOG_INFO("Gateway " << m_gatewayId << " forwarded request of size " 
                        << requestSize << " to cloud for cluster " << clusterId);
        }
        else
        {
            NS_LOG_WARN("Gateway " << m_gatewayId << " cloud rejected request for cluster " << clusterId);
        }
        
        return cloudAccepted;
    }
    
    return false;
}

// MEC Inter-Cluster Communication Methods
void
ArpmecMecGateway::ProcessClusterMessage(Ptr<Packet> packet, uint32_t sourceCluster)
{
    NS_LOG_FUNCTION(this << sourceCluster);
    
    // Determine best MEC gateway for inter-cluster communication
    uint32_t targetGateway = FindBestMecGateway(sourceCluster);
    
    if (targetGateway != 0 && targetGateway != m_gatewayId)
    {
        // Forward to target MEC gateway
        ForwardToMecGateway(packet, targetGateway);
        
        NS_LOG_INFO("Gateway " << m_gatewayId << " forwarded cluster message from cluster " 
                    << sourceCluster << " to gateway " << targetGateway);
    }
    else
    {
        // Handle locally or find alternative
        ProcessLocalClusterMessage(packet, sourceCluster);
    }
}

void
ArpmecMecGateway::ForwardToMecGateway(Ptr<Packet> packet, uint32_t targetGateway)
{
    NS_LOG_FUNCTION(this << targetGateway);
    
    // For now, create a simple header or use existing ARPMEC header
    // TODO: Add proper MEC header when packet headers are implemented
    
    // Send via routing protocol callback
    if (!m_sendCallback.IsNull())
    {
        m_sendCallback(packet, GetNodeIdFromGatewayId(targetGateway));
        
        NS_LOG_INFO("MEC Gateway " << m_gatewayId << " forwarded packet to gateway " << targetGateway);
    }
}

void
ArpmecMecGateway::ProcessLocalClusterMessage(Ptr<Packet> packet, uint32_t sourceCluster)
{
    NS_LOG_FUNCTION(this << sourceCluster);
    
    // Find target cluster in our coverage area
    std::vector<uint32_t> localClusters = GetCoveredClusters();
    
    for (uint32_t targetCluster : localClusters)
    {
        if (targetCluster != sourceCluster)
        {
            // Forward to target cluster via cluster head
            ForwardToCluster(packet, targetCluster);
            break;
        }
    }
}

void
ArpmecMecGateway::ForwardToCluster(Ptr<Packet> packet, uint32_t targetCluster)
{
    NS_LOG_FUNCTION(this << targetCluster);
    
    // Find cluster head of target cluster
    uint32_t clusterHead = GetClusterHead(targetCluster);
    
    if (clusterHead != 0)
    {
        // For now, send directly without custom header
        // TODO: Add proper cluster delivery header when packet headers are implemented
        
        // Send to cluster head
        if (!m_sendCallback.IsNull())
        {
            m_sendCallback(packet, clusterHead);
            
            NS_LOG_INFO("MEC Gateway " << m_gatewayId << " delivered packet to cluster " 
                        << targetCluster << " via cluster head " << clusterHead);
        }
    }
}

uint32_t
ArpmecMecGateway::FindBestMecGateway(uint32_t excludeCluster)
{
    NS_LOG_FUNCTION(this << excludeCluster);
    
    // Simple load balancing: find gateway with least load
    uint32_t bestGateway = 0;
    double minLoad = 1.0;
    
    for (auto& gateway : m_knownGateways)
    {
        uint32_t gatewayId = gateway.first;
        GatewayInfo& info = gateway.second;
        
        if (gatewayId != m_gatewayId && info.load < minLoad)
        {
            bestGateway = gatewayId;
            minLoad = info.load;
        }
    }
    
    return bestGateway;
}

std::vector<uint32_t>
ArpmecMecGateway::GetCoveredClusters()
{
    NS_LOG_FUNCTION(this);
    
    std::vector<uint32_t> clusters;
    
    // Return list of clusters within our coverage area
    for (auto& cluster : m_managedClusters)
    {
        clusters.push_back(cluster.first);
    }
    
    return clusters;
}

uint32_t
ArpmecMecGateway::GetClusterHead(uint32_t clusterId)
{
    NS_LOG_FUNCTION(this << clusterId);
    
    auto it = m_managedClusters.find(clusterId);
    if (it != m_managedClusters.end())
    {
        return it->second.clusterHeadId;
    }
    
    return 0; // No cluster head found
}

uint32_t
ArpmecMecGateway::GetNodeIdFromGatewayId(uint32_t gatewayId)
{
    NS_LOG_FUNCTION(this << gatewayId);
    
    // Convert gateway ID to node ID (simple mapping)
    // Gateway IDs are 101, 105, 109, etc. -> Node IDs are 1, 5, 9, etc.
    return gatewayId - 100;
}

void
ArpmecMecGateway::SetSendCallback(Callback<void, Ptr<Packet>, uint32_t> callback)
{
    m_sendCallback = callback;
}

void
ArpmecMecGateway::AddKnownGateway(uint32_t gatewayId, double load)
{
    NS_LOG_FUNCTION(this << gatewayId << load);
    
    GatewayInfo info;
    info.gatewayId = gatewayId;
    info.load = load;
    info.lastUpdate = Simulator::Now();
    
    m_knownGateways[gatewayId] = info;
    
    NS_LOG_INFO("Gateway " << m_gatewayId << " learned about gateway " << gatewayId 
                << " with load " << load);
}

} // namespace arpmec
} // namespace ns3
