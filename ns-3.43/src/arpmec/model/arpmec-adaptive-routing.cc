/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * ARPMEC Adaptive Routing (Algorithm 3) Implementation
 */

#include "arpmec-adaptive-routing.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/double.h"
#include <algorithm>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ArpmecAdaptiveRouting");

namespace arpmec
{

NS_OBJECT_ENSURE_REGISTERED(ArpmecAdaptiveRouting);

TypeId
ArpmecAdaptiveRouting::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::ArpmecAdaptiveRouting")
                            .SetParent<Object>()
                            .SetGroupName("Arpmec")
                            .AddConstructor<ArpmecAdaptiveRouting>();
    return tid;
}

ArpmecAdaptiveRouting::ArpmecAdaptiveRouting()
    : m_nodeId(0)
{
    NS_LOG_FUNCTION(this);
    
    // Initialize routing statistics
    m_routingStats[INTRA_CLUSTER] = 0;
    m_routingStats[INTER_CLUSTER] = 0;
    m_routingStats[GATEWAY_ROUTE] = 0;
    m_routingStats[AODV_FALLBACK] = 0;

    // Set up topology update timer
    m_topologyUpdateTimer.SetFunction(&ArpmecAdaptiveRouting::UpdateTopology, this);
}

ArpmecAdaptiveRouting::~ArpmecAdaptiveRouting()
{
    NS_LOG_FUNCTION(this);
    m_topologyUpdateTimer.Cancel();
}

void
ArpmecAdaptiveRouting::Initialize(uint32_t nodeId, Ptr<ArpmecClustering> clustering, Ptr<ArpmecLqe> lqe)
{
    NS_LOG_FUNCTION(this << nodeId);

    m_nodeId = nodeId;
    m_clustering = clustering;
    m_lqe = lqe;

    // Start topology updates
    m_topologyUpdateTimer.Schedule(Seconds(5.0));

    NS_LOG_INFO("Adaptive routing initialized for node " << m_nodeId);
}

ArpmecAdaptiveRouting::RoutingInfo
ArpmecAdaptiveRouting::DetermineRoute(Ipv4Address destination, uint32_t destinationNodeId)
{
    NS_LOG_FUNCTION(this << destination << destinationNodeId);

    // Algorithm 3 from ARPMEC paper implementation
    RoutingInfo routeInfo;

    // Step 1: Check if destination is within our cluster (Algorithm 3, line 1)
    if (IsInSameCluster(destinationNodeId))
    {
        NS_LOG_DEBUG("Destination " << destinationNodeId << " is in same cluster");
        routeInfo = FindIntraClusterRoute(destinationNodeId);
        RecordRoutingDecision(routeInfo.decision, routeInfo.routeQuality);
        return routeInfo;
    }

    // Step 2: Check if destination is in another known cluster (Algorithm 3, line 3)
    uint32_t targetClusterHead = FindClusterHead(destinationNodeId);
    if (targetClusterHead != 0)
    {
        NS_LOG_DEBUG("Destination " << destinationNodeId << " is in cluster " << targetClusterHead);
        routeInfo = FindInterClusterRoute(destinationNodeId);
        RecordRoutingDecision(routeInfo.decision, routeInfo.routeQuality);
        return routeInfo;
    }

    // Step 3: Fallback to traditional AODV routing (Algorithm 3, line 8)
    NS_LOG_DEBUG("Using AODV fallback for destination " << destinationNodeId);
    routeInfo = CreateAodvFallback(destinationNodeId);
    RecordRoutingDecision(routeInfo.decision, routeInfo.routeQuality);
    return routeInfo;
}

bool
ArpmecAdaptiveRouting::IsInSameCluster(uint32_t destinationNodeId)
{
    if (!m_clustering)
    {
        return false;
    }

    // Check if we're in a cluster
    if (!m_clustering->IsInCluster())
    {
        return false;
    }

    // Get our cluster head
    uint32_t ourClusterHead = m_clustering->GetClusterHeadId();
    
    // Check if destination is in our cluster
    auto clusterIt = m_clusterTopology.find(ourClusterHead);
    if (clusterIt != m_clusterTopology.end())
    {
        const std::vector<uint32_t>& members = clusterIt->second;
        return std::find(members.begin(), members.end(), destinationNodeId) != members.end();
    }

    return false;
}

uint32_t
ArpmecAdaptiveRouting::FindClusterHead(uint32_t nodeId)
{
    auto nodeIt = m_nodeToCluster.find(nodeId);
    if (nodeIt != m_nodeToCluster.end())
    {
        return nodeIt->second;
    }
    return 0; // Not found
}

uint32_t
ArpmecAdaptiveRouting::SelectGateway(uint32_t targetCluster)
{
    // For now, use cluster head as gateway
    // In a real implementation, this would select optimal gateway nodes
    if (m_clustering && m_clustering->IsClusterHead())
    {
        return m_nodeId; // We are the gateway
    }
    
    // Return our cluster head as gateway
    if (m_clustering && m_clustering->IsInCluster())
    {
        return m_clustering->GetClusterHeadId();
    }
    
    return 0; // No gateway found
}

ArpmecAdaptiveRouting::RoutingInfo
ArpmecAdaptiveRouting::FindIntraClusterRoute(uint32_t destinationNodeId)
{
    NS_LOG_FUNCTION(this << destinationNodeId);

    RoutingInfo routeInfo;
    routeInfo.decision = INTRA_CLUSTER;
    routeInfo.clusterHead = m_clustering->GetClusterHeadId();
    routeInfo.gateway = 0;
    routeInfo.hopCount = 1; // Assume direct connection within cluster

    if (!m_lqe)
    {
        routeInfo.nextHop = destinationNodeId;
        routeInfo.routeQuality = 0.5; // Default quality
        return routeInfo;
    }

    // Check if we have direct link to destination
    std::vector<uint32_t> neighbors = m_lqe->GetNeighborsByQuality();
    bool directLink = std::find(neighbors.begin(), neighbors.end(), destinationNodeId) != neighbors.end();

    if (directLink)
    {
        // Direct intra-cluster route
        routeInfo.nextHop = destinationNodeId;
        routeInfo.routeQuality = CalculateIntraClusterQuality(destinationNodeId);
        routeInfo.hopCount = 1;
        NS_LOG_DEBUG("Direct intra-cluster route to " << destinationNodeId << 
                     " quality=" << routeInfo.routeQuality);
    }
    else
    {
        // Route via cluster head
        routeInfo.nextHop = routeInfo.clusterHead;
        routeInfo.routeQuality = CalculateIntraClusterQuality(routeInfo.clusterHead) * 0.8; // Penalty for extra hop
        routeInfo.hopCount = 2;
        NS_LOG_DEBUG("Intra-cluster route via CH " << routeInfo.clusterHead << 
                     " to " << destinationNodeId << " quality=" << routeInfo.routeQuality);
    }

    return routeInfo;
}

ArpmecAdaptiveRouting::RoutingInfo
ArpmecAdaptiveRouting::FindInterClusterRoute(uint32_t destinationNodeId)
{
    NS_LOG_FUNCTION(this << destinationNodeId);

    RoutingInfo routeInfo;
    routeInfo.decision = INTER_CLUSTER;
    routeInfo.clusterHead = m_clustering->GetClusterHeadId();
    
    // Find target cluster head
    uint32_t targetClusterHead = FindClusterHead(destinationNodeId);
    routeInfo.gateway = SelectGateway(targetClusterHead);
    
    if (routeInfo.gateway == 0)
    {
        // No gateway available, fallback to AODV
        return CreateAodvFallback(destinationNodeId);
    }

    // Route to our cluster head first (if we're not the CH)
    if (m_clustering->IsClusterHead())
    {
        routeInfo.nextHop = routeInfo.gateway; // We are CH, route to gateway
        routeInfo.hopCount = 2; // CH -> Gateway -> Target CH -> Destination
    }
    else
    {
        routeInfo.nextHop = routeInfo.clusterHead; // Route to our CH first
        routeInfo.hopCount = 3; // Us -> CH -> Gateway -> Target CH -> Destination
    }

    routeInfo.routeQuality = CalculateInterClusterQuality(targetClusterHead);
    
    NS_LOG_DEBUG("Inter-cluster route to " << destinationNodeId << 
                 " via gateway " << routeInfo.gateway << 
                 " quality=" << routeInfo.routeQuality);

    return routeInfo;
}

ArpmecAdaptiveRouting::RoutingInfo
ArpmecAdaptiveRouting::CreateAodvFallback(uint32_t destinationNodeId)
{
    NS_LOG_FUNCTION(this << destinationNodeId);

    RoutingInfo routeInfo;
    routeInfo.decision = AODV_FALLBACK;
    routeInfo.nextHop = 0; // Will be determined by AODV
    routeInfo.clusterHead = 0;
    routeInfo.gateway = 0;
    routeInfo.routeQuality = 0.3; // Lower quality for AODV fallback
    routeInfo.hopCount = 0; // Unknown, will be determined by AODV

    NS_LOG_DEBUG("AODV fallback route for " << destinationNodeId);
    return routeInfo;
}

double
ArpmecAdaptiveRouting::CalculateIntraClusterQuality(uint32_t destinationNodeId)
{
    if (!m_lqe)
    {
        return 0.5; // Default quality
    }

    double linkScore = m_lqe->PredictLinkScore(destinationNodeId);
    
    // Intra-cluster routes get quality bonus for being within cluster
    double quality = linkScore * 1.2; // 20% bonus for intra-cluster
    return std::min(1.0, quality);
}

double
ArpmecAdaptiveRouting::CalculateInterClusterQuality(uint32_t targetClusterHead)
{
    if (!m_lqe)
    {
        return 0.4; // Default quality for inter-cluster
    }

    // Calculate quality to our cluster head
    uint32_t ourClusterHead = m_clustering->GetClusterHeadId();
    double chQuality = m_lqe->PredictLinkScore(ourClusterHead);
    
    // Inter-cluster routes have quality penalty due to multiple hops
    double quality = chQuality * 0.7; // 30% penalty for inter-cluster complexity
    return std::max(0.1, quality);
}

void
ArpmecAdaptiveRouting::UpdateClusterTopology(uint32_t clusterId, const std::vector<uint32_t>& members)
{
    NS_LOG_FUNCTION(this << clusterId << members.size());

    // Update cluster topology
    m_clusterTopology[clusterId] = members;
    
    // Update node to cluster mapping
    for (uint32_t member : members)
    {
        m_nodeToCluster[member] = clusterId;
    }

    NS_LOG_DEBUG("Updated topology for cluster " << clusterId << 
                 " with " << members.size() << " members");
}

void
ArpmecAdaptiveRouting::UpdateTopology()
{
    NS_LOG_FUNCTION(this);

    if (!m_clustering)
    {
        m_topologyUpdateTimer.Schedule(Seconds(5.0));
        return;
    }

    // Update our own cluster information
    if (m_clustering->IsClusterHead())
    {
        std::vector<uint32_t> members = m_clustering->GetClusterMembers();
        members.push_back(m_nodeId); // Include ourselves
        UpdateClusterTopology(m_nodeId, members);
    }

    // Schedule next update
    m_topologyUpdateTimer.Schedule(Seconds(5.0));
}

void
ArpmecAdaptiveRouting::SetRoutingMetricsCallback(Callback<void, RouteDecision, double> callback)
{
    m_metricsCallback = callback;
}

std::map<ArpmecAdaptiveRouting::RouteDecision, uint32_t>
ArpmecAdaptiveRouting::GetRoutingStatistics() const
{
    return m_routingStats;
}

void
ArpmecAdaptiveRouting::ResetStatistics()
{
    NS_LOG_FUNCTION(this);
    
    m_routingStats[INTRA_CLUSTER] = 0;
    m_routingStats[INTER_CLUSTER] = 0;
    m_routingStats[GATEWAY_ROUTE] = 0;
    m_routingStats[AODV_FALLBACK] = 0;
}

void
ArpmecAdaptiveRouting::RecordRoutingDecision(RouteDecision decision, double quality)
{
    m_routingStats[decision]++;
    
    if (!m_metricsCallback.IsNull())
    {
        m_metricsCallback(decision, quality);
    }

    NS_LOG_DEBUG("Recorded routing decision: " << decision << " quality: " << quality);
}

} // namespace arpmec
} // namespace ns3
