/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * ARPMEC Adaptive Routing (Algorithm 3) Implementation
 */

#ifndef ARPMEC_ADAPTIVE_ROUTING_H
#define ARPMEC_ADAPTIVE_ROUTING_H

#include "ns3/object.h"
#include "ns3/ipv4-address.h"
#include "ns3/ptr.h"
#include "ns3/packet.h"
#include "ns3/callback.h"
#include "ns3/timer.h"
#include "arpmec-clustering.h"
#include "arpmec-lqe.h"

namespace ns3
{
namespace arpmec
{

/**
 * \ingroup arpmec
 * \brief ARPMEC Adaptive Routing Protocol Implementation (Algorithm 3)
 *
 * This class implements Algorithm 3 from the ARPMEC paper:
 * - Intra-cluster routing: Direct routing within same cluster
 * - Inter-cluster routing: Via cluster head to gateway/MEC
 * - Fallback routing: Traditional AODV when no cluster structure
 */
class ArpmecAdaptiveRouting : public Object
{
public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    /**
     * \brief Constructor
     */
    ArpmecAdaptiveRouting();

    /**
     * \brief Destructor
     */
    ~ArpmecAdaptiveRouting() override;

    /**
     * \brief Route decision types based on Algorithm 3
     */
    enum RouteDecision
    {
        INTRA_CLUSTER,    ///< Route within same cluster
        INTER_CLUSTER,    ///< Route via cluster head to other cluster
        GATEWAY_ROUTE,    ///< Route via gateway to MEC server
        AODV_FALLBACK     ///< Use traditional AODV routing
    };

    /**
     * \brief Routing information structure
     */
    struct RoutingInfo
    {
        RouteDecision decision;         ///< Routing decision
        uint32_t nextHop;              ///< Next hop node ID
        uint32_t clusterHead;          ///< Cluster head to use (if applicable)
        uint32_t gateway;              ///< Gateway node (if applicable)
        double routeQuality;           ///< Expected route quality score
        uint32_t hopCount;             ///< Expected hop count
    };

    /**
     * \brief Initialize adaptive routing
     * \param nodeId Local node ID
     * \param clustering Clustering module reference
     * \param lqe LQE module reference
     */
    void Initialize(uint32_t nodeId, Ptr<ArpmecClustering> clustering, Ptr<ArpmecLqe> lqe);

    /**
     * \brief Determine optimal routing decision for destination
     * \param destination Destination IP address
     * \param destinationNodeId Destination node ID
     * \return Routing information with decision and next hop
     */
    RoutingInfo DetermineRoute(Ipv4Address destination, uint32_t destinationNodeId);

    /**
     * \brief Check if destination is in same cluster
     * \param destinationNodeId Destination node ID
     * \return true if destination is in same cluster
     */
    bool IsInSameCluster(uint32_t destinationNodeId);

    /**
     * \brief Find cluster head for a given node
     * \param nodeId Node ID to find cluster head for
     * \return Cluster head node ID, 0 if not found
     */
    uint32_t FindClusterHead(uint32_t nodeId);

    /**
     * \brief Select best gateway for inter-cluster communication
     * \param targetCluster Target cluster head ID
     * \return Gateway node ID, 0 if not found
     */
    uint32_t SelectGateway(uint32_t targetCluster);

    /**
     * \brief Update cluster topology information
     * \param clusterId Cluster head ID
     * \param members List of cluster members
     */
    void UpdateClusterTopology(uint32_t clusterId, const std::vector<uint32_t>& members);

    /**
     * \brief Set callback for routing metrics
     * \param callback Callback for routing statistics
     */
    void SetRoutingMetricsCallback(Callback<void, RouteDecision, double> callback);

    /**
     * \brief Get routing statistics
     * \return Map of route decisions to usage counts
     */
    std::map<RouteDecision, uint32_t> GetRoutingStatistics() const;

    /**
     * \brief Reset routing statistics
     */
    void ResetStatistics();

private:
    // Node information
    uint32_t m_nodeId;                          ///< Local node ID
    Ptr<ArpmecClustering> m_clustering;         ///< Clustering module
    Ptr<ArpmecLqe> m_lqe;                      ///< LQE module

    // Cluster topology tracking
    std::map<uint32_t, std::vector<uint32_t>> m_clusterTopology;  ///< CH -> members mapping
    std::map<uint32_t, uint32_t> m_nodeToCluster;                ///< Node -> CH mapping
    std::set<uint32_t> m_gatewayNodes;                           ///< Known gateway nodes

    // Routing statistics
    std::map<RouteDecision, uint32_t> m_routingStats;           ///< Route decision statistics
    Callback<void, RouteDecision, double> m_metricsCallback;    ///< Metrics callback

    // Timers
    Timer m_topologyUpdateTimer;                                ///< Topology update timer

    /**
     * \brief Calculate route quality for intra-cluster route
     * \param destinationNodeId Destination node
     * \return Route quality score (0-1)
     */
    double CalculateIntraClusterQuality(uint32_t destinationNodeId);

    /**
     * \brief Calculate route quality for inter-cluster route
     * \param targetClusterHead Target cluster head
     * \return Route quality score (0-1)
     */
    double CalculateInterClusterQuality(uint32_t targetClusterHead);

    /**
     * \brief Find best intra-cluster route
     * \param destinationNodeId Destination node
     * \return Routing information
     */
    RoutingInfo FindIntraClusterRoute(uint32_t destinationNodeId);

    /**
     * \brief Find best inter-cluster route
     * \param destinationNodeId Destination node
     * \return Routing information
     */
    RoutingInfo FindInterClusterRoute(uint32_t destinationNodeId);

    /**
     * \brief Create fallback AODV route
     * \param destinationNodeId Destination node
     * \return Routing information for AODV fallback
     */
    RoutingInfo CreateAodvFallback(uint32_t destinationNodeId);

    /**
     * \brief Update topology periodically
     */
    void UpdateTopology();

    /**
     * \brief Record routing decision for statistics
     * \param decision Routing decision made
     * \param quality Route quality achieved
     */
    void RecordRoutingDecision(RouteDecision decision, double quality);
};

} // namespace arpmec
} // namespace ns3

#endif /* ARPMEC_ADAPTIVE_ROUTING_H */
