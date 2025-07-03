/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * ARPMEC MEC Gateway Implementation
 *
 * This class implements the MEC (Mobile Edge Computing) Gateway functionality
 * described in the ARPMEC paper. The gateway is responsible for:
 * - Cluster cleanup and maintenance
 * - Edge-to-cloud communication coordination
 * - Distributed processing management
 */

#ifndef ARPMEC_MEC_GATEWAY_H
#define ARPMEC_MEC_GATEWAY_H

#include "ns3/object.h"
#include "ns3/ipv4-address.h"
#include "ns3/simulator.h"
#include "ns3/timer.h"
#include "ns3/callback.h"
#include "ns3/packet.h"
#include <map>
#include <set>
#include <vector>

namespace ns3
{
namespace arpmec
{

/**
 * \ingroup arpmec
 * \brief MEC Gateway for cluster management and edge computing
 *
 * This class implements the MEC Gateway functionality from the ARPMEC paper.
 * The gateway coordinates cluster operations and provides edge computing services.
 */
class ArpmecMecGateway : public Object
{
public:
    /// Gateway operational states
    enum GatewayState
    {
        INACTIVE = 0,        ///< Gateway is inactive
        ACTIVE = 1,          ///< Gateway is actively managing clusters
        OVERLOADED = 2       ///< Gateway is overloaded and delegating
    };

    /// Cluster management operations
    enum ClusterOperation
    {
        CLEANUP_ORPHANED = 0,    ///< Clean up orphaned cluster nodes
        MERGE_SMALL = 1,         ///< Merge small clusters
        SPLIT_LARGE = 2,         ///< Split oversized clusters
        REBALANCE = 3            ///< Rebalance cluster load
    };

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    /**
     * \brief Constructor
     */
    ArpmecMecGateway();

    /**
     * \brief Destructor
     */
    virtual ~ArpmecMecGateway();

    /**
     * \brief Initialize the MEC Gateway
     * \param gatewayId The unique ID of this gateway
     * \param coverageArea The area covered by this gateway (in meters)
     */
    void Initialize(uint32_t gatewayId, double coverageArea);

    /**
     * \brief Start the gateway operations
     */
    void Start();

    /**
     * \brief Stop the gateway operations
     */
    void Stop();

    /**
     * \brief Register a cluster with this gateway
     * \param clusterId The cluster ID
     * \param clusterHeadId The cluster head node ID
     * \param memberCount Number of members in the cluster
     */
    void RegisterCluster(uint32_t clusterId, uint32_t clusterHeadId, uint32_t memberCount);

    /**
     * \brief Unregister a cluster from this gateway
     * \param clusterId The cluster ID to remove
     */
    void UnregisterCluster(uint32_t clusterId);

    /**
     * \brief Update cluster status
     * \param clusterId The cluster ID
     * \param memberCount Current number of members
     * \param avgEnergyLevel Average energy level of cluster members
     */
    void UpdateClusterStatus(uint32_t clusterId, uint32_t memberCount, double avgEnergyLevel);

    /**
     * \brief Process a computation request from a cluster
     * \param clusterId The requesting cluster ID
     * \param requestSize Size of the computational task
     * \param deadline Maximum time to complete the task
     * \return True if request can be handled locally, false if needs cloud
     */
    bool ProcessComputationRequest(uint32_t clusterId, uint32_t requestSize, double deadline);

    /**
     * \brief Get the current gateway state
     * \return The current operational state
     */
    GatewayState GetGatewayState() const;

    /**
     * \brief Get the number of managed clusters
     * \return Number of clusters under this gateway's management
     */
    uint32_t GetManagedClusterCount() const;

    /**
     * \brief Set callback for cluster management events
     * \param callback The callback function
     */
    void SetClusterManagementCallback(Callback<void, ClusterOperation, uint32_t> callback);

    /**
     * \brief Set callback for cloud communication requests
     * \param callback The callback function for cloud requests
     */
    void SetCloudCommunicationCallback(Callback<bool, uint32_t, uint32_t> callback);

    // MEC Inter-Cluster Communication Methods
    /**
     * \brief Process a message from a cluster for inter-cluster communication
     * \param packet The packet to process
     * \param sourceCluster The ID of the source cluster
     */
    void ProcessClusterMessage(Ptr<Packet> packet, uint32_t sourceCluster);

    /**
     * \brief Forward packet to another MEC gateway
     * \param packet The packet to forward
     * \param targetGateway The target gateway ID
     */
    void ForwardToMecGateway(Ptr<Packet> packet, uint32_t targetGateway);

    /**
     * \brief Process cluster message locally within coverage area
     * \param packet The packet to process
     * \param sourceCluster The source cluster ID
     */
    void ProcessLocalClusterMessage(Ptr<Packet> packet, uint32_t sourceCluster);

    /**
     * \brief Forward packet to a specific cluster
     * \param packet The packet to forward
     * \param targetCluster The target cluster ID
     */
    void ForwardToCluster(Ptr<Packet> packet, uint32_t targetCluster);

    /**
     * \brief Find the best MEC gateway for load balancing
     * \param excludeCluster Cluster to exclude from consideration
     * \return Best gateway ID, or 0 if none found
     */
    uint32_t FindBestMecGateway(uint32_t excludeCluster);

    /**
     * \brief Get list of clusters covered by this gateway
     * \return Vector of cluster IDs
     */
    std::vector<uint32_t> GetCoveredClusters();

    /**
     * \brief Get cluster head node ID for a given cluster
     * \param clusterId The cluster ID
     * \return Cluster head node ID, or 0 if not found
     */
    uint32_t GetClusterHead(uint32_t clusterId);

    /**
     * \brief Convert gateway ID to node ID
     * \param gatewayId The gateway ID
     * \return Corresponding node ID
     */
    uint32_t GetNodeIdFromGatewayId(uint32_t gatewayId);

    /**
     * \brief Set callback for sending packets
     * \param callback The send packet callback
     */
    void SetSendCallback(Callback<void, Ptr<Packet>, uint32_t> callback);

    /**
     * \brief Add information about a known gateway
     * \param gatewayId The gateway ID
     * \param load Current load of the gateway
     */
    void AddKnownGateway(uint32_t gatewayId, double load);

private:
    /**
     * \brief Structure to store cluster information
     */
    struct ClusterInfo
    {
        uint32_t clusterId;          ///< Cluster identifier
        uint32_t clusterHeadId;      ///< ID of the cluster head
        uint32_t memberCount;        ///< Number of cluster members
        double avgEnergyLevel;       ///< Average energy level of members
        Time lastUpdate;             ///< Last time cluster was updated
        uint32_t computationLoad;    ///< Current computational load

        ClusterInfo() : clusterId(0), clusterHeadId(0), memberCount(0),
                       avgEnergyLevel(0.0), lastUpdate(Seconds(0)), computationLoad(0) {}
    };

    /**
     * \brief Perform periodic cluster cleanup operations
     */
    void PerformClusterCleanup();

    /**
     * \brief Check for orphaned clusters (no active cluster head)
     */
    void CheckOrphanedClusters();

    /**
     * \brief Balance load across clusters
     */
    void BalanceClusterLoad();

    /**
     * \brief Forward computation request to cloud
     * \param clusterId The requesting cluster
     * \param requestSize Size of the task
     * \return True if cloud accepts the request
     */
    bool ForwardToCloud(uint32_t clusterId, uint32_t requestSize);

    // Member variables
    uint32_t m_gatewayId;                                   ///< Gateway identifier
    double m_coverageArea;                                  ///< Coverage area in meters
    GatewayState m_state;                                   ///< Current gateway state
    std::map<uint32_t, ClusterInfo> m_managedClusters;     ///< Clusters under management
    uint32_t m_maxComputationCapacity;                     ///< Maximum computation capacity
    uint32_t m_currentComputationLoad;                     ///< Current computation load
    double m_energyThreshold;                               ///< Energy threshold for cluster operations

    // Timers
    Timer m_cleanupTimer;                                   ///< Timer for periodic cleanup
    Timer m_loadBalanceTimer;                               ///< Timer for load balancing

    // Configuration parameters
    Time m_cleanupInterval;                                 ///< Interval for cleanup operations
    Time m_clusterTimeout;                                  ///< Timeout for inactive clusters
    bool m_isRunning;                                       ///< Whether gateway is active

    // Callbacks
    Callback<void, ClusterOperation, uint32_t> m_clusterMgmtCallback;  ///< Cluster management callback
    Callback<bool, uint32_t, uint32_t> m_cloudCommCallback;           ///< Cloud communication callback
    Callback<void, Ptr<Packet>, uint32_t> m_sendCallback;             ///< Send packet callback

    // Inter-cluster communication structures
    /**
     * \brief Structure to store information about other gateways
     */
    struct GatewayInfo
    {
        uint32_t gatewayId;          ///< Gateway identifier
        double load;                 ///< Current load (0.0-1.0)
        Time lastUpdate;             ///< Last update time

        GatewayInfo() : gatewayId(0), load(0.0), lastUpdate(Seconds(0)) {}
    };
    
    std::map<uint32_t, GatewayInfo> m_knownGateways;       ///< Known gateways for load balancing

    // Static configuration
    static constexpr uint32_t DEFAULT_MAX_COMPUTATION_CAPACITY = 1000;  ///< Default computation capacity
    static constexpr double DEFAULT_ENERGY_THRESHOLD = 0.3;             ///< Default energy threshold
    static const Time DEFAULT_CLEANUP_INTERVAL;                         ///< Default cleanup interval
    static const Time DEFAULT_CLUSTER_TIMEOUT;                          ///< Default cluster timeout
};

} // namespace arpmec
} // namespace ns3

#endif /* ARPMEC_MEC_GATEWAY_H */
