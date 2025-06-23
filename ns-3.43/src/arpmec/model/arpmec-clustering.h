/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * ARPMEC Clustering Protocol Module
 *
 * This module implements the clustering functionality from Algorithm 2 of the ARPMEC paper:
 * "Adaptive Routing Protocol for Mobile Edge Computing-based IoT Networks"
 *
 * Key features:
 * - Cluster Head (CH) election based on LQE and energy
 * - Cluster formation and maintenance
 * - Member node management
 * - Integration with LQE module for neighbor evaluation
 */

#ifndef ARPMEC_CLUSTERING_H
#define ARPMEC_CLUSTERING_H

#include "ns3/object.h"
#include "ns3/ipv4-address.h"
#include "ns3/simulator.h"
#include "ns3/timer.h"
#include "ns3/callback.h"
#include "ns3/packet.h"
#include "arpmec-lqe.h"
#include "arpmec-packet.h"
#include <map>
#include <set>
#include <vector>

namespace ns3
{
namespace arpmec
{

/**
 * \ingroup arpmec
 * \brief Clustering protocol module for ARPMEC
 *
 * This class implements the clustering functionality described in Algorithm 2
 * of the ARPMEC paper. It uses the LQE module to make intelligent CH election
 * decisions and manages cluster formation and maintenance.
 */
class ArpmecClustering : public Object
{
public:
    /// Node state in the clustering protocol
    enum NodeState
    {
        UNDECIDED = 0,      ///< Node hasn't decided its role yet
        CLUSTER_HEAD = 1,   ///< Node is acting as a cluster head
        CLUSTER_MEMBER = 2, ///< Node is a member of a cluster
        ISOLATED = 3        ///< Node has no neighbors or cluster
    };

    /// Cluster events for callbacks
    enum ClusterEvent
    {
        CH_ELECTED = 0,     ///< Node became cluster head
        JOINED_CLUSTER = 1, ///< Node joined a cluster
        LEFT_CLUSTER = 2,   ///< Node left a cluster
        CH_CHANGED = 3      ///< Cluster head changed
    };

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    /**
     * \brief Constructor
     */
    ArpmecClustering();

    /**
     * \brief Destructor
     */
    virtual ~ArpmecClustering();

    /**
     * \brief Initialize the clustering protocol
     * \param nodeId The ID of this node
     * \param lqe Pointer to the LQE module
     */
    void Initialize(uint32_t nodeId, Ptr<ArpmecLqe> lqe);

    /**
     * \brief Start the clustering protocol
     */
    void Start();

    /**
     * \brief Stop the clustering protocol
     */
    void Stop();

    /**
     * \brief Process a received HELLO message
     * \param senderId The ID of the sender node
     * \param hello The HELLO header
     */
    void ProcessHelloMessage(uint32_t senderId, const ArpmecHelloHeader& hello);

    /**
     * \brief Process a received JOIN message
     * \param senderId The ID of the sender node
     * \param join The JOIN header
     */
    void ProcessJoinMessage(uint32_t senderId, const ArpmecJoinHeader& join);

    /**
     * \brief Process a received CH_NOTIFICATION message
     * \param senderId The ID of the sender node
     * \param notification The CH_NOTIFICATION header
     */
    void ProcessChNotificationMessage(uint32_t senderId, const ArpmecChNotificationHeader& notification);

    /**
     * \brief Process a received ABDICATE message
     * \param senderId The ID of the sender node
     * \param abdicate The ABDICATE header
     */
    void ProcessAbdicateMessage(uint32_t senderId, const ArpmecAbdicateHeader& abdicate);

    /**
     * \brief Get the current node state
     * \return The current NodeState
     */
    NodeState GetNodeState() const;

    /**
     * \brief Get the current cluster head ID
     * \return The cluster head ID (0 if not in a cluster)
     */
    uint32_t GetClusterHeadId() const;

    /**
     * \brief Get the list of cluster members (if this node is CH)
     * \return Vector of member node IDs
     */
    std::vector<uint32_t> GetClusterMembers() const;

    /**
     * \brief Check if this node is a cluster head
     * \return True if this node is a CH
     */
    bool IsClusterHead() const;

    /**
     * \brief Check if this node is in a cluster
     * \return True if this node is in a cluster
     */
    bool IsInCluster() const;

    /**
     * \brief Set the energy level of this node
     * \param energy The current energy level (0.0 to 1.0)
     */
    void SetEnergyLevel(double energy);

    /**
     * \brief Get the current energy level
     * \return The current energy level (0.0 to 1.0)
     */
    double GetEnergyLevel() const;

    /**
     * \brief Set the energy threshold for CH election
     * \param threshold The energy threshold (0.0 to 1.0)
     */
    void SetEnergyThreshold(double threshold);

    /**
     * \brief Set callback for cluster events
     * \param callback The callback function
     */
    void SetClusterEventCallback(Callback<void, ClusterEvent, uint32_t> callback);

    /**
     * \brief Set callback for sending packets
     * \param callback The callback function for packet transmission
     */
    void SetSendPacketCallback(Callback<void, Ptr<Packet>, uint32_t> callback);

private:
    /**
     * \brief Structure to store cluster information
     */
    struct ClusterInfo
    {
        uint32_t headId;                    ///< Cluster head ID
        std::set<uint32_t> members;         ///< Set of member node IDs
        Time lastUpdate;                    ///< Last time cluster was updated
        double headLinkScore;               ///< Link score to the cluster head

        ClusterInfo() : headId(0), lastUpdate(Seconds(0)), headLinkScore(0.0) {}
    };

    /**
     * \brief Execute the clustering algorithm (Algorithm 2 from paper)
     */
    void ExecuteClusteringAlgorithm();

    /**
     * \brief Evaluate if this node should become a cluster head
     * \return True if this node should become CH
     */
    bool ShouldBecomeClusterHead();

    /**
     * \brief Select the best cluster head from neighbors
     * \return The node ID of the best CH candidate (0 if none)
     */
    uint32_t SelectBestClusterHead();

    /**
     * \brief Count nearby cluster heads based on link quality
     * \return Number of nearby cluster heads
     */
    uint32_t CountNearbyClusterHeads();

    /**
     * \brief Become a cluster head
     */
    void BecomeClusterHead();

    /**
     * \brief Join a cluster
     * \param headId The cluster head ID to join
     */
    void JoinCluster(uint32_t headId);

    /**
     * \brief Leave the current cluster
     */
    void LeaveCluster();

    /**
     * \brief Send a JOIN message to a cluster head
     * \param headId The cluster head ID
     */
    void SendJoinMessage(uint32_t headId);

    /**
     * \brief Send a CH_NOTIFICATION message to cluster members
     */
    void SendChNotificationMessage();

    /**
     * \brief Send an ABDICATE message when stepping down as CH
     */
    void SendAbdicateMessage();

    /**
     * \brief Check for cluster maintenance needs
     */
    void CheckClusterMaintenance();

    /**
     * \brief Handle cluster head timeout
     */
    void HandleClusterHeadTimeout();

    /**
     * \brief Update cluster member list
     * \param memberId The member node ID
     * \param isJoining True if joining, false if leaving
     */
    void UpdateClusterMember(uint32_t memberId, bool isJoining);

    /**
     * \brief Cleanup inactive cluster members
     */
    void CleanupInactiveMembers();

    // Member variables
    uint32_t m_nodeId;                                      ///< This node's ID
    NodeState m_nodeState;                                  ///< Current node state
    Ptr<ArpmecLqe> m_lqe;                                   ///< LQE module pointer
    ClusterInfo m_currentCluster;                           ///< Current cluster information
    std::set<uint32_t> m_clusterMembers;                    ///< Cluster members (if CH)
    double m_energyLevel;                                   ///< Current energy level
    double m_energyThreshold;                               ///< Energy threshold for CH election
    Time m_clusteringInterval;                              ///< Interval for clustering decisions
    Time m_memberTimeout;                                   ///< Timeout for cluster members
    Timer m_clusteringTimer;                                ///< Timer for periodic clustering
    Timer m_maintenanceTimer;                               ///< Timer for cluster maintenance
    bool m_isRunning;                                       ///< Whether clustering is active
    Time m_startTime;                                       ///< Time when clustering started

    // Callbacks
    Callback<void, ClusterEvent, uint32_t> m_clusterEventCallback;  ///< Cluster event callback
    Callback<void, Ptr<Packet>, uint32_t> m_sendPacketCallback;     ///< Packet send callback

    // ARPMEC clustering parameters from the paper
    static constexpr double DEFAULT_ENERGY_THRESHOLD = 0.7;     ///< Default energy threshold for CH election
    static constexpr double CLUSTER_HEAD_BONUS = 0.1;           ///< Bonus for being CH candidate
    static const Time DEFAULT_CLUSTERING_INTERVAL;              ///< Default clustering interval
    static const Time DEFAULT_MEMBER_TIMEOUT;                   ///< Default member timeout
};

} // namespace arpmec
} // namespace ns3

#endif /* ARPMEC_CLUSTERING_H */
