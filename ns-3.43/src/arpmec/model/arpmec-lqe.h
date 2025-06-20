/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * ARPMEC Link Quality Estimation (LQE) Module
 *
 * This module implements the LQE functionality from Algorithm 2 of the ARPMEC paper:
 * "Adaptive Routing Protocol for Mobile Edge Computing-based IoT Networks"
 *
 * Key features:
 * - PDR (Packet Delivery Ratio) calculation from HELLO messages
 * - RSSI processing for signal strength analysis
 * - Simplified Random Forest prediction for link scoring
 * - Neighbor ranking for cluster head election
 */

#ifndef ARPMEC_LQE_H
#define ARPMEC_LQE_H

#include "ns3/object.h"
#include "ns3/ipv4-address.h"
#include "ns3/simulator.h"
#include "ns3/timer.h"
#include <map>
#include <vector>
#include <deque>

namespace ns3
{
namespace arpmec
{

/**
 * \ingroup arpmec
 * \brief Link Quality Estimation module for ARPMEC protocol
 *
 * This class implements the LQE functionality described in Algorithm 2
 * of the ARPMEC paper. It processes HELLO messages containing RSSI,
 * PDR, and timestamp information to evaluate link quality between nodes.
 */
class ArpmecLqe : public Object
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
    ArpmecLqe();

    /**
     * \brief Destructor
     */
    virtual ~ArpmecLqe();

    /**
     * \brief Update link quality information for a neighbor
     * \param neighborId The ID of the neighbor node
     * \param rssi The RSSI value from the received HELLO message
     * \param pdr The PDR value from the received HELLO message
     * \param timestamp The timestamp from the received HELLO message
     * \param receivedTime The time when the message was received
     */
    void UpdateLinkQuality(uint32_t neighborId, double rssi, double pdr, uint64_t timestamp, Time receivedTime);

    /**
     * \brief Calculate PDR for a specific neighbor
     * \param neighborId The ID of the neighbor node
     * \return The calculated PDR value (0.0 to 1.0)
     */
    double CalculatePdr(uint32_t neighborId);

    /**
     * \brief Get the current RSSI for a neighbor
     * \param neighborId The ID of the neighbor node
     * \return The most recent RSSI value
     */
    double GetRssi(uint32_t neighborId);

    /**
     * \brief Predict link score using simplified Random Forest algorithm
     * \param neighborId The ID of the neighbor node
     * \return The predicted link quality score (0.0 to 1.0)
     */
    double PredictLinkScore(uint32_t neighborId);

    /**
     * \brief Get the best neighbor based on link quality score
     * \return The neighbor ID with the highest link quality score
     */
    uint32_t GetBestNeighbor();

    /**
     * \brief Get all neighbors sorted by link quality score (best first)
     * \return Vector of neighbor IDs sorted by quality score
     */
    std::vector<uint32_t> GetNeighborsByQuality();

    /**
     * \brief Check if a neighbor is still active (recently heard from)
     * \param neighborId The ID of the neighbor node
     * \return True if the neighbor is considered active
     */
    bool IsNeighborActive(uint32_t neighborId);

    /**
     * \brief Remove inactive neighbors from the table
     */
    void CleanupInactiveNeighbors();

    /**
     * \brief Set the HELLO interval for PDR calculation
     * \param interval The expected interval between HELLO messages
     */
    void SetHelloInterval(Time interval);

    /**
     * \brief Set the neighbor timeout duration
     * \param timeout The time after which a neighbor is considered inactive
     */
    void SetNeighborTimeout(Time timeout);

private:
    /**
     * \brief Structure to store link information for each neighbor
     */
    struct LinkInfo
    {
        double currentRssi;                    ///< Current RSSI value
        double currentPdr;                     ///< Current PDR value
        double linkScore;                      ///< Calculated link quality score
        uint32_t helloCount;                   ///< Number of HELLO messages received
        uint32_t expectedHellos;               ///< Number of expected HELLO messages
        Time lastHeardFrom;                    ///< Last time we heard from this neighbor
        std::deque<double> rssiHistory;        ///< History of RSSI values for averaging
        std::deque<Time> helloTimes;           ///< History of HELLO message arrival times
        uint64_t lastTimestamp;                ///< Last timestamp from neighbor

        LinkInfo() : currentRssi(-100.0), currentPdr(0.0), linkScore(0.0),
                    helloCount(0), expectedHellos(0), lastTimestamp(0) {}
    };

    /**
     * \brief Calculate the average RSSI from history
     * \param info The link information structure
     * \return The average RSSI value
     */
    double CalculateAverageRssi(const LinkInfo& info);

    /**
     * \brief Update PDR calculation for a neighbor
     * \param info The link information structure to update
     */
    void UpdatePdrCalculation(LinkInfo& info);

    /**
     * \brief Simplified Random Forest prediction
     * \param rssi The RSSI value
     * \param pdr The PDR value
     * \return The predicted link quality score
     */
    double SimplifiedRandomForestPredict(double rssi, double pdr);

    /**
     * \brief Decision tree 1 for Random Forest
     * \param rssi The RSSI value
     * \param pdr The PDR value
     * \return Tree 1 prediction
     */
    double DecisionTree1(double rssi, double pdr);

    /**
     * \brief Decision tree 2 for Random Forest
     * \param rssi The RSSI value
     * \param pdr The PDR value
     * \return Tree 2 prediction
     */
    double DecisionTree2(double rssi, double pdr);

    /**
     * \brief Decision tree 3 for Random Forest
     * \param rssi The RSSI value
     * \param pdr The PDR value
     * \return Tree 3 prediction
     */
    double DecisionTree3(double rssi, double pdr);

    std::map<uint32_t, LinkInfo> m_neighbors;  ///< Map of neighbor information
    Time m_helloInterval;                      ///< Expected HELLO interval
    Time m_neighborTimeout;                    ///< Neighbor timeout duration
    uint32_t m_maxRssiHistory;                 ///< Maximum RSSI history size
    Timer m_cleanupTimer;                      ///< Timer for cleaning up inactive neighbors

    // ARPMEC parameters from the paper
    static constexpr double RSSI_THRESHOLD_GOOD = -70.0;    ///< Good RSSI threshold (dBm)
    static constexpr double RSSI_THRESHOLD_POOR = -90.0;    ///< Poor RSSI threshold (dBm)
    static constexpr double PDR_THRESHOLD_GOOD = 0.8;       ///< Good PDR threshold
    static constexpr double PDR_THRESHOLD_POOR = 0.5;       ///< Poor PDR threshold
};

} // namespace arpmec
} // namespace ns3

#endif /* ARPMEC_LQE_H */
