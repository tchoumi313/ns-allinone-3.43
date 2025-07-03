/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * ARPMEC Link Quality Estimation (LQE) Module Implementation
 */

#include "arpmec-lqe.h"
#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include <algorithm>
#include <cmath>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ArpmecLqe");

namespace arpmec
{

NS_OBJECT_ENSURE_REGISTERED(ArpmecLqe);

TypeId
ArpmecLqe::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::ArpmecLqe")
                            .SetParent<Object>()
                            .SetGroupName("Arpmec")
                            .AddConstructor<ArpmecLqe>()
                            .AddAttribute("HelloInterval",
                                        "Expected interval between HELLO messages",
                                        TimeValue(Seconds(1.0)),
                                        MakeTimeAccessor(&ArpmecLqe::m_helloInterval),
                                        MakeTimeChecker())
                            .AddAttribute("NeighborTimeout",
                                        "Time after which a neighbor is considered inactive",
                                        TimeValue(Seconds(3.0)),
                                        MakeTimeAccessor(&ArpmecLqe::m_neighborTimeout),
                                        MakeTimeChecker())
                            .AddAttribute("MaxRssiHistory",
                                        "Maximum number of RSSI values to keep in history",
                                        UintegerValue(10),
                                        MakeUintegerAccessor(&ArpmecLqe::m_maxRssiHistory),
                                        MakeUintegerChecker<uint32_t>());
    return tid;
}

ArpmecLqe::ArpmecLqe()
    : m_helloInterval(Seconds(1.0)),
      m_neighborTimeout(Seconds(3.0)),
      m_maxRssiHistory(10)
{
    NS_LOG_FUNCTION(this);

    // Set up the cleanup timer function (but don't schedule yet)
    m_cleanupTimer.SetFunction(&ArpmecLqe::CleanupInactiveNeighbors, this);
}

ArpmecLqe::~ArpmecLqe()
{
    NS_LOG_FUNCTION(this);
    m_cleanupTimer.Cancel();
}

void
ArpmecLqe::UpdateLinkQuality(uint32_t neighborId, double rssi, double pdr, uint64_t timestamp, Time receivedTime)
{
    NS_LOG_FUNCTION(this << neighborId << rssi << pdr << timestamp);

    LinkInfo& info = m_neighbors[neighborId];

    // Update basic information
    info.currentRssi = rssi;
    info.currentPdr = pdr;
    info.lastHeardFrom = receivedTime;
    info.helloCount++;

    // Update RSSI history
    info.rssiHistory.push_back(rssi);
    if (info.rssiHistory.size() > m_maxRssiHistory)
    {
        info.rssiHistory.pop_front();
    }

    // Update HELLO timing history for PDR calculation
    info.helloTimes.push_back(receivedTime);
    if (info.helloTimes.size() > m_maxRssiHistory)
    {
        info.helloTimes.pop_front();
    }

    // Check for missed HELLOs based on timestamp sequence
    if (info.lastTimestamp != 0 && timestamp > info.lastTimestamp)
    {
        uint64_t timeDiff = timestamp - info.lastTimestamp;
        uint64_t expectedHellos = timeDiff / m_helloInterval.GetMicroSeconds();
        info.expectedHellos += expectedHellos;
    }
    else if (info.lastTimestamp == 0)
    {
        // First HELLO - initialize expected count
        info.expectedHellos = 1;
    }
    info.lastTimestamp = timestamp;

    // Update PDR calculation
    UpdatePdrCalculation(info);

    // Calculate new link score using simplified Random Forest
    double avgRssi = CalculateAverageRssi(info);
    info.linkScore = SimplifiedRandomForestPredict(avgRssi, info.currentPdr);
}

double
ArpmecLqe::CalculatePdr(uint32_t neighborId)
{
    auto it = m_neighbors.find(neighborId);
    if (it == m_neighbors.end())
    {
        return 0.0;
    }

    const LinkInfo& info = it->second;
    if (info.expectedHellos == 0)
    {
        return 1.0; // No missed messages yet
    }

    return static_cast<double>(info.helloCount) / static_cast<double>(info.expectedHellos);
}

double
ArpmecLqe::GetRssi(uint32_t neighborId)
{
    auto it = m_neighbors.find(neighborId);
    if (it == m_neighbors.end())
    {
        return -100.0; // Very poor RSSI for unknown neighbors
    }

    return it->second.currentRssi;
}

double
ArpmecLqe::PredictLinkScore(uint32_t neighborId)
{
    auto it = m_neighbors.find(neighborId);
    if (it == m_neighbors.end())
    {
        return 0.0;
    }

    return it->second.linkScore;
}

uint32_t
ArpmecLqe::GetBestNeighbor()
{
    uint32_t bestNeighbor = 0;
    double bestScore = -1.0;

    for (const auto& neighbor : m_neighbors)
    {
        if (IsNeighborActive(neighbor.first) && neighbor.second.linkScore > bestScore)
        {
            bestScore = neighbor.second.linkScore;
            bestNeighbor = neighbor.first;
        }
    }

    NS_LOG_DEBUG("Best neighbor: " << bestNeighbor << " with score: " << bestScore);
    return bestNeighbor;
}

std::vector<uint32_t>
ArpmecLqe::GetNeighborsByQuality()
{
    NS_LOG_FUNCTION(this);

    std::vector<std::pair<uint32_t, double>> neighborScores;

    // Collect active neighbors and their scores
    for (const auto& neighbor : m_neighbors)
    {
        if (IsNeighborActive(neighbor.first))
        {
            neighborScores.emplace_back(neighbor.first, neighbor.second.linkScore);
        }
    }

    // Sort by score (highest first)
    std::sort(neighborScores.begin(), neighborScores.end(),
              [](const std::pair<uint32_t, double>& a, const std::pair<uint32_t, double>& b) {
                  return a.second > b.second;
              });

    // Extract neighbor IDs
    std::vector<uint32_t> result;
    for (const auto& pair : neighborScores)
    {
        result.push_back(pair.first);
    }

    return result;
}

bool
ArpmecLqe::IsNeighborActive(uint32_t neighborId)
{
    auto it = m_neighbors.find(neighborId);
    if (it == m_neighbors.end())
    {
        return false;
    }

    Time now = Simulator::Now();
    return (now - it->second.lastHeardFrom) <= m_neighborTimeout;
}

void
ArpmecLqe::CleanupInactiveNeighbors()
{
    NS_LOG_FUNCTION(this);

    Time now = Simulator::Now();
    auto it = m_neighbors.begin();

    while (it != m_neighbors.end())
    {
        if ((now - it->second.lastHeardFrom) > m_neighborTimeout)
        {
            NS_LOG_DEBUG("Removing inactive neighbor: " << it->first);
            it = m_neighbors.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Schedule next cleanup only if timer is not already running
    if (!m_cleanupTimer.IsRunning())
    {
        m_cleanupTimer.Schedule(m_neighborTimeout);
    }
}

void
ArpmecLqe::SetHelloInterval(Time interval)
{
    m_helloInterval = interval;
}

void
ArpmecLqe::SetNeighborTimeout(Time timeout)
{
    m_neighborTimeout = timeout;
}

// Private helper methods

double
ArpmecLqe::CalculateAverageRssi(const LinkInfo& info)
{
    if (info.rssiHistory.empty())
    {
        return info.currentRssi;
    }

    double sum = 0.0;
    for (double rssi : info.rssiHistory)
    {
        sum += rssi;
    }

    return sum / info.rssiHistory.size();
}

void
ArpmecLqe::UpdatePdrCalculation(LinkInfo& info)
{
    // Simple PDR calculation based on received vs expected HELLOs
    if (info.expectedHellos > 0)
    {
        info.currentPdr = std::min(1.0, static_cast<double>(info.helloCount) /
                                       static_cast<double>(info.expectedHellos));
    }
    else
    {
        info.currentPdr = 1.0; // No expected messages yet
    }
}

double
ArpmecLqe::SimplifiedRandomForestPredict(double rssi, double pdr)
{
    // Simplified Random Forest with 3 decision trees
    double tree1 = DecisionTree1(rssi, pdr);
    double tree2 = DecisionTree2(rssi, pdr);
    double tree3 = DecisionTree3(rssi, pdr);

    // Average the predictions (ensemble method)
    double prediction = (tree1 + tree2 + tree3) / 3.0;

    // Ensure result is between 0 and 1
    return std::max(0.0, std::min(1.0, prediction));
}

double
ArpmecLqe::DecisionTree1(double rssi, double pdr)
{
    // Tree 1: Focus on RSSI thresholds
    if (rssi >= RSSI_THRESHOLD_GOOD)
    {
        if (pdr >= PDR_THRESHOLD_GOOD)
        {
            return 0.9; // Excellent link
        }
        else
        {
            return 0.7; // Good RSSI, moderate PDR
        }
    }
    else if (rssi >= RSSI_THRESHOLD_POOR)
    {
        if (pdr >= PDR_THRESHOLD_GOOD)
        {
            return 0.6; // Moderate RSSI, good PDR
        }
        else if (pdr >= PDR_THRESHOLD_POOR)
        {
            return 0.4; // Moderate link
        }
        else
        {
            return 0.2; // Poor PDR
        }
    }
    else
    {
        // Very poor RSSI
        if (pdr >= PDR_THRESHOLD_GOOD)
        {
            return 0.3; // Poor RSSI but good PDR
        }
        else
        {
            return 0.1; // Very poor link
        }
    }
}

double
ArpmecLqe::DecisionTree2(double rssi, double pdr)
{
    // Tree 2: Focus on PDR thresholds
    if (pdr >= PDR_THRESHOLD_GOOD)
    {
        if (rssi >= RSSI_THRESHOLD_GOOD)
        {
            return 0.95; // Excellent both
        }
        else if (rssi >= RSSI_THRESHOLD_POOR)
        {
            return 0.75; // Good PDR, moderate RSSI
        }
        else
        {
            return 0.4; // Good PDR, poor RSSI
        }
    }
    else if (pdr >= PDR_THRESHOLD_POOR)
    {
        if (rssi >= RSSI_THRESHOLD_GOOD)
        {
            return 0.6; // Good RSSI, moderate PDR
        }
        else
        {
            return 0.3; // Moderate link
        }
    }
    else
    {
        // Poor PDR
        if (rssi >= RSSI_THRESHOLD_GOOD)
        {
            return 0.25; // Good RSSI but poor PDR
        }
        else
        {
            return 0.05; // Poor both
        }
    }
}

double
ArpmecLqe::DecisionTree3(double rssi, double pdr)
{
    // Tree 3: Combined weighted approach
    double rssiScore = 0.0;
    double pdrScore = 0.0;

    // Normalize RSSI score (assuming range -100 to -30 dBm)
    rssiScore = std::max(0.0, std::min(1.0, (rssi + 100.0) / 70.0));

    // PDR score is already normalized (0.0 to 1.0)
    pdrScore = pdr;

    // Weighted combination: 60% PDR, 40% RSSI
    return 0.6 * pdrScore + 0.4 * rssiScore;
}

double
ArpmecLqe::GetLinkScore(uint32_t neighborId)
{
    return PredictLinkScore(neighborId);
}

std::vector<uint32_t>
ArpmecLqe::GetNeighbors()
{
    NS_LOG_FUNCTION(this);

    std::vector<uint32_t> result;
    for (const auto& neighbor : m_neighbors)
    {
        result.push_back(neighbor.first);
    }

    return result;
}

} // namespace arpmec
} // namespace ns3
