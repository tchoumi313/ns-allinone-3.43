/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * ARPMEC MEC Edge Server
 * 
 * This class implements the MEC Edge Server functionality for distributed
 * computing and edge processing as described in the ARPMEC paper.
 */

#ifndef ARPMEC_MEC_SERVER_H
#define ARPMEC_MEC_SERVER_H

#include "ns3/object.h"
#include "ns3/ipv4-address.h"
#include "ns3/simulator.h"
#include "ns3/timer.h"
#include "ns3/callback.h"
#include <map>
#include <queue>
#include <vector>

namespace ns3
{
namespace arpmec
{

/**
 * \ingroup arpmec
 * \brief MEC Edge Server for distributed computing
 *
 * This class implements edge computing services as part of the MEC infrastructure
 * described in the ARPMEC paper. It handles computational tasks from clusters
 * and coordinates with cloud resources when needed.
 */
class ArpmecMecServer : public Object
{
public:
    /// Server operational states
    enum ServerState
    {
        OFFLINE = 0,         ///< Server is offline
        IDLE = 1,            ///< Server is idle and ready
        PROCESSING = 2,      ///< Server is processing tasks
        OVERLOADED = 3       ///< Server is overloaded
    };

    /// Task priority levels
    enum TaskPriority
    {
        LOW_PRIORITY = 0,    ///< Low priority task
        NORMAL_PRIORITY = 1, ///< Normal priority task
        HIGH_PRIORITY = 2,   ///< High priority task
        CRITICAL_PRIORITY = 3 ///< Critical priority task
    };

    /**
     * \brief Structure representing a computational task
     */
    struct ComputationTask
    {
        uint32_t taskId;         ///< Unique task identifier
        uint32_t sourceCluster;  ///< Source cluster ID
        uint32_t requestSize;    ///< Computational complexity
        double deadline;         ///< Task deadline in seconds
        TaskPriority priority;   ///< Task priority level
        Time arrivalTime;        ///< When task arrived
        Time startTime;          ///< When processing started
        
        ComputationTask() : taskId(0), sourceCluster(0), requestSize(0), 
                           deadline(0.0), priority(NORMAL_PRIORITY),
                           arrivalTime(Seconds(0)), startTime(Seconds(0)) {}
    };

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    /**
     * \brief Constructor
     */
    ArpmecMecServer();

    /**
     * \brief Destructor
     */
    virtual ~ArpmecMecServer();

    /**
     * \brief Initialize the MEC Edge Server
     * \param serverId The unique server identifier
     * \param processingCapacity Maximum processing capacity (ops/sec)
     * \param memoryCapacity Available memory capacity (MB)
     */
    void Initialize(uint32_t serverId, uint32_t processingCapacity, uint32_t memoryCapacity);

    /**
     * \brief Start the server
     */
    void Start();

    /**
     * \brief Stop the server
     */
    void Stop();

    /**
     * \brief Submit a computational task
     * \param task The task to be processed
     * \return True if task is accepted, false if rejected
     */
    bool SubmitTask(const ComputationTask& task);

    /**
     * \brief Get current server state
     * \return The current operational state
     */
    ServerState GetServerState() const;

    /**
     * \brief Get current processing load (0.0 to 1.0)
     * \return Current load ratio
     */
    double GetProcessingLoad() const;

    /**
     * \brief Get number of queued tasks
     * \return Number of tasks waiting for processing
     */
    uint32_t GetQueuedTaskCount() const;

    /**
     * \brief Get server statistics
     * \param tasksProcessed Number of tasks completed
     * \param averageResponseTime Average task response time
     * \param rejectionRate Task rejection rate
     */
    void GetStatistics(uint32_t& tasksProcessed, double& averageResponseTime, double& rejectionRate) const;

    /**
     * \brief Set callback for task completion
     * \param callback The callback function
     */
    void SetTaskCompletionCallback(Callback<void, uint32_t, uint32_t, double> callback);

    /**
     * \brief Set callback for cloud offloading
     * \param callback The callback function for offloading tasks to cloud
     */
    void SetCloudOffloadCallback(Callback<bool, ComputationTask> callback);

private:
    /**
     * \brief Process the next task in queue
     */
    void ProcessNextTask();

    /**
     * \brief Complete current task
     */
    void CompleteCurrentTask();

    /**
     * \brief Check for task deadlines and handle timeouts
     */
    void CheckTaskDeadlines();

    /**
     * \brief Update server statistics
     */
    void UpdateStatistics();

    /**
     * \brief Determine if task should be offloaded to cloud
     * \param task The task to evaluate
     * \return True if should be offloaded
     */
    bool ShouldOffloadToCloud(const ComputationTask& task) const;

    /**
     * \brief Update server state based on current load and processing status
     */
    void UpdateServerState();

    // Member variables
    uint32_t m_serverId;                    ///< Server identifier
    uint32_t m_processingCapacity;          ///< Maximum processing capacity
    uint32_t m_memoryCapacity;              ///< Available memory capacity
    uint32_t m_currentLoad;                 ///< Current processing load
    ServerState m_state;                    ///< Current server state
    
    // Task management
    std::priority_queue<ComputationTask> m_taskQueue;  ///< Queue of pending tasks
    ComputationTask m_currentTask;          ///< Currently processing task
    bool m_processingActive;                ///< Whether currently processing a task
    uint32_t m_nextTaskId;                  ///< Next task ID to assign
    
    // Timers
    Timer m_processingTimer;                ///< Timer for task processing
    Timer m_deadlineCheckTimer;             ///< Timer for deadline checking
    Timer m_statisticsTimer;                ///< Timer for statistics updates
    
    // Statistics
    uint32_t m_tasksProcessed;              ///< Total tasks completed
    uint32_t m_tasksRejected;               ///< Total tasks rejected
    double m_totalResponseTime;             ///< Cumulative response time
    std::vector<double> m_responseTimes;    ///< Response time history
    
    // Configuration
    bool m_isRunning;                       ///< Whether server is active
    double m_overloadThreshold;             ///< Load threshold for overload state
    double m_offloadThreshold;              ///< Load threshold for cloud offloading
    Time m_deadlineCheckInterval;           ///< Interval for deadline checking
    
    // Callbacks
    Callback<void, uint32_t, uint32_t, double> m_taskCompletionCallback;  ///< Task completion callback
    Callback<bool, ComputationTask> m_cloudOffloadCallback;               ///< Cloud offload callback
    
    // Static configuration
    static constexpr double DEFAULT_OVERLOAD_THRESHOLD = 0.8;    ///< Default overload threshold
    static constexpr double DEFAULT_OFFLOAD_THRESHOLD = 0.6;     ///< Default offload threshold
    static const Time DEFAULT_DEADLINE_CHECK_INTERVAL;           ///< Default deadline check interval
    static const Time DEFAULT_STATISTICS_INTERVAL;               ///< Default statistics update interval
};

// Comparison operator for priority queue (higher priority = lower value)
bool operator<(const ArpmecMecServer::ComputationTask& lhs, const ArpmecMecServer::ComputationTask& rhs);

} // namespace arpmec
} // namespace ns3

#endif /* ARPMEC_MEC_SERVER_H */
