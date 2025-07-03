/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * ARPMEC MEC Edge Server Implementation
 */

#include "arpmec-mec-server.h"
#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ArpmecMecServer");

namespace arpmec
{

// Static Time constants
const Time ArpmecMecServer::DEFAULT_DEADLINE_CHECK_INTERVAL = Seconds(1.0);
const Time ArpmecMecServer::DEFAULT_STATISTICS_INTERVAL = Seconds(5.0);

// Comparison operator for priority queue (higher priority tasks processed first)
bool operator<(const ArpmecMecServer::ComputationTask& lhs, const ArpmecMecServer::ComputationTask& rhs)
{
    // Lower priority value = higher priority (reverse order for max-heap behavior)
    if (lhs.priority != rhs.priority)
    {
        return lhs.priority > rhs.priority;
    }
    // For same priority, earlier deadline has higher priority
    return lhs.deadline > rhs.deadline;
}

NS_OBJECT_ENSURE_REGISTERED(ArpmecMecServer);

TypeId
ArpmecMecServer::GetTypeId()
{
    static TypeId tid = TypeId("ns3::arpmec::ArpmecMecServer")
                            .SetParent<Object>()
                            .SetGroupName("Arpmec")
                            .AddConstructor<ArpmecMecServer>()
                            .AddAttribute("ProcessingCapacity",
                                        "Maximum processing capacity of the server (ops/sec)",
                                        UintegerValue(1000),
                                        MakeUintegerAccessor(&ArpmecMecServer::m_processingCapacity),
                                        MakeUintegerChecker<uint32_t>())
                            .AddAttribute("MemoryCapacity",
                                        "Available memory capacity (MB)",
                                        UintegerValue(1024),
                                        MakeUintegerAccessor(&ArpmecMecServer::m_memoryCapacity),
                                        MakeUintegerChecker<uint32_t>())
                            .AddAttribute("OverloadThreshold",
                                        "Load threshold for overload state",
                                        DoubleValue(DEFAULT_OVERLOAD_THRESHOLD),
                                        MakeDoubleAccessor(&ArpmecMecServer::m_overloadThreshold),
                                        MakeDoubleChecker<double>(0.0, 1.0))
                            .AddAttribute("OffloadThreshold",
                                        "Load threshold for cloud offloading",
                                        DoubleValue(DEFAULT_OFFLOAD_THRESHOLD),
                                        MakeDoubleAccessor(&ArpmecMecServer::m_offloadThreshold),
                                        MakeDoubleChecker<double>(0.0, 1.0));
    return tid;
}

ArpmecMecServer::ArpmecMecServer()
    : m_serverId(0),
      m_processingCapacity(1000),
      m_memoryCapacity(1024),
      m_currentLoad(0),
      m_state(OFFLINE),
      m_processingActive(false),
      m_nextTaskId(1),
      m_tasksProcessed(0),
      m_tasksRejected(0),
      m_totalResponseTime(0.0),
      m_isRunning(false),
      m_overloadThreshold(DEFAULT_OVERLOAD_THRESHOLD),
      m_offloadThreshold(DEFAULT_OFFLOAD_THRESHOLD),
      m_deadlineCheckInterval(DEFAULT_DEADLINE_CHECK_INTERVAL)
{
    NS_LOG_FUNCTION(this);

    // Set up timers
    m_processingTimer.SetFunction(&ArpmecMecServer::CompleteCurrentTask, this);
    m_deadlineCheckTimer.SetFunction(&ArpmecMecServer::CheckTaskDeadlines, this);
    m_statisticsTimer.SetFunction(&ArpmecMecServer::UpdateStatistics, this);
}

ArpmecMecServer::~ArpmecMecServer()
{
    NS_LOG_FUNCTION(this);
    Stop();
}

void
ArpmecMecServer::Initialize(uint32_t serverId, uint32_t processingCapacity, uint32_t memoryCapacity)
{
    NS_LOG_FUNCTION(this << serverId << processingCapacity << memoryCapacity);

    m_serverId = serverId;
    m_processingCapacity = processingCapacity;
    m_memoryCapacity = memoryCapacity;

    NS_LOG_INFO("MEC Server " << m_serverId << " initialized with capacity " 
                << m_processingCapacity << " ops/sec and " << m_memoryCapacity << " MB memory");
}

void
ArpmecMecServer::Start()
{
    NS_LOG_FUNCTION(this);

    if (m_isRunning)
    {
        return;
    }

    m_isRunning = true;
    m_state = IDLE;
    m_currentLoad = 0;
    m_processingActive = false;

    // Start periodic deadline checking
    if (!m_deadlineCheckTimer.IsRunning())
    {
        m_deadlineCheckTimer.Schedule(m_deadlineCheckInterval);
    }

    // Start statistics updates
    if (!m_statisticsTimer.IsRunning())
    {
        m_statisticsTimer.Schedule(DEFAULT_STATISTICS_INTERVAL);
    }

    NS_LOG_INFO("MEC Server " << m_serverId << " started");
}

void
ArpmecMecServer::Stop()
{
    NS_LOG_FUNCTION(this);

    if (!m_isRunning)
    {
        return;
    }

    m_isRunning = false;
    m_state = OFFLINE;
    m_processingTimer.Cancel();
    m_deadlineCheckTimer.Cancel();
    m_statisticsTimer.Cancel();

    // Clear task queue
    while (!m_taskQueue.empty())
    {
        m_taskQueue.pop();
    }
    
    m_currentLoad = 0;
    m_processingActive = false;

    NS_LOG_INFO("MEC Server " << m_serverId << " stopped");
}

bool
ArpmecMecServer::SubmitTask(const ComputationTask& task)
{
    NS_LOG_FUNCTION(this << task.taskId << task.sourceCluster << task.requestSize);

    if (!m_isRunning || m_state == OFFLINE)
    {
        NS_LOG_WARN("Server " << m_serverId << " is offline, rejecting task " << task.taskId);
        m_tasksRejected++;
        return false;
    }

    // Check if we should offload to cloud
    if (ShouldOffloadToCloud(task))
    {
        if (!m_cloudOffloadCallback.IsNull())
        {
            bool cloudAccepted = m_cloudOffloadCallback(task);
            if (cloudAccepted)
            {
                NS_LOG_INFO("Server " << m_serverId << " offloaded task " << task.taskId << " to cloud");
                return true;
            }
        }
        
        // Cloud didn't accept, try to process locally anyway
        NS_LOG_WARN("Cloud offload failed for task " << task.taskId << ", processing locally");
    }

    // Check capacity
    if (m_currentLoad + task.requestSize > m_processingCapacity)
    {
        NS_LOG_WARN("Server " << m_serverId << " at capacity, rejecting task " << task.taskId);
        m_tasksRejected++;
        return false;
    }

    // Accept the task
    ComputationTask newTask = task;
    newTask.taskId = m_nextTaskId++;
    newTask.arrivalTime = Simulator::Now();
    
    m_taskQueue.push(newTask);
    m_currentLoad += task.requestSize;

    NS_LOG_INFO("Server " << m_serverId << " accepted task " << newTask.taskId 
                << " from cluster " << task.sourceCluster);

    // Update server state
    UpdateServerState();

    // Start processing if not already busy
    if (!m_processingActive)
    {
        ProcessNextTask();
    }

    return true;
}

ArpmecMecServer::ServerState
ArpmecMecServer::GetServerState() const
{
    return m_state;
}

double
ArpmecMecServer::GetProcessingLoad() const
{
    return static_cast<double>(m_currentLoad) / m_processingCapacity;
}

uint32_t
ArpmecMecServer::GetQueuedTaskCount() const
{
    return m_taskQueue.size();
}

void
ArpmecMecServer::GetStatistics(uint32_t& tasksProcessed, double& averageResponseTime, double& rejectionRate) const
{
    tasksProcessed = m_tasksProcessed;
    
    if (m_tasksProcessed > 0)
    {
        averageResponseTime = m_totalResponseTime / m_tasksProcessed;
    }
    else
    {
        averageResponseTime = 0.0;
    }

    uint32_t totalTasks = m_tasksProcessed + m_tasksRejected;
    if (totalTasks > 0)
    {
        rejectionRate = static_cast<double>(m_tasksRejected) / totalTasks;
    }
    else
    {
        rejectionRate = 0.0;
    }
}

void
ArpmecMecServer::SetTaskCompletionCallback(Callback<void, uint32_t, uint32_t, double> callback)
{
    m_taskCompletionCallback = callback;
}

void
ArpmecMecServer::SetCloudOffloadCallback(Callback<bool, ComputationTask> callback)
{
    m_cloudOffloadCallback = callback;
}

void
ArpmecMecServer::ProcessNextTask()
{
    NS_LOG_FUNCTION(this);

    if (!m_isRunning || m_taskQueue.empty() || m_processingActive)
    {
        return;
    }

    // Get the highest priority task
    m_currentTask = m_taskQueue.top();
    m_taskQueue.pop();
    
    m_currentTask.startTime = Simulator::Now();
    m_processingActive = true;

    // Calculate processing time (simplified model)
    double processingTime = static_cast<double>(m_currentTask.requestSize) / m_processingCapacity;
    
    // Add some randomness to processing time (±10%)
    double variation = 0.1 * processingTime * (2.0 * drand48() - 1.0);
    processingTime += variation;
    
    NS_LOG_INFO("Server " << m_serverId << " started processing task " << m_currentTask.taskId 
                << " (estimated time: " << processingTime << "s)");

    // Schedule task completion
    m_processingTimer.Schedule(Seconds(processingTime));
    
    // Update server state
    UpdateServerState();
}

void
ArpmecMecServer::CompleteCurrentTask()
{
    NS_LOG_FUNCTION(this);

    if (!m_processingActive)
    {
        return;
    }

    Time responseTime = Simulator::Now() - m_currentTask.arrivalTime;
    double responseTimeSeconds = responseTime.GetSeconds();

    m_tasksProcessed++;
    m_totalResponseTime += responseTimeSeconds;
    m_responseTimes.push_back(responseTimeSeconds);
    
    // Reduce current load
    m_currentLoad -= m_currentTask.requestSize;
    m_processingActive = false;

    NS_LOG_INFO("Server " << m_serverId << " completed task " << m_currentTask.taskId 
                << " in " << responseTimeSeconds << "s");

    // Fire task completion callback
    if (!m_taskCompletionCallback.IsNull())
    {
        m_taskCompletionCallback(m_currentTask.taskId, m_currentTask.sourceCluster, responseTimeSeconds);
    }

    // Update server state
    UpdateServerState();

    // Process next task if available
    if (!m_taskQueue.empty())
    {
        ProcessNextTask();
    }
}

void
ArpmecMecServer::CheckTaskDeadlines()
{
    NS_LOG_FUNCTION(this);

    if (!m_isRunning)
    {
        return;
    }

    Time now = Simulator::Now();
    
    // Check current task deadline
    if (m_processingActive)
    {
        double timeElapsed = (now - m_currentTask.startTime).GetSeconds();
        if (timeElapsed > m_currentTask.deadline)
        {
            NS_LOG_WARN("Server " << m_serverId << " task " << m_currentTask.taskId 
                        << " missed deadline (" << timeElapsed << "s > " << m_currentTask.deadline << "s)");
        }
    }

    // Schedule next deadline check
    if (m_isRunning)
    {
        m_deadlineCheckTimer.Schedule(m_deadlineCheckInterval);
    }
}

void
ArpmecMecServer::UpdateStatistics()
{
    NS_LOG_FUNCTION(this);

    if (!m_isRunning)
    {
        return;
    }

    UpdateServerState();

    // Schedule next statistics update
    if (m_isRunning)
    {
        m_statisticsTimer.Schedule(DEFAULT_STATISTICS_INTERVAL);
    }
}

bool
ArpmecMecServer::ShouldOffloadToCloud(const ComputationTask& task) const
{
    // Offload if current load exceeds threshold
    double currentLoadRatio = GetProcessingLoad();
    
    if (currentLoadRatio > m_offloadThreshold)
    {
        return true;
    }
    
    // Offload high-complexity tasks if approaching capacity
    if (currentLoadRatio > 0.5 && task.requestSize > (m_processingCapacity / 4))
    {
        return true;
    }
    
    return false;
}

void
ArpmecMecServer::UpdateServerState()
{
    double loadRatio = GetProcessingLoad();
    
    if (loadRatio > m_overloadThreshold)
    {
        m_state = OVERLOADED;
    }
    else if (m_processingActive || !m_taskQueue.empty())
    {
        m_state = PROCESSING;
    }
    else
    {
        m_state = IDLE;
    }
}

} // namespace arpmec
} // namespace ns3
