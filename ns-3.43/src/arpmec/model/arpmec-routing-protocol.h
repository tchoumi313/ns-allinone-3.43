/*
 * Copyright (c) 2009 IITP RAS
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Based on
 *      NS-2 ARPMEC model developed by the CMU/MONARCH group and optimized and
 *      tuned by Samir Das and Mahesh Marina, University of Cincinnati;
 *
 *      ARPMEC-UU implementation by Erik Nordström of Uppsala University
 *      https://web.archive.org/web/20100527072022/http://core.it.uu.se/core/index.php/ARPMEC-UU
 *
 * Authors: Elena Buchatskaia <borovkovaes@iitp.ru>
 *          Pavel Boyko <boyko@iitp.ru>
 */
#ifndef ARPMECROUTINGPROTOCOL_H
#define ARPMECROUTINGPROTOCOL_H

#include "arpmec-dpd.h"
#include "arpmec-neighbor.h"
#include "arpmec-packet.h"
#include "arpmec-rqueue.h"
#include "arpmec-rtable.h"
#include "arpmec-lqe.h"
#include "arpmec-clustering.h"
#include "arpmec-adaptive-routing.h"
#include "arpmec-mec-gateway.h"
#include "arpmec-mec-server.h"

#include "ns3/ipv4-interface.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/node.h"
#include "ns3/output-stream-wrapper.h"
#include "ns3/random-variable-stream.h"
#include "ns3/traced-callback.h"

#include <map>

namespace ns3
{

class WifiMpdu;
enum WifiMacDropReason : uint8_t; // opaque enum declaration

namespace arpmec
{

/**
 * \brief ARPMEC TracedCallback typedefs
 */
typedef TracedCallback<uint32_t, bool> ClusterHeadTracedCallback;
typedef TracedCallback<uint32_t, std::string> RouteDecisionTracedCallback;
typedef TracedCallback<uint32_t, double> LqeUpdateTracedCallback;
typedef TracedCallback<uint32_t, double> EnergyUpdateTracedCallback;

/**
 * \ingroup arpmec
 *
 * \brief ARPMEC routing protocol
 */
class RoutingProtocol : public Ipv4RoutingProtocol
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();
    static const uint32_t ARPMEC_PORT;

    /// constructor
    RoutingProtocol();
    ~RoutingProtocol() override;
    void DoDispose() override;

    // Inherited from Ipv4RoutingProtocol
    Ptr<Ipv4Route> RouteOutput(Ptr<Packet> p,
                               const Ipv4Header& header,
                               Ptr<NetDevice> oif,
                               Socket::SocketErrno& sockerr) override;
    bool RouteInput(Ptr<const Packet> p,
                    const Ipv4Header& header,
                    Ptr<const NetDevice> idev,
                    const UnicastForwardCallback& ucb,
                    const MulticastForwardCallback& mcb,
                    const LocalDeliverCallback& lcb,
                    const ErrorCallback& ecb) override;
    void NotifyInterfaceUp(uint32_t interface) override;
    void NotifyInterfaceDown(uint32_t interface) override;
    void NotifyAddAddress(uint32_t interface, Ipv4InterfaceAddress address) override;
    void NotifyRemoveAddress(uint32_t interface, Ipv4InterfaceAddress address) override;
    void SetIpv4(Ptr<Ipv4> ipv4) override;
    void PrintRoutingTable(Ptr<OutputStreamWrapper> stream,
                           Time::Unit unit = Time::S) const override;

    // Handle protocol parameters
    /**
     * Get maximum queue time
     * \returns the maximum queue time
     */
    Time GetMaxQueueTime() const
    {
        return m_maxQueueTime;
    }

    /**
     * Set the maximum queue time
     * \param t the maximum queue time
     */
    void SetMaxQueueTime(Time t);

    /**
     * Get the maximum queue length
     * \returns the maximum queue length
     */
    uint32_t GetMaxQueueLen() const
    {
        return m_maxQueueLen;
    }

    /**
     * Set the maximum queue length
     * \param len the maximum queue length
     */
    void SetMaxQueueLen(uint32_t len);

    /**
     * Get destination only flag
     * \returns the destination only flag
     */
    bool GetDestinationOnlyFlag() const
    {
        return m_destinationOnly;
    }

    /**
     * Set destination only flag
     * \param f the destination only flag
     */
    void SetDestinationOnlyFlag(bool f)
    {
        m_destinationOnly = f;
    }

    /**
     * Get gratuitous reply flag
     * \returns the gratuitous reply flag
     */
    bool GetGratuitousReplyFlag() const
    {
        return m_gratuitousReply;
    }

    /**
     * Set gratuitous reply flag
     * \param f the gratuitous reply flag
     */
    void SetGratuitousReplyFlag(bool f)
    {
        m_gratuitousReply = f;
    }

    /**
     * Set hello enable
     * \param f the hello enable flag
     */
    void SetHelloEnable(bool f)
    {
        m_enableHello = f;
    }

    /**
     * Get hello enable flag
     * \returns the enable hello flag
     */
    bool GetHelloEnable() const
    {
        return m_enableHello;
    }

    /**
     * Set broadcast enable flag
     * \param f enable broadcast flag
     */
    void SetBroadcastEnable(bool f)
    {
        m_enableBroadcast = f;
    }

    /**
     * Get broadcast enable flag
     * \returns the broadcast enable flag
     */
    bool GetBroadcastEnable() const
    {
        return m_enableBroadcast;
    }

    /**
     * Assign a fixed random variable stream number to the random variables
     * used by this model.  Return the number of streams (possibly zero) that
     * have been assigned.
     *
     * \param stream first stream index to use
     * \return the number of stream indices assigned by this model
     */
    int64_t AssignStreams(int64_t stream);

    /**
     * Get adaptive routing statistics
     * \returns map of routing decision types to usage counts
     */
    std::map<ArpmecAdaptiveRouting::RouteDecision, uint32_t> GetAdaptiveRoutingStats() const;

    /**
     * Get clustering module
     * \returns pointer to the clustering component
     */
    Ptr<ArpmecClustering> GetClustering() const
    {
        return m_clustering;
    }

    /**
     * \brief Trace packet transmission
     * \param packet the transmitted packet
     * \param from source address
     * \param to destination address
     */
    void TraceTx(Ptr<const Packet> packet, const Address& from, const Address& to);

    /**
     * \brief Trace packet reception
     * \param packet the received packet
     * \param from source address
     */
    void TraceRx(Ptr<const Packet> packet, const Address& from);

    /**
     * \brief Trace cluster head status change
     * \param nodeId the node ID
     * \param isClusterHead true if node becomes cluster head
     */
    void TraceClusterHead(uint32_t nodeId, bool isClusterHead);

    /**
     * \brief Trace adaptive routing decision
     * \param nodeId the node ID making the decision
     * \param decision the routing decision as string
     */
    void TraceRouteDecision(uint32_t nodeId, const std::string& decision);

    /**
     * \brief Trace LQE update
     * \param nodeId the node ID
     * \param lqeValue the updated LQE value
     */
    void TraceLqeUpdate(uint32_t nodeId, double lqeValue);

    /**
     * \brief Trace energy update
     * \param nodeId the node ID
     * \param energyLevel the updated energy level
     */
    void TraceEnergyUpdate(uint32_t nodeId, double energyLevel);

    /**
     * \brief Enable MEC Gateway functionality on this node
     * \param gatewayId Unique gateway identifier
     * \param coverageArea Coverage area in meters
     */
    void EnableMecGateway(uint32_t gatewayId, double coverageArea);

    /**
     * \brief Enable MEC Edge Server functionality on this node
     * \param serverId Unique server identifier
     * \param processingCapacity Processing capacity (ops/sec)
     * \param memoryCapacity Memory capacity (MB)
     */
    void EnableMecServer(uint32_t serverId, uint32_t processingCapacity, uint32_t memoryCapacity);

    /**
     * \brief Get pointer to MEC Gateway (if enabled)
     * \return Pointer to MEC Gateway or nullptr
     */
    Ptr<ArpmecMecGateway> GetMecGateway() const
    {
        return m_mecGateway;
    }

    /**
     * \brief Get pointer to MEC Server (if enabled)
     * \return Pointer to MEC Server or nullptr
     */
    Ptr<ArpmecMecServer> GetMecServer() const
    {
        return m_mecServer;
    }

    /**
     * \brief Check if this node is a MEC Gateway
     * \return True if MEC Gateway is enabled
     */
    bool IsMecGateway() const
    {
        return m_isMecGateway;
    }

    /**
     * \brief Check if this node is a MEC Edge Server
     * \return True if MEC Server is enabled
     */
    bool IsMecServer() const
    {
        return m_isMecServer;
    }

    /**
     * Handle MEC cluster management operations
     * \param operation The cluster management operation
     * \param clusterId The cluster ID involved
     */
    void OnMecClusterManagement(ArpmecMecGateway::ClusterOperation operation, uint32_t clusterId);

    /**
     * Handle MEC task completion
     * \param taskId The completed task ID
     * \param clusterId The source cluster ID
     * \param processingTime Time taken to process the task
     */
    void OnMecTaskCompletion(uint32_t taskId, uint32_t clusterId, double processingTime);

    /**
     * Handle MEC cloud offload requests
     * \param task The task to be offloaded
     * \return True if cloud accepts the task
     */
    bool OnMecCloudOffload(ArpmecMecServer::ComputationTask task);

    /**
     * Send packet from MEC Gateway for inter-cluster communication
     * \param packet The packet to send
     * \param targetNodeId Target node ID
     */
    void SendPacketFromMecGateway(Ptr<Packet> packet, uint32_t targetNodeId);

  protected:
    void DoInitialize() override;

  private:
    /**
     * Notify that an MPDU was dropped.
     *
     * \param reason the reason why the MPDU was dropped
     * \param mpdu the dropped MPDU
     */
    void NotifyTxError(WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu);

    // Protocol parameters.
    uint32_t m_rreqRetries; ///< Maximum number of retransmissions of RREQ with TTL = NetDiameter to
                            ///< discover a route
    uint16_t m_ttlStart;    ///< Initial TTL value for RREQ.
    uint16_t m_ttlIncrement; ///< TTL increment for each attempt using the expanding ring search for
                             ///< RREQ dissemination.
    uint16_t m_ttlThreshold; ///< Maximum TTL value for expanding ring search, TTL = NetDiameter is
                             ///< used beyond this value.
    uint16_t m_timeoutBuffer;  ///< Provide a buffer for the timeout.
    uint16_t m_rreqRateLimit;  ///< Maximum number of RREQ per second.
    uint16_t m_rerrRateLimit;  ///< Maximum number of REER per second.
    Time m_activeRouteTimeout; ///< Period of time during which the route is considered to be valid.
    uint32_t m_netDiameter; ///< Net diameter measures the maximum possible number of hops between
                            ///< two nodes in the network
    /**
     * NodeTraversalTime is a conservative estimate of the average one hop traversal time for
     * packets and should include queuing delays, interrupt processing times and transfer times.
     */
    Time m_nodeTraversalTime;
    Time m_netTraversalTime;  ///< Estimate of the average net traversal time.
    Time m_pathDiscoveryTime; ///< Estimate of maximum time needed to find route in network.
    Time m_myRouteTimeout;    ///< Value of lifetime field in RREP generating by this node.
    /**
     * Every HelloInterval the node checks whether it has sent a broadcast  within the last
     * HelloInterval. If it has not, it MAY broadcast a  Hello message
     */
    Time m_helloInterval;
    uint32_t m_allowedHelloLoss; ///< Number of hello messages which may be loss for valid link
    /**
     * DeletePeriod is intended to provide an upper bound on the time for which an upstream node A
     * can have a neighbor B as an active next hop for destination D, while B has invalidated the
     * route to D.
     */
    Time m_deletePeriod;
    Time m_nextHopWait;      ///< Period of our waiting for the neighbour's RREP_ACK
    Time m_blackListTimeout; ///< Time for which the node is put into the blacklist
    uint32_t m_maxQueueLen;  ///< The maximum number of packets that we allow a routing protocol to
                             ///< buffer.
    Time m_maxQueueTime;     ///< The maximum period of time that a routing protocol is allowed to
                             ///< buffer a packet for.
    bool m_destinationOnly;  ///< Indicates only the destination may respond to this RREQ.
    bool m_gratuitousReply;  ///< Indicates whether a gratuitous RREP should be unicast to the node
                             ///< originated route discovery.
    bool m_enableHello;      ///< Indicates whether a hello messages enable
    bool m_enableBroadcast;  ///< Indicates whether a a broadcast data packets forwarding enable

    /// IP protocol
    Ptr<Ipv4> m_ipv4;
    /// Raw unicast socket per each IP interface, map socket -> iface address (IP + mask)
    std::map<Ptr<Socket>, Ipv4InterfaceAddress> m_socketAddresses;
    /// Raw subnet directed broadcast socket per each IP interface, map socket -> iface address (IP
    /// + mask)
    std::map<Ptr<Socket>, Ipv4InterfaceAddress> m_socketSubnetBroadcastAddresses;
    /// Loopback device used to defer RREQ until packet will be fully formed
    Ptr<NetDevice> m_lo;

    /// Routing table
    RoutingTable m_routingTable;
    /// A "drop-front" queue used by the routing layer to buffer packets to which it does not have a
    /// route.
    RequestQueue m_queue;
    /// Broadcast ID
    uint32_t m_requestId;
    /// Request sequence number
    uint32_t m_seqNo;
    /// Handle duplicated RREQ
    IdCache m_rreqIdCache;
    /// Handle duplicated broadcast/multicast packets
    DuplicatePacketDetection m_dpd;
    /// Handle neighbors
    Neighbors m_nb;
    /// Number of RREQs used for RREQ rate control
    uint16_t m_rreqCount;
    /// Number of RERRs used for RERR rate control
    uint16_t m_rerrCount;

    /// ARPMEC LQE module for link quality estimation
    Ptr<ArpmecLqe> m_lqe;
    /// ARPMEC Clustering module for cluster management
    Ptr<ArpmecClustering> m_clustering;
    /// ARPMEC Adaptive Routing module for Algorithm 3 implementation
    Ptr<ArpmecAdaptiveRouting> m_adaptiveRouting;

  private:
    /// Start protocol operation
    void Start();
    /**
     * Queue packet and send route request
     *
     * \param p the packet to route
     * \param header the IP header
     * \param ucb the UnicastForwardCallback function
     * \param ecb the ErrorCallback function
     */
    void DeferredRouteOutput(Ptr<const Packet> p,
                             const Ipv4Header& header,
                             UnicastForwardCallback ucb,
                             ErrorCallback ecb);
    /**
     * If route exists and is valid, forward packet.
     *
     * \param p the packet to route
     * \param header the IP header
     * \param ucb the UnicastForwardCallback function
     * \param ecb the ErrorCallback function
     * \returns true if forwarded
     */
    bool Forwarding(Ptr<const Packet> p,
                    const Ipv4Header& header,
                    UnicastForwardCallback ucb,
                    ErrorCallback ecb);
    /**
     * Repeated attempts by a source node at route discovery for a single destination
     * use the expanding ring search technique.
     * \param dst the destination IP address
     */
    void ScheduleRreqRetry(Ipv4Address dst);
    /**
     * Set lifetime field in routing table entry to the maximum of existing lifetime and lt, if the
     * entry exists
     * \param addr destination address
     * \param lt proposed time for lifetime field in routing table entry for destination with
     * address addr.
     * \return true if route to destination address addr exist
     */
    bool UpdateRouteLifeTime(Ipv4Address addr, Time lt);
    /**
     * Update neighbor record.
     * \param receiver is supposed to be my interface
     * \param sender is supposed to be IP address of my neighbor.
     */
    void UpdateRouteToNeighbor(Ipv4Address sender, Ipv4Address receiver);
    /**
     * Test whether the provided address is assigned to an interface on this node
     * \param src the source IP address
     * \returns true if the IP address is the node's IP address
     */
    bool IsMyOwnAddress(Ipv4Address src);
    /**
     * Find unicast socket with local interface address iface
     *
     * \param iface the interface
     * \returns the socket associated with the interface
     */
    Ptr<Socket> FindSocketWithInterfaceAddress(Ipv4InterfaceAddress iface) const;
    /**
     * Find subnet directed broadcast socket with local interface address iface
     *
     * \param iface the interface
     * \returns the socket associated with the interface
     */
    Ptr<Socket> FindSubnetBroadcastSocketWithInterfaceAddress(Ipv4InterfaceAddress iface) const;
    /**
     * Process hello message
     *
     * \param rrepHeader RREP message header
     * \param receiverIfaceAddr receiver interface IP address
     */
    void ProcessHello(const RrepHeader& rrepHeader, Ipv4Address receiverIfaceAddr);
    /**
     * Create loopback route for given header
     *
     * \param header the IP header
     * \param oif the output interface net device
     * \returns the route
     */
    Ptr<Ipv4Route> LoopbackRoute(const Ipv4Header& header, Ptr<NetDevice> oif) const;

    /**
     * \name Receive control packets
     * @{
     */
    /**
     * Receive and process control packet
     * \param socket input socket
     */
    void RecvArpmec(Ptr<Socket> socket);
    /**
     * Receive RREQ
     * \param p packet
     * \param receiver receiver address
     * \param src sender address
     */
    void RecvRequest(Ptr<Packet> p, Ipv4Address receiver, Ipv4Address src);
    /**
     * Receive RREP
     * \param p packet
     * \param my destination address
     * \param src sender address
     */
    void RecvReply(Ptr<Packet> p, Ipv4Address my, Ipv4Address src);
    /**
     * Receive RREP_ACK
     * \param neighbor neighbor address
     */
    void RecvReplyAck(Ipv4Address neighbor);
    /**
     * Receive RERR
     * \param p packet
     * \param src sender address
     */
    /// Receive  from node with address src
    void RecvError(Ptr<Packet> p, Ipv4Address src);

    /**
     * Process ARPMEC HELLO message
     * \param p packet containing HELLO header
     * \param src sender address
     */
    void ProcessArpmecHello(Ptr<Packet> p, Ipv4Address src);

    /**
     * Process ARPMEC JOIN message
     * \param p packet containing JOIN header
     * \param src sender address
     */
    void ProcessArpmecJoin(Ptr<Packet> p, Ipv4Address src);

    /**
     * Process ARPMEC CH_NOTIFICATION message
     * \param p packet containing CH_NOTIFICATION header
     * \param src sender address
     */
    void ProcessArpmecChNotification(Ptr<Packet> p, Ipv4Address src);

    /**
     * Convert IPv4 address to node ID
     * \param address IPv4 address to convert
     * \return node ID
     */
    uint32_t GetNodeIdFromAddress(Ipv4Address address);

    /**
     * Convert node ID to IP address
     * \param nodeId the node ID
     * \return the IP address for this node
     */
    Ipv4Address GetAddressFromNodeId(uint32_t nodeId);

    /**
     * \name Callback methods for ARPMEC modules
     * @{
     */
    /**
     * Handle cluster events from clustering module
     * \param event the cluster event
     * \param nodeId the node ID involved in the event
     */
    void OnClusterEvent(ArpmecClustering::ClusterEvent event, uint32_t nodeId);

    /**
     * Handle routing decision from adaptive routing module
     * \param decision the routing decision made
     * \param quality the route quality score
     */
    void OnRoutingDecision(ArpmecAdaptiveRouting::RouteDecision decision, double quality);

    /**
     * Handle packet send requests from clustering module
     * \param packet the packet to send
     * \param destination the destination node ID (0 for broadcast)
     */
    void OnClusterPacketSend(Ptr<Packet> packet, uint32_t destination);
    /** @} */

    /**
     * Send clustering packets (callback for clustering module)
     * \param packet packet to send
     * \param destination destination node ID (0 for broadcast)
     */
    void SendClusteringPacket(Ptr<Packet> packet, uint32_t destination);
    /** @} */

    /**
     * \name Send
     * @{
     */
    /** Forward packet from route request queue
     * \param dst destination address
     * \param route route to use
     */
    void SendPacketFromQueue(Ipv4Address dst, Ptr<Ipv4Route> route);
    /// Send hello
    void SendHello();
    /** Send RREQ
     * \param dst destination address
     */
    void SendRequest(Ipv4Address dst);
    /** Send RREP
     * \param rreqHeader route request header
     * \param toOrigin routing table entry to originator
     */
    void SendReply(const RreqHeader& rreqHeader, const RoutingTableEntry& toOrigin);
    /** Send RREP by intermediate node
     * \param toDst routing table entry to destination
     * \param toOrigin routing table entry to originator
     * \param gratRep indicates whether a gratuitous RREP should be unicast to destination
     */
    void SendReplyByIntermediateNode(RoutingTableEntry& toDst,
                                     RoutingTableEntry& toOrigin,
                                     bool gratRep);
    /** Send RREP_ACK
     * \param neighbor neighbor address
     */
    void SendReplyAck(Ipv4Address neighbor);
    /** Initiate RERR
     * \param nextHop next hop address
     */
    void SendRerrWhenBreaksLinkToNextHop(Ipv4Address nextHop);
    /** Forward RERR
     * \param packet packet
     * \param precursors list of addresses of the visited nodes
     */
    void SendRerrMessage(Ptr<Packet> packet, std::vector<Ipv4Address> precursors);
    /**
     * Send RERR message when no route to forward input packet. Unicast if there is reverse route to
     * originating node, broadcast otherwise.
     * \param dst destination node IP address
     * \param dstSeqNo destination node sequence number
     * \param origin originating node IP address
     */
    void SendRerrWhenNoRouteToForward(Ipv4Address dst, uint32_t dstSeqNo, Ipv4Address origin);
    /** @} */

    /**
     * Send packet to destination socket
     * \param socket destination node socket
     * \param packet packet to send
     * \param destination destination node IP address
     */
    void SendTo(Ptr<Socket> socket, Ptr<Packet> packet, Ipv4Address destination);

    /// Hello timer
    Timer m_htimer;
    /// Schedule next send of hello message
    void HelloTimerExpire();
    /// RREQ rate limit timer
    Timer m_rreqRateLimitTimer;
    /// Reset RREQ count and schedule RREQ rate limit timer with delay 1 sec.
    void RreqRateLimitTimerExpire();
    /// RERR rate limit timer
    Timer m_rerrRateLimitTimer;
    /// Reset RERR count and schedule RERR rate limit timer with delay 1 sec.
    void RerrRateLimitTimerExpire();
    /// Map IP address + RREQ timer.
    std::map<Ipv4Address, Timer> m_addressReqTimer;
    /**
     * Handle route discovery process
     * \param dst the destination IP address
     */
    void RouteRequestTimerExpire(Ipv4Address dst);
    /**
     * Mark link to neighbor node as unidirectional for blacklistTimeout
     *
     * \param neighbor the IP address of the neighbor node
     * \param blacklistTimeout the black list timeout time
     */
    void AckTimerExpire(Ipv4Address neighbor, Time blacklistTimeout);

    /// Provides uniform random variables.
    Ptr<UniformRandomVariable> m_uniformRandomVariable;
    /// Keep track of the last bcast time
    Time m_lastBcastTime;

    /// ARPMEC Trace Sources for validation and monitoring
    /// Trace fired when a packet is transmitted
    TracedCallback<Ptr<const Packet>, const Address&, const Address&> m_txTrace;
    /// Trace fired when a packet is received
    TracedCallback<Ptr<const Packet>, const Address&> m_rxTrace;
    /// Trace fired when cluster head status changes
    TracedCallback<uint32_t, bool> m_clusterHeadTrace;
    /// Trace fired when adaptive routing decision is made
    TracedCallback<uint32_t, std::string> m_routeDecisionTrace;
    /// Trace fired when LQE value is updated
    TracedCallback<uint32_t, double> m_lqeUpdateTrace;
    /// Trace fired when energy level is updated
    TracedCallback<uint32_t, double> m_energyUpdateTrace;

private:
    // MEC Infrastructure
    Ptr<ArpmecMecGateway> m_mecGateway;   ///< MEC Gateway instance
    Ptr<ArpmecMecServer> m_mecServer;     ///< MEC Server instance
    bool m_isMecGateway;                  ///< Whether this node is a MEC Gateway
    bool m_isMecServer;                   ///< Whether this node is a MEC Server
};

} // namespace arpmec
} // namespace ns3

#endif /* ARPMECROUTINGPROTOCOL_H */
