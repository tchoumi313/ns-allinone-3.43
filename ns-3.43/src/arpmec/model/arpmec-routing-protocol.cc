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
#define NS_LOG_APPEND_CONTEXT                                                                      \
    if (m_ipv4)                                                                                    \
    {                                                                                              \
        std::clog << "[node " << m_ipv4->GetObject<Node>()->GetId() << "] ";                       \
    }

#include "arpmec-routing-protocol.h"

#include "ns3/adhoc-wifi-mac.h"
#include "ns3/boolean.h"
#include "ns3/inet-socket-address.h"
#include "ns3/log.h"
#include "ns3/pointer.h"
#include "ns3/random-variable-stream.h"
#include "ns3/string.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/udp-header.h"
#include "ns3/udp-l4-protocol.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/wifi-mpdu.h"
#include "ns3/wifi-net-device.h"

#include <algorithm>
#include <limits>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ArpmecRoutingProtocol");

namespace arpmec
{
NS_OBJECT_ENSURE_REGISTERED(RoutingProtocol);

/// UDP Port for ARPMEC control traffic
const uint32_t RoutingProtocol::ARPMEC_PORT = 654;

/**
 * \ingroup arpmec
 * \brief Tag used by ARPMEC implementation
 */
class DeferredRouteOutputTag : public Tag
{
  public:
    /**
     * \brief Constructor
     * \param o the output interface
     */
    DeferredRouteOutputTag(int32_t o = -1)
        : Tag(),
          m_oif(o)
    {
    }

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId()
    {
        static TypeId tid = TypeId("ns3::arpmec::DeferredRouteOutputTag")
                                .SetParent<Tag>()
                                .SetGroupName("Arpmec")
                                .AddConstructor<DeferredRouteOutputTag>();
        return tid;
    }

    TypeId GetInstanceTypeId() const override
    {
        return GetTypeId();
    }

    /**
     * \brief Get the output interface
     * \return the output interface
     */
    int32_t GetInterface() const
    {
        return m_oif;
    }

    /**
     * \brief Set the output interface
     * \param oif the output interface
     */
    void SetInterface(int32_t oif)
    {
        m_oif = oif;
    }

    uint32_t GetSerializedSize() const override
    {
        return sizeof(int32_t);
    }

    void Serialize(TagBuffer i) const override
    {
        i.WriteU32(m_oif);
    }

    void Deserialize(TagBuffer i) override
    {
        m_oif = i.ReadU32();
    }

    void Print(std::ostream& os) const override
    {
        os << "DeferredRouteOutputTag: output interface = " << m_oif;
    }

  private:
    /// Positive if output device is fixed in RouteOutput
    int32_t m_oif;
};

NS_OBJECT_ENSURE_REGISTERED(DeferredRouteOutputTag);

//-----------------------------------------------------------------------------
RoutingProtocol::RoutingProtocol()
    : m_rreqRetries(2),
      m_ttlStart(1),
      m_ttlIncrement(2),
      m_ttlThreshold(7),
      m_timeoutBuffer(2),
      m_rreqRateLimit(10),
      m_rerrRateLimit(10),
      m_activeRouteTimeout(Seconds(3)),
      m_netDiameter(35),
      m_nodeTraversalTime(MilliSeconds(40)),
      m_netTraversalTime(Time((2 * m_netDiameter) * m_nodeTraversalTime)),
      m_pathDiscoveryTime(Time(2 * m_netTraversalTime)),
      m_myRouteTimeout(Time(2 * std::max(m_pathDiscoveryTime, m_activeRouteTimeout))),
      m_helloInterval(Seconds(1)),
      m_allowedHelloLoss(2),
      m_deletePeriod(Time(5 * std::max(m_activeRouteTimeout, m_helloInterval))),
      m_nextHopWait(m_nodeTraversalTime + MilliSeconds(10)),
      m_blackListTimeout(Time(m_rreqRetries * m_netTraversalTime)),
      m_maxQueueLen(64),
      m_maxQueueTime(Seconds(30)),
      m_destinationOnly(false),
      m_gratuitousReply(true),
      m_enableHello(false),
      m_routingTable(m_deletePeriod),
      m_queue(m_maxQueueLen, m_maxQueueTime),
      m_requestId(0),
      m_seqNo(0),
      m_rreqIdCache(m_pathDiscoveryTime),
      m_dpd(m_pathDiscoveryTime),
      m_nb(m_helloInterval),
      m_rreqCount(0),
      m_rerrCount(0),
      m_htimer(Timer::CANCEL_ON_DESTROY),
      m_rreqRateLimitTimer(Timer::CANCEL_ON_DESTROY),
      m_rerrRateLimitTimer(Timer::CANCEL_ON_DESTROY),
      m_lastBcastTime(Seconds(0)),
      m_isMecGateway(false),
      m_isMecServer(false)
{
    m_nb.SetCallback(MakeCallback(&RoutingProtocol::SendRerrWhenBreaksLinkToNextHop, this));
    
    // Initialize ARPMEC modules
    m_lqe = CreateObject<ArpmecLqe>();
    m_clustering = CreateObject<ArpmecClustering>();
    m_adaptiveRouting = CreateObject<ArpmecAdaptiveRouting>();
}

TypeId
RoutingProtocol::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::arpmec::RoutingProtocol")
            .SetParent<Ipv4RoutingProtocol>()
            .SetGroupName("Arpmec")
            .AddConstructor<RoutingProtocol>()
            .AddAttribute("HelloInterval",
                          "HELLO messages emission interval.",
                          TimeValue(Seconds(1)),
                          MakeTimeAccessor(&RoutingProtocol::m_helloInterval),
                          MakeTimeChecker())
            .AddAttribute("TtlStart",
                          "Initial TTL value for RREQ.",
                          UintegerValue(1),
                          MakeUintegerAccessor(&RoutingProtocol::m_ttlStart),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("TtlIncrement",
                          "TTL increment for each attempt using the expanding ring search for RREQ "
                          "dissemination.",
                          UintegerValue(2),
                          MakeUintegerAccessor(&RoutingProtocol::m_ttlIncrement),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("TtlThreshold",
                          "Maximum TTL value for expanding ring search, TTL = NetDiameter is used "
                          "beyond this value.",
                          UintegerValue(7),
                          MakeUintegerAccessor(&RoutingProtocol::m_ttlThreshold),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("TimeoutBuffer",
                          "Provide a buffer for the timeout.",
                          UintegerValue(2),
                          MakeUintegerAccessor(&RoutingProtocol::m_timeoutBuffer),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("RreqRetries",
                          "Maximum number of retransmissions of RREQ to discover a route",
                          UintegerValue(2),
                          MakeUintegerAccessor(&RoutingProtocol::m_rreqRetries),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("RreqRateLimit",
                          "Maximum number of RREQ per second.",
                          UintegerValue(10),
                          MakeUintegerAccessor(&RoutingProtocol::m_rreqRateLimit),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("RerrRateLimit",
                          "Maximum number of RERR per second.",
                          UintegerValue(10),
                          MakeUintegerAccessor(&RoutingProtocol::m_rerrRateLimit),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("NodeTraversalTime",
                          "Conservative estimate of the average one hop traversal time for packets "
                          "and should include "
                          "queuing delays, interrupt processing times and transfer times.",
                          TimeValue(MilliSeconds(40)),
                          MakeTimeAccessor(&RoutingProtocol::m_nodeTraversalTime),
                          MakeTimeChecker())
            .AddAttribute(
                "NextHopWait",
                "Period of our waiting for the neighbour's RREP_ACK = 10 ms + NodeTraversalTime",
                TimeValue(MilliSeconds(50)),
                MakeTimeAccessor(&RoutingProtocol::m_nextHopWait),
                MakeTimeChecker())
            .AddAttribute("ActiveRouteTimeout",
                          "Period of time during which the route is considered to be valid",
                          TimeValue(Seconds(3)),
                          MakeTimeAccessor(&RoutingProtocol::m_activeRouteTimeout),
                          MakeTimeChecker())
            .AddAttribute("MyRouteTimeout",
                          "Value of lifetime field in RREP generating by this node = 2 * "
                          "max(ActiveRouteTimeout, PathDiscoveryTime)",
                          TimeValue(Seconds(11.2)),
                          MakeTimeAccessor(&RoutingProtocol::m_myRouteTimeout),
                          MakeTimeChecker())
            .AddAttribute("BlackListTimeout",
                          "Time for which the node is put into the blacklist = RreqRetries * "
                          "NetTraversalTime",
                          TimeValue(Seconds(5.6)),
                          MakeTimeAccessor(&RoutingProtocol::m_blackListTimeout),
                          MakeTimeChecker())
            .AddAttribute("DeletePeriod",
                          "DeletePeriod is intended to provide an upper bound on the time for "
                          "which an upstream node A "
                          "can have a neighbor B as an active next hop for destination D, while B "
                          "has invalidated the route to D."
                          " = 5 * max (HelloInterval, ActiveRouteTimeout)",
                          TimeValue(Seconds(15)),
                          MakeTimeAccessor(&RoutingProtocol::m_deletePeriod),
                          MakeTimeChecker())
            .AddAttribute("NetDiameter",
                          "Net diameter measures the maximum possible number of hops between two "
                          "nodes in the network",
                          UintegerValue(35),
                          MakeUintegerAccessor(&RoutingProtocol::m_netDiameter),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute(
                "NetTraversalTime",
                "Estimate of the average net traversal time = 2 * NodeTraversalTime * NetDiameter",
                TimeValue(Seconds(2.8)),
                MakeTimeAccessor(&RoutingProtocol::m_netTraversalTime),
                MakeTimeChecker())
            .AddAttribute(
                "PathDiscoveryTime",
                "Estimate of maximum time needed to find route in network = 2 * NetTraversalTime",
                TimeValue(Seconds(5.6)),
                MakeTimeAccessor(&RoutingProtocol::m_pathDiscoveryTime),
                MakeTimeChecker())
            .AddAttribute("MaxQueueLen",
                          "Maximum number of packets that we allow a routing protocol to buffer.",
                          UintegerValue(64),
                          MakeUintegerAccessor(&RoutingProtocol::SetMaxQueueLen,
                                               &RoutingProtocol::GetMaxQueueLen),
                          MakeUintegerChecker<uint32_t>())
            .AddAttribute("MaxQueueTime",
                          "Maximum time packets can be queued (in seconds)",
                          TimeValue(Seconds(30)),
                          MakeTimeAccessor(&RoutingProtocol::SetMaxQueueTime,
                                           &RoutingProtocol::GetMaxQueueTime),
                          MakeTimeChecker())
            .AddAttribute("AllowedHelloLoss",
                          "Number of hello messages which may be loss for valid link.",
                          UintegerValue(2),
                          MakeUintegerAccessor(&RoutingProtocol::m_allowedHelloLoss),
                          MakeUintegerChecker<uint16_t>())
            .AddAttribute("GratuitousReply",
                          "Indicates whether a gratuitous RREP should be unicast to the node "
                          "originated route discovery.",
                          BooleanValue(true),
                          MakeBooleanAccessor(&RoutingProtocol::SetGratuitousReplyFlag,
                                              &RoutingProtocol::GetGratuitousReplyFlag),
                          MakeBooleanChecker())
            .AddAttribute("DestinationOnly",
                          "Indicates only the destination may respond to this RREQ.",
                          BooleanValue(false),
                          MakeBooleanAccessor(&RoutingProtocol::SetDestinationOnlyFlag,
                                              &RoutingProtocol::GetDestinationOnlyFlag),
                          MakeBooleanChecker())
            .AddAttribute("EnableHello",
                          "Indicates whether a hello messages enable.",
                          BooleanValue(true),
                          MakeBooleanAccessor(&RoutingProtocol::SetHelloEnable,
                                              &RoutingProtocol::GetHelloEnable),
                          MakeBooleanChecker())
            .AddAttribute("EnableBroadcast",
                          "Indicates whether a broadcast data packets forwarding enable.",
                          BooleanValue(true),
                          MakeBooleanAccessor(&RoutingProtocol::SetBroadcastEnable,
                                              &RoutingProtocol::GetBroadcastEnable),
                          MakeBooleanChecker())
            .AddAttribute("UniformRv",
                          "Access to the underlying UniformRandomVariable",
                          StringValue("ns3::UniformRandomVariable"),
                          MakePointerAccessor(&RoutingProtocol::m_uniformRandomVariable),
                          MakePointerChecker<UniformRandomVariable>())
            .AddTraceSource("Tx",
                          "A packet is transmitted by the routing protocol",
                          MakeTraceSourceAccessor(&RoutingProtocol::m_txTrace),
                          "ns3::Packet::TwoAddressTracedCallback")
            .AddTraceSource("Rx", 
                          "A packet is received by the routing protocol",
                          MakeTraceSourceAccessor(&RoutingProtocol::m_rxTrace),
                          "ns3::Packet::AddressTracedCallback")
            .AddTraceSource("ClusterHead",
                          "Cluster head status change",
                          MakeTraceSourceAccessor(&RoutingProtocol::m_clusterHeadTrace),
                          "ns3::arpmec::RoutingProtocol::ClusterHeadTracedCallback")
            .AddTraceSource("RouteDecision",
                          "Adaptive routing decision made",
                          MakeTraceSourceAccessor(&RoutingProtocol::m_routeDecisionTrace),
                          "ns3::arpmec::RoutingProtocol::RouteDecisionTracedCallback") 
            .AddTraceSource("LqeUpdate",
                          "LQE value updated",
                          MakeTraceSourceAccessor(&RoutingProtocol::m_lqeUpdateTrace),
                          "ns3::arpmec::RoutingProtocol::LqeUpdateTracedCallback")
            .AddTraceSource("EnergyUpdate",
                          "Energy level updated", 
                          MakeTraceSourceAccessor(&RoutingProtocol::m_energyUpdateTrace),
                          "ns3::arpmec::RoutingProtocol::EnergyUpdateTracedCallback");
    return tid;
}

void
RoutingProtocol::SetMaxQueueLen(uint32_t len)
{
    m_maxQueueLen = len;
    m_queue.SetMaxQueueLen(len);
}

void
RoutingProtocol::SetMaxQueueTime(Time t)
{
    m_maxQueueTime = t;
    m_queue.SetQueueTimeout(t);
}

RoutingProtocol::~RoutingProtocol()
{
}

void
RoutingProtocol::DoDispose()
{
    m_ipv4 = nullptr;
    for (auto iter = m_socketAddresses.begin(); iter != m_socketAddresses.end(); iter++)
    {
        iter->first->Close();
    }
    m_socketAddresses.clear();
    for (auto iter = m_socketSubnetBroadcastAddresses.begin();
         iter != m_socketSubnetBroadcastAddresses.end();
         iter++)
    {
        iter->first->Close();
    }
    m_socketSubnetBroadcastAddresses.clear();
    Ipv4RoutingProtocol::DoDispose();
}

void
RoutingProtocol::PrintRoutingTable(Ptr<OutputStreamWrapper> stream, Time::Unit unit) const
{
    *stream->GetStream() << "Node: " << m_ipv4->GetObject<Node>()->GetId()
                         << "; Time: " << Now().As(unit)
                         << ", Local time: " << m_ipv4->GetObject<Node>()->GetLocalTime().As(unit)
                         << ", ARPMEC Routing table" << std::endl;

    m_routingTable.Print(stream, unit);
    *stream->GetStream() << std::endl;
}

int64_t
RoutingProtocol::AssignStreams(int64_t stream)
{
    NS_LOG_FUNCTION(this << stream);
    m_uniformRandomVariable->SetStream(stream);
    return 1;
}

void
RoutingProtocol::Start()
{
    NS_LOG_FUNCTION(this);
    if (m_enableHello)
    {
        m_nb.ScheduleTimer();
    }
    m_rreqRateLimitTimer.SetFunction(&RoutingProtocol::RreqRateLimitTimerExpire, this);
    m_rreqRateLimitTimer.Schedule(Seconds(1));

    m_rerrRateLimitTimer.SetFunction(&RoutingProtocol::RerrRateLimitTimerExpire, this);
    m_rerrRateLimitTimer.Schedule(Seconds(1));

    // Start ARPMEC clustering protocol
    m_clustering->Start();
}

Ptr<Ipv4Route>
RoutingProtocol::RouteOutput(Ptr<Packet> p,
                             const Ipv4Header& header,
                             Ptr<NetDevice> oif,
                             Socket::SocketErrno& sockerr)
{
    NS_LOG_FUNCTION(this << header << (oif ? oif->GetIfIndex() : 0));
    if (!p)
    {
        NS_LOG_DEBUG("Packet is == 0");
        return LoopbackRoute(header, oif); // later
    }
    if (m_socketAddresses.empty())
    {
        sockerr = Socket::ERROR_NOROUTETOHOST;
        NS_LOG_LOGIC("No arpmec interfaces");
        Ptr<Ipv4Route> route;
        return route;
    }
    sockerr = Socket::ERROR_NOTERROR;
    Ptr<Ipv4Route> route;
    Ipv4Address dst = header.GetDestination();
    RoutingTableEntry rt;
    if (m_routingTable.LookupValidRoute(dst, rt))
    {
        route = rt.GetRoute();
        NS_ASSERT(route);
        NS_LOG_DEBUG("Exist route to " << route->GetDestination() << " from interface "
                                       << route->GetSource());
        if (oif && route->GetOutputDevice() != oif)
        {
            NS_LOG_DEBUG("Output device doesn't match. Dropped.");
            sockerr = Socket::ERROR_NOROUTETOHOST;
            return Ptr<Ipv4Route>();
        }
        UpdateRouteLifeTime(dst, m_activeRouteTimeout);
        UpdateRouteLifeTime(route->GetGateway(), m_activeRouteTimeout);
        return route;
    }

    // Valid route not found, in this case we check ARPMEC adaptive routing
    // before falling back to traditional AODV route discovery
    uint32_t destinationNodeId = GetNodeIdFromAddress(dst);
    
    // Use Algorithm 3 - Adaptive Routing to determine best route
    if (m_adaptiveRouting)
    {
        ArpmecAdaptiveRouting::RoutingInfo routeInfo = m_adaptiveRouting->DetermineRoute(dst, destinationNodeId);
        
        NS_LOG_DEBUG("Adaptive routing decision: " << routeInfo.decision << 
                     " for destination " << dst << " (node " << destinationNodeId << ")");
        
        // Check if we have a specific routing recommendation
        if (routeInfo.decision == ArpmecAdaptiveRouting::INTRA_CLUSTER)
        {
            NS_LOG_DEBUG("Using intra-cluster routing to " << dst);
            // For intra-cluster routing, try to find direct route or via cluster head
            if (routeInfo.nextHop != 0)
            {
                // Try to get route to next hop
                Ipv4Address nextHopAddr = GetAddressFromNodeId(routeInfo.nextHop);
                RoutingTableEntry nextHopRt;
                if (m_routingTable.LookupValidRoute(nextHopAddr, nextHopRt))
                {
                    // Create route via next hop
                    route = nextHopRt.GetRoute();
                    UpdateRouteLifeTime(nextHopAddr, m_activeRouteTimeout);
                    return route;
                }
            }
        }
        else if (routeInfo.decision == ArpmecAdaptiveRouting::INTER_CLUSTER)
        {
            NS_LOG_DEBUG("Using inter-cluster routing to " << dst << " via gateway " << routeInfo.gateway);
            
            // Check if we are a MEC Gateway and should handle inter-cluster communication
            if (m_isMecGateway && m_mecGateway)
            {
                // Get source cluster (our cluster)
                uint32_t sourceCluster = 0;
                if (m_clustering && m_clustering->IsInCluster())
                {
                    sourceCluster = m_clustering->GetClusterHeadId();
                }
                
                // Trigger MEC Gateway inter-cluster communication
                Ptr<Packet> packetCopy = p->Copy();
                m_mecGateway->ProcessClusterMessage(packetCopy, sourceCluster);
                
                // Create a dummy route to satisfy the interface
                route = Create<Ipv4Route>();
                route->SetDestination(dst);
                route->SetGateway(m_ipv4->GetAddress(1, 0).GetLocal());
                route->SetSource(m_ipv4->GetAddress(1, 0).GetLocal());
                route->SetOutputDevice(m_ipv4->GetNetDevice(1));
                return route;
            }
            
            // For inter-cluster routing, route via cluster head or gateway
            if (routeInfo.nextHop != 0)
            {
                Ipv4Address nextHopAddr = GetAddressFromNodeId(routeInfo.nextHop);
                RoutingTableEntry nextHopRt;
                if (m_routingTable.LookupValidRoute(nextHopAddr, nextHopRt))
                {
                    route = nextHopRt.GetRoute();
                    UpdateRouteLifeTime(nextHopAddr, m_activeRouteTimeout);
                    return route;
                }
            }
        }
        // For AODV_FALLBACK, continue with standard AODV below
    }
    
    // If adaptive routing didn't provide a route, fall back to standard AODV behavior
    uint32_t iif = (oif ? m_ipv4->GetInterfaceForDevice(oif) : -1);
    DeferredRouteOutputTag tag(iif);
    NS_LOG_DEBUG("Valid Route not found, using AODV fallback for " << dst);
    if (!p->PeekPacketTag(tag))
    {
        p->AddPacketTag(tag);
    }
    return LoopbackRoute(header, oif);
}

void
RoutingProtocol::DeferredRouteOutput(Ptr<const Packet> p,
                                     const Ipv4Header& header,
                                     UnicastForwardCallback ucb,
                                     ErrorCallback ecb)
{
    NS_LOG_FUNCTION(this << p << header);
    NS_ASSERT(p && p != Ptr<Packet>());

    QueueEntry newEntry(p, header, ucb, ecb);
    bool result = m_queue.Enqueue(newEntry);
    if (result)
    {
        NS_LOG_LOGIC("Add packet " << p->GetUid() << " to queue. Protocol "
                                   << (uint16_t)header.GetProtocol());
        RoutingTableEntry rt;
        bool result = m_routingTable.LookupRoute(header.GetDestination(), rt);
        if (!result || ((rt.GetFlag() != IN_SEARCH) && result))
        {
            NS_LOG_LOGIC("Send new RREQ for outbound packet to " << header.GetDestination());
            SendRequest(header.GetDestination());
        }
    }
}

bool
RoutingProtocol::RouteInput(Ptr<const Packet> p,
                            const Ipv4Header& header,
                            Ptr<const NetDevice> idev,
                            const UnicastForwardCallback& ucb,
                            const MulticastForwardCallback& mcb,
                            const LocalDeliverCallback& lcb,
                            const ErrorCallback& ecb)
{
    NS_LOG_FUNCTION(this << p->GetUid() << header.GetDestination() << idev->GetAddress());
    if (m_socketAddresses.empty())
    {
        NS_LOG_LOGIC("No arpmec interfaces");
        return false;
    }
    NS_ASSERT(m_ipv4);
    NS_ASSERT(p);
    // Check if input device supports IP
    NS_ASSERT(m_ipv4->GetInterfaceForDevice(idev) >= 0);
    int32_t iif = m_ipv4->GetInterfaceForDevice(idev);

    Ipv4Address dst = header.GetDestination();
    Ipv4Address origin = header.GetSource();

    // Deferred route request
    if (idev == m_lo)
    {
        DeferredRouteOutputTag tag;
        if (p->PeekPacketTag(tag))
        {
            DeferredRouteOutput(p, header, ucb, ecb);
            return true;
        }
    }

    // Duplicate of own packet
    if (IsMyOwnAddress(origin))
    {
        return true;
    }

    // ARPMEC is not a multicast routing protocol
    if (dst.IsMulticast())
    {
        return false;
    }

    // Broadcast local delivery/forwarding
    for (auto j = m_socketAddresses.begin(); j != m_socketAddresses.end(); ++j)
    {
        Ipv4InterfaceAddress iface = j->second;
        if (m_ipv4->GetInterfaceForAddress(iface.GetLocal()) == iif)
        {
            if (dst == iface.GetBroadcast() || dst.IsBroadcast())
            {
                if (m_dpd.IsDuplicate(p, header))
                {
                    NS_LOG_DEBUG("Duplicated packet " << p->GetUid() << " from " << origin
                                                      << ". Drop.");
                    return true;
                }
                UpdateRouteLifeTime(origin, m_activeRouteTimeout);
                Ptr<Packet> packet = p->Copy();
                if (!lcb.IsNull())
                {
                    NS_LOG_LOGIC("Broadcast local delivery to " << iface.GetLocal());
                    lcb(p, header, iif);
                    // Fall through to additional processing
                }
                else
                {
                    NS_LOG_ERROR("Unable to deliver packet locally due to null callback "
                                 << p->GetUid() << " from " << origin);
                    ecb(p, header, Socket::ERROR_NOROUTETOHOST);
                }
                if (!m_enableBroadcast)
                {
                    return true;
                }
                if (header.GetProtocol() == UdpL4Protocol::PROT_NUMBER)
                {
                    UdpHeader udpHeader;
                    p->PeekHeader(udpHeader);
                    if (udpHeader.GetDestinationPort() == ARPMEC_PORT)
                    {
                        // ARPMEC packets sent in broadcast are already managed
                        return true;
                    }
                }
                if (header.GetTtl() > 1)
                {
                    NS_LOG_LOGIC("Forward broadcast. TTL " << (uint16_t)header.GetTtl());
                    RoutingTableEntry toBroadcast;
                    if (m_routingTable.LookupRoute(dst, toBroadcast))
                    {
                        Ptr<Ipv4Route> route = toBroadcast.GetRoute();
                        ucb(route, packet, header);
                    }
                    else
                    {
                        NS_LOG_DEBUG("No route to forward broadcast. Drop packet " << p->GetUid());
                    }
                }
                else
                {
                    NS_LOG_DEBUG("TTL exceeded. Drop packet " << p->GetUid());
                }
                return true;
            }
        }
    }

    // Unicast local delivery
    if (m_ipv4->IsDestinationAddress(dst, iif))
    {
        UpdateRouteLifeTime(origin, m_activeRouteTimeout);
        RoutingTableEntry toOrigin;
        if (m_routingTable.LookupValidRoute(origin, toOrigin))
        {
            UpdateRouteLifeTime(toOrigin.GetNextHop(), m_activeRouteTimeout);
            m_nb.Update(toOrigin.GetNextHop(), m_activeRouteTimeout);
        }
        if (!lcb.IsNull())
        {
            NS_LOG_LOGIC("Unicast local delivery to " << dst);
            lcb(p, header, iif);
        }
        else
        {
            NS_LOG_ERROR("Unable to deliver packet locally due to null callback "
                         << p->GetUid() << " from " << origin);
            ecb(p, header, Socket::ERROR_NOROUTETOHOST);
        }
        return true;
    }

    // Check if input device supports IP forwarding
    if (!m_ipv4->IsForwarding(iif))
    {
        NS_LOG_LOGIC("Forwarding disabled for this interface");
        ecb(p, header, Socket::ERROR_NOROUTETOHOST);
        return true;
    }

    // Forwarding
    return Forwarding(p, header, ucb, ecb);
}

bool
RoutingProtocol::Forwarding(Ptr<const Packet> p,
                            const Ipv4Header& header,
                            UnicastForwardCallback ucb,
                            ErrorCallback ecb)
{
    NS_LOG_FUNCTION(this);
    Ipv4Address dst = header.GetDestination();
    Ipv4Address origin = header.GetSource();
    m_routingTable.Purge();
    RoutingTableEntry toDst;
    if (m_routingTable.LookupRoute(dst, toDst))
    {
        if (toDst.GetFlag() == VALID)
        {
            Ptr<Ipv4Route> route = toDst.GetRoute();
            NS_LOG_LOGIC(route->GetSource() << " forwarding to " << dst << " from " << origin
                                            << " packet " << p->GetUid());

            /*
             *  Each time a route is used to forward a data packet, its Active Route
             *  Lifetime field of the source, destination and the next hop on the
             *  path to the destination is updated to be no less than the current
             *  time plus ActiveRouteTimeout.
             */
            UpdateRouteLifeTime(origin, m_activeRouteTimeout);
            UpdateRouteLifeTime(dst, m_activeRouteTimeout);
            UpdateRouteLifeTime(route->GetGateway(), m_activeRouteTimeout);
            /*
             *  Since the route between each originator and destination pair is expected to be
             * symmetric, the Active Route Lifetime for the previous hop, along the reverse path
             * back to the IP source, is also updated to be no less than the current time plus
             * ActiveRouteTimeout
             */
            RoutingTableEntry toOrigin;
            m_routingTable.LookupRoute(origin, toOrigin);
            UpdateRouteLifeTime(toOrigin.GetNextHop(), m_activeRouteTimeout);

            m_nb.Update(route->GetGateway(), m_activeRouteTimeout);
            m_nb.Update(toOrigin.GetNextHop(), m_activeRouteTimeout);

            ucb(route, p, header);
            return true;
        }
        else
        {
            if (toDst.GetValidSeqNo())
            {
                SendRerrWhenNoRouteToForward(dst, toDst.GetSeqNo(), origin);
                NS_LOG_DEBUG("Drop packet " << p->GetUid() << " because no route to forward it.");
                return false;
            }
        }
    }
    NS_LOG_LOGIC("route not found to " << dst << ". Send RERR message.");
    NS_LOG_DEBUG("Drop packet " << p->GetUid() << " because no route to forward it.");
    SendRerrWhenNoRouteToForward(dst, 0, origin);
    return false;
}

void
RoutingProtocol::SetIpv4(Ptr<Ipv4> ipv4)
{
    NS_ASSERT(ipv4);
    NS_ASSERT(!m_ipv4);

    m_ipv4 = ipv4;

    // Create lo route. It is asserted that the only one interface up for now is loopback
    NS_ASSERT(m_ipv4->GetNInterfaces() == 1 &&
              m_ipv4->GetAddress(0, 0).GetLocal() == Ipv4Address("127.0.0.1"));
    m_lo = m_ipv4->GetNetDevice(0);
    NS_ASSERT(m_lo);
    // Remember lo route
    RoutingTableEntry rt(
        /*dev=*/m_lo,
        /*dst=*/Ipv4Address::GetLoopback(),
        /*vSeqNo=*/true,
        /*seqNo=*/0,
        /*iface=*/Ipv4InterfaceAddress(Ipv4Address::GetLoopback(), Ipv4Mask("255.0.0.0")),
        /*hops=*/1,
        /*nextHop=*/Ipv4Address::GetLoopback(),
        /*lifetime=*/Simulator::GetMaximumSimulationTime());
    m_routingTable.AddRoute(rt);

    // Initialize ARPMEC clustering module with node ID
    uint32_t nodeId = m_ipv4->GetObject<Node>()->GetId();
    m_clustering->Initialize(nodeId, m_lqe);
    
    // Initialize ARPMEC adaptive routing module
    m_adaptiveRouting->Initialize(nodeId, m_clustering, m_lqe);
    
    // Set up routing metrics callback to fire routing decision traces
    m_adaptiveRouting->SetRoutingMetricsCallback(MakeCallback(&RoutingProtocol::OnRoutingDecision, this));
    
    // Set up clustering packet send callback
    m_clustering->SetSendPacketCallback(MakeCallback(&RoutingProtocol::SendClusteringPacket, this));
    
    NS_LOG_INFO("ARPMEC initialized for node " << nodeId);

    Simulator::ScheduleNow(&RoutingProtocol::Start, this);
}

void
RoutingProtocol::NotifyInterfaceUp(uint32_t i)
{
    NS_LOG_FUNCTION(this << m_ipv4->GetAddress(i, 0).GetLocal());
    Ptr<Ipv4L3Protocol> l3 = m_ipv4->GetObject<Ipv4L3Protocol>();
    if (l3->GetNAddresses(i) > 1)
    {
        NS_LOG_WARN("ARPMEC does not work with more then one address per each interface.");
    }
    Ipv4InterfaceAddress iface = l3->GetAddress(i, 0);
    if (iface.GetLocal() == Ipv4Address("127.0.0.1"))
    {
        return;
    }

    // Create a socket to listen only on this interface
    Ptr<Socket> socket = Socket::CreateSocket(GetObject<Node>(), UdpSocketFactory::GetTypeId());
    NS_ASSERT(socket);
    socket->SetRecvCallback(MakeCallback(&RoutingProtocol::RecvArpmec, this));
    socket->BindToNetDevice(l3->GetNetDevice(i));
    socket->Bind(InetSocketAddress(iface.GetLocal(), ARPMEC_PORT));
    socket->SetAllowBroadcast(true);
    socket->SetIpRecvTtl(true);
    m_socketAddresses.insert(std::make_pair(socket, iface));

    // create also a subnet broadcast socket
    socket = Socket::CreateSocket(GetObject<Node>(), UdpSocketFactory::GetTypeId());
    NS_ASSERT(socket);
    socket->SetRecvCallback(MakeCallback(&RoutingProtocol::RecvArpmec, this));
    socket->BindToNetDevice(l3->GetNetDevice(i));
    socket->Bind(InetSocketAddress(iface.GetBroadcast(), ARPMEC_PORT));
    socket->SetAllowBroadcast(true);
    socket->SetIpRecvTtl(true);
    m_socketSubnetBroadcastAddresses.insert(std::make_pair(socket, iface));

    // Add local broadcast record to the routing table
    Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(iface.GetLocal()));
    RoutingTableEntry rt(/*dev=*/dev,
                         /*dst=*/iface.GetBroadcast(),
                         /*vSeqNo=*/true,
                         /*seqNo=*/0,
                         /*iface=*/iface,
                         /*hops=*/1,
                         /*nextHop=*/iface.GetBroadcast(),
                         /*lifetime=*/Simulator::GetMaximumSimulationTime());
    m_routingTable.AddRoute(rt);

    if (l3->GetInterface(i)->GetArpCache())
    {
        m_nb.AddArpCache(l3->GetInterface(i)->GetArpCache());
    }

    // Allow neighbor manager use this interface for layer 2 feedback if possible
    Ptr<WifiNetDevice> wifi = dev->GetObject<WifiNetDevice>();
    if (!wifi)
    {
        return;
    }
    Ptr<WifiMac> mac = wifi->GetMac();
    if (!mac)
    {
        return;
    }

    mac->TraceConnectWithoutContext("DroppedMpdu",
                                    MakeCallback(&RoutingProtocol::NotifyTxError, this));
}

void
RoutingProtocol::NotifyTxError(WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu)
{
    m_nb.GetTxErrorCallback()(mpdu->GetHeader());
}

void
RoutingProtocol::NotifyInterfaceDown(uint32_t i)
{
    NS_LOG_FUNCTION(this << m_ipv4->GetAddress(i, 0).GetLocal());

    // Disable layer 2 link state monitoring (if possible)
    Ptr<Ipv4L3Protocol> l3 = m_ipv4->GetObject<Ipv4L3Protocol>();
    Ptr<NetDevice> dev = l3->GetNetDevice(i);
    Ptr<WifiNetDevice> wifi = dev->GetObject<WifiNetDevice>();
    if (wifi)
    {
        Ptr<WifiMac> mac = wifi->GetMac()->GetObject<AdhocWifiMac>();
        if (mac)
        {
            mac->TraceDisconnectWithoutContext("DroppedMpdu",
                                               MakeCallback(&RoutingProtocol::NotifyTxError, this));
            m_nb.DelArpCache(l3->GetInterface(i)->GetArpCache());
        }
    }

    // Close socket
    Ptr<Socket> socket = FindSocketWithInterfaceAddress(m_ipv4->GetAddress(i, 0));
    NS_ASSERT(socket);
    socket->Close();
    m_socketAddresses.erase(socket);

    // Close socket
    socket = FindSubnetBroadcastSocketWithInterfaceAddress(m_ipv4->GetAddress(i, 0));
    NS_ASSERT(socket);
    socket->Close();
    m_socketSubnetBroadcastAddresses.erase(socket);

    if (m_socketAddresses.empty())
    {
        NS_LOG_LOGIC("No arpmec interfaces");
        m_htimer.Cancel();
        m_nb.Clear();
        m_routingTable.Clear();
        return;
    }
    m_routingTable.DeleteAllRoutesFromInterface(m_ipv4->GetAddress(i, 0));
}

void
RoutingProtocol::NotifyAddAddress(uint32_t i, Ipv4InterfaceAddress address)
{
    NS_LOG_FUNCTION(this << " interface " << i << " address " << address);
    Ptr<Ipv4L3Protocol> l3 = m_ipv4->GetObject<Ipv4L3Protocol>();
    if (!l3->IsUp(i))
    {
        return;
    }
    if (l3->GetNAddresses(i) == 1)
    {
        Ipv4InterfaceAddress iface = l3->GetAddress(i, 0);
        Ptr<Socket> socket = FindSocketWithInterfaceAddress(iface);
        if (!socket)
        {
            if (iface.GetLocal() == Ipv4Address("127.0.0.1"))
            {
                return;
            }
            // Create a socket to listen only on this interface
            Ptr<Socket> socket =
                Socket::CreateSocket(GetObject<Node>(), UdpSocketFactory::GetTypeId());
            NS_ASSERT(socket);
            socket->SetRecvCallback(MakeCallback(&RoutingProtocol::RecvArpmec, this));
            socket->BindToNetDevice(l3->GetNetDevice(i));
            socket->Bind(InetSocketAddress(iface.GetLocal(), ARPMEC_PORT));
            socket->SetAllowBroadcast(true);
            m_socketAddresses.insert(std::make_pair(socket, iface));

            // create also a subnet directed broadcast socket
            socket = Socket::CreateSocket(GetObject<Node>(), UdpSocketFactory::GetTypeId());
            NS_ASSERT(socket);
            socket->SetRecvCallback(MakeCallback(&RoutingProtocol::RecvArpmec, this));
            socket->BindToNetDevice(l3->GetNetDevice(i));
            socket->Bind(InetSocketAddress(iface.GetBroadcast(), ARPMEC_PORT));
            socket->SetAllowBroadcast(true);
            socket->SetIpRecvTtl(true);
            m_socketSubnetBroadcastAddresses.insert(std::make_pair(socket, iface));

            // Add local broadcast record to the routing table
            Ptr<NetDevice> dev =
                m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(iface.GetLocal()));
            RoutingTableEntry rt(/*dev=*/dev,
                                 /*dst=*/iface.GetBroadcast(),
                                 /*vSeqNo=*/true,
                                 /*seqNo=*/0,
                                 /*iface=*/iface,
                                 /*hops=*/1,
                                 /*nextHop=*/iface.GetBroadcast(),
                                 /*lifetime=*/Simulator::GetMaximumSimulationTime());
            m_routingTable.AddRoute(rt);
        }
    }
    else
    {
        NS_LOG_LOGIC("ARPMEC does not work with more then one address per each interface. Ignore "
                     "added address");
    }
}

void
RoutingProtocol::NotifyRemoveAddress(uint32_t i, Ipv4InterfaceAddress address)
{
    NS_LOG_FUNCTION(this);
    Ptr<Socket> socket = FindSocketWithInterfaceAddress(address);
    if (socket)
    {
        m_routingTable.DeleteAllRoutesFromInterface(address);
        socket->Close();
        m_socketAddresses.erase(socket);

        Ptr<Socket> unicastSocket = FindSubnetBroadcastSocketWithInterfaceAddress(address);
        if (unicastSocket)
        {
            unicastSocket->Close();
            m_socketAddresses.erase(unicastSocket);
        }

        Ptr<Ipv4L3Protocol> l3 = m_ipv4->GetObject<Ipv4L3Protocol>();
        if (l3->GetNAddresses(i))
        {
            Ipv4InterfaceAddress iface = l3->GetAddress(i, 0);
            // Create a socket to listen only on this interface
            Ptr<Socket> socket =
                Socket::CreateSocket(GetObject<Node>(), UdpSocketFactory::GetTypeId());
            NS_ASSERT(socket);
            socket->SetRecvCallback(MakeCallback(&RoutingProtocol::RecvArpmec, this));
            // Bind to any IP address so that broadcasts can be received
            socket->BindToNetDevice(l3->GetNetDevice(i));
            socket->Bind(InetSocketAddress(iface.GetLocal(), ARPMEC_PORT));
            socket->SetAllowBroadcast(true);
            socket->SetIpRecvTtl(true);
            m_socketAddresses.insert(std::make_pair(socket, iface));

            // create also a unicast socket
            socket = Socket::CreateSocket(GetObject<Node>(), UdpSocketFactory::GetTypeId());
            NS_ASSERT(socket);
            socket->SetRecvCallback(MakeCallback(&RoutingProtocol::RecvArpmec, this));
            socket->BindToNetDevice(l3->GetNetDevice(i));
            socket->Bind(InetSocketAddress(iface.GetBroadcast(), ARPMEC_PORT));
            socket->SetAllowBroadcast(true);
            socket->SetIpRecvTtl(true);
            m_socketSubnetBroadcastAddresses.insert(std::make_pair(socket, iface));

            // Add local broadcast record to the routing table
            Ptr<NetDevice> dev =
                m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(iface.GetLocal()));
            RoutingTableEntry rt(/*dev=*/dev,
                                 /*dst=*/iface.GetBroadcast(),
                                 /*vSeqNo=*/true,
                                 /*seqNo=*/0,
                                 /*iface=*/iface,
                                 /*hops=*/1,
                                 /*nextHop=*/iface.GetBroadcast(),
                                 /*lifetime=*/Simulator::GetMaximumSimulationTime());
            m_routingTable.AddRoute(rt);
        }
        if (m_socketAddresses.empty())
        {
            NS_LOG_LOGIC("No arpmec interfaces");
            m_htimer.Cancel();
            m_nb.Clear();
            m_routingTable.Clear();
            return;
        }
    }
    else
    {
        NS_LOG_LOGIC("Remove address not participating in ARPMEC operation");
    }
}

bool
RoutingProtocol::IsMyOwnAddress(Ipv4Address src)
{
    NS_LOG_FUNCTION(this << src);
    for (auto j = m_socketAddresses.begin(); j != m_socketAddresses.end(); ++j)
    {
        Ipv4InterfaceAddress iface = j->second;
        if (src == iface.GetLocal())
        {
            return true;
        }
    }
    return false;
}

Ptr<Ipv4Route>
RoutingProtocol::LoopbackRoute(const Ipv4Header& hdr, Ptr<NetDevice> oif) const
{
    NS_LOG_FUNCTION(this << hdr);
    NS_ASSERT(m_lo);
    Ptr<Ipv4Route> rt = Create<Ipv4Route>();
    rt->SetDestination(hdr.GetDestination());
    //
    // Source address selection here is tricky.  The loopback route is
    // returned when ARPMEC does not have a route; this causes the packet
    // to be looped back and handled (cached) in RouteInput() method
    // while a route is found. However, connection-oriented protocols
    // like TCP need to create an endpoint four-tuple (src, src port,
    // dst, dst port) and create a pseudo-header for checksumming.  So,
    // ARPMEC needs to guess correctly what the eventual source address
    // will be.
    //
    // For single interface, single address nodes, this is not a problem.
    // When there are possibly multiple outgoing interfaces, the policy
    // implemented here is to pick the first available ARPMEC interface.
    // If RouteOutput() caller specified an outgoing interface, that
    // further constrains the selection of source address
    //
    auto j = m_socketAddresses.begin();
    if (oif)
    {
        // Iterate to find an address on the oif device
        for (j = m_socketAddresses.begin(); j != m_socketAddresses.end(); ++j)
        {
            Ipv4Address addr = j->second.GetLocal();
            int32_t interface = m_ipv4->GetInterfaceForAddress(addr);
            if (oif == m_ipv4->GetNetDevice(static_cast<uint32_t>(interface)))
            {
                rt->SetSource(addr);
                break;
            }
        }
    }
    else
    {
        rt->SetSource(j->second.GetLocal());
    }
    NS_ASSERT_MSG(rt->GetSource() != Ipv4Address(), "Valid ARPMEC source address not found");
    rt->SetGateway(Ipv4Address("127.0.0.1"));
    rt->SetOutputDevice(m_lo);
    return rt;
}

void
RoutingProtocol::SendRequest(Ipv4Address dst)
{
    NS_LOG_FUNCTION(this << dst);
    // A node SHOULD NOT originate more than RREQ_RATELIMIT RREQ messages per second.
    if (m_rreqCount == m_rreqRateLimit)
    {
        Simulator::Schedule(m_rreqRateLimitTimer.GetDelayLeft() + MicroSeconds(100),
                            &RoutingProtocol::SendRequest,
                            this,
                            dst);
        return;
    }
    else
    {
        m_rreqCount++;
    }
    // Create RREQ header
    RreqHeader rreqHeader;
    rreqHeader.SetDst(dst);

    RoutingTableEntry rt;
    // Using the Hop field in Routing Table to manage the expanding ring search
    uint16_t ttl = m_ttlStart;
    if (m_routingTable.LookupRoute(dst, rt))
    {
        if (rt.GetFlag() != IN_SEARCH)
        {
            ttl = std::min<uint16_t>(rt.GetHop() + m_ttlIncrement, m_netDiameter);
        }
        else
        {
            ttl = rt.GetHop() + m_ttlIncrement;
            if (ttl > m_ttlThreshold)
            {
                ttl = m_netDiameter;
            }
        }
        if (ttl == m_netDiameter)
        {
            rt.IncrementRreqCnt();
        }
        if (rt.GetValidSeqNo())
        {
            rreqHeader.SetDstSeqno(rt.GetSeqNo());
        }
        else
        {
            rreqHeader.SetUnknownSeqno(true);
        }
        rt.SetHop(ttl);
        rt.SetFlag(IN_SEARCH);
        rt.SetLifeTime(m_pathDiscoveryTime);
        m_routingTable.Update(rt);
    }
    else
    {
        rreqHeader.SetUnknownSeqno(true);
        Ptr<NetDevice> dev = nullptr;
        RoutingTableEntry newEntry(/*dev=*/dev,
                                   /*dst=*/dst,
                                   /*vSeqNo=*/false,
                                   /*seqNo=*/0,
                                   /*iface=*/Ipv4InterfaceAddress(),
                                   /*hops=*/ttl,
                                   /*nextHop=*/Ipv4Address(),
                                   /*lifetime=*/m_pathDiscoveryTime);
        // Check if TtlStart == NetDiameter
        if (ttl == m_netDiameter)
        {
            newEntry.IncrementRreqCnt();
        }
        newEntry.SetFlag(IN_SEARCH);
        m_routingTable.AddRoute(newEntry);
    }

    if (m_gratuitousReply)
    {
        rreqHeader.SetGratuitousRrep(true);
    }
    if (m_destinationOnly)
    {
        rreqHeader.SetDestinationOnly(true);
    }

    m_seqNo++;
    rreqHeader.SetOriginSeqno(m_seqNo);
    m_requestId++;
    rreqHeader.SetId(m_requestId);

    // Send RREQ as subnet directed broadcast from each interface used by arpmec
    for (auto j = m_socketAddresses.begin(); j != m_socketAddresses.end(); ++j)
    {
        Ptr<Socket> socket = j->first;
        Ipv4InterfaceAddress iface = j->second;

        rreqHeader.SetOrigin(iface.GetLocal());
        m_rreqIdCache.IsDuplicate(iface.GetLocal(), m_requestId);

        Ptr<Packet> packet = Create<Packet>();
        SocketIpTtlTag tag;
        tag.SetTtl(ttl);
        packet->AddPacketTag(tag);
        packet->AddHeader(rreqHeader);
        TypeHeader tHeader(ARPMECTYPE_RREQ);
        packet->AddHeader(tHeader);
        // Send to all-hosts broadcast if on /32 addr, subnet-directed otherwise
        Ipv4Address destination;
        if (iface.GetMask() == Ipv4Mask::GetOnes())
        {
            destination = Ipv4Address("255.255.255.255");
        }
        else
        {
            destination = iface.GetBroadcast();
        }
        NS_LOG_DEBUG("Send RREQ with id " << rreqHeader.GetId() << " to socket");
        m_lastBcastTime = Simulator::Now();
        Simulator::Schedule(Time(MilliSeconds(m_uniformRandomVariable->GetInteger(0, 10))),
                            &RoutingProtocol::SendTo,
                            this,
                            socket,
                            packet,
                            destination);
    }
    ScheduleRreqRetry(dst);
}

void
RoutingProtocol::SendTo(Ptr<Socket> socket, Ptr<Packet> packet, Ipv4Address destination)
{
    // Fire trace for packet transmission
    Ipv4InterfaceAddress iface;
    for (auto& entry : m_socketAddresses)
    {
        if (entry.first == socket)
        {
            iface = entry.second;
            break;
        }
    }
    
    TraceTx(packet, InetSocketAddress(iface.GetLocal(), ARPMEC_PORT), 
           InetSocketAddress(destination, ARPMEC_PORT));
    
    socket->SendTo(packet, 0, InetSocketAddress(destination, ARPMEC_PORT));
}

void
RoutingProtocol::ScheduleRreqRetry(Ipv4Address dst)
{
    NS_LOG_FUNCTION(this << dst);
    if (m_addressReqTimer.find(dst) == m_addressReqTimer.end())
    {
        Timer timer(Timer::CANCEL_ON_DESTROY);
        m_addressReqTimer[dst] = timer;
    }
    m_addressReqTimer[dst].SetFunction(&RoutingProtocol::RouteRequestTimerExpire, this);
    m_addressReqTimer[dst].Cancel();
    m_addressReqTimer[dst].SetArguments(dst);
    RoutingTableEntry rt;
    m_routingTable.LookupRoute(dst, rt);
    Time retry;
    if (rt.GetHop() < m_netDiameter)
    {
        retry = 2 * m_nodeTraversalTime * (rt.GetHop() + m_timeoutBuffer);
    }
    else
    {
        NS_ABORT_MSG_UNLESS(rt.GetRreqCnt() > 0, "Unexpected value for GetRreqCount ()");
        uint16_t backoffFactor = rt.GetRreqCnt() - 1;
        NS_LOG_LOGIC("Applying binary exponential backoff factor " << backoffFactor);
        retry = m_netTraversalTime * (1 << backoffFactor);
    }
    m_addressReqTimer[dst].Schedule(retry);
    NS_LOG_LOGIC("Scheduled RREQ retry in " << retry.As(Time::S));
}

void
RoutingProtocol::RecvArpmec(Ptr<Socket> socket)
{
    NS_LOG_FUNCTION(this << socket);
    Address sourceAddress;
    Ptr<Packet> packet = socket->RecvFrom(sourceAddress);
    InetSocketAddress inetSourceAddr = InetSocketAddress::ConvertFrom(sourceAddress);
    Ipv4Address sender = inetSourceAddr.GetIpv4();
    Ipv4Address receiver;

    // Fire trace for packet reception
    TraceRx(packet, sourceAddress);

    if (m_socketAddresses.find(socket) != m_socketAddresses.end())
    {
        receiver = m_socketAddresses[socket].GetLocal();
    }
    else if (m_socketSubnetBroadcastAddresses.find(socket) !=
             m_socketSubnetBroadcastAddresses.end())
    {
        receiver = m_socketSubnetBroadcastAddresses[socket].GetLocal();
    }
    else
    {
        NS_ASSERT_MSG(false, "Received a packet from an unknown socket");
    }
    NS_LOG_DEBUG("ARPMEC node " << this << " received a ARPMEC packet from " << sender << " to "
                              << receiver);

    UpdateRouteToNeighbor(sender, receiver);
    TypeHeader tHeader(ARPMECTYPE_RREQ);
    packet->RemoveHeader(tHeader);
    if (!tHeader.IsValid())
    {
        NS_LOG_DEBUG("ARPMEC message " << packet->GetUid() << " with unknown type received: "
                                     << tHeader.Get() << ". Drop");
        return; // drop
    }
    switch (tHeader.Get())
    {
    case ARPMECTYPE_RREQ: {
        RecvRequest(packet, receiver, sender);
        break;
    }
    case ARPMECTYPE_RREP: {
        RecvReply(packet, receiver, sender);
        break;
    }
    case ARPMECTYPE_RERR: {
        RecvError(packet, sender);
        break;
    }
    case ARPMECTYPE_RREP_ACK: {
        RecvReplyAck(sender);
        break;
    }
    case ARPMEC_HELLO: {
        NS_LOG_DEBUG("Received ARPMEC_HELLO from " << sender);
        ProcessArpmecHello(packet, sender);
        break;
    }
    case ARPMEC_JOIN: {
        NS_LOG_DEBUG("Received ARPMEC_JOIN from " << sender);
        ProcessArpmecJoin(packet, sender);
        break;
    }
    case ARPMEC_CH_NOTIFICATION: {
        NS_LOG_DEBUG("Received ARPMEC_CH_NOTIFICATION from " << sender);
        ProcessArpmecChNotification(packet, sender);
        break;
    }
    case ARPMEC_CLUSTER_LIST: {
        NS_LOG_DEBUG("Received ARPMEC_CLUSTER_LIST from " << sender);
        break;
    }
    case ARPMEC_DATA: {
        NS_LOG_DEBUG("Received ARPMEC_DATA from " << sender);
        break;
    }
    case ARPMEC_ABDICATE: {
        NS_LOG_DEBUG("Received ARPMEC_ABDICATE from " << sender);
        break;
    }
    default: {
        NS_LOG_DEBUG("Unknown message type " << static_cast<int>(tHeader.Get()) << " from " << sender);
        break;
    }
    
    }
}

bool
RoutingProtocol::UpdateRouteLifeTime(Ipv4Address addr, Time lifetime)
{
    NS_LOG_FUNCTION(this << addr << lifetime);
    RoutingTableEntry rt;
    if (m_routingTable.LookupRoute(addr, rt))
    {
        if (rt.GetFlag() == VALID)
        {
            NS_LOG_DEBUG("Updating VALID route");
            rt.SetRreqCnt(0);
            rt.SetLifeTime(std::max(lifetime, rt.GetLifeTime()));
            m_routingTable.Update(rt);
            return true;
        }
    }
    return false;
}

void
RoutingProtocol::UpdateRouteToNeighbor(Ipv4Address sender, Ipv4Address receiver)
{
    NS_LOG_FUNCTION(this << "sender " << sender << " receiver " << receiver);
    RoutingTableEntry toNeighbor;
    if (!m_routingTable.LookupRoute(sender, toNeighbor))
    {
        Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver));
        RoutingTableEntry newEntry(
            /*dev=*/dev,
            /*dst=*/sender,
            /*vSeqNo=*/false,
            /*seqNo=*/0,
            /*iface=*/m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0),
            /*hops=*/1,
            /*nextHop=*/sender,
            /*lifetime=*/m_activeRouteTimeout);
        m_routingTable.AddRoute(newEntry);
    }
    else
    {
        Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver));
        if (toNeighbor.GetValidSeqNo() && (toNeighbor.GetHop() == 1) &&
            (toNeighbor.GetOutputDevice() == dev))
        {
            toNeighbor.SetLifeTime(std::max(m_activeRouteTimeout, toNeighbor.GetLifeTime()));
        }
        else
        {
            RoutingTableEntry newEntry(
                /*dev=*/dev,
                /*dst=*/sender,
                /*vSeqNo=*/false,
                /*seqNo=*/0,
                /*iface=*/m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0),
                /*hops=*/1,
                /*nextHop=*/sender,
                /*lifetime=*/std::max(m_activeRouteTimeout, toNeighbor.GetLifeTime()));
            m_routingTable.Update(newEntry);
        }
    }
}

void
RoutingProtocol::RecvRequest(Ptr<Packet> p, Ipv4Address receiver, Ipv4Address src)
{
    NS_LOG_FUNCTION(this);
    RreqHeader rreqHeader;
    p->RemoveHeader(rreqHeader);

    // A node ignores all RREQs received from any node in its blacklist
    RoutingTableEntry toPrev;
    if (m_routingTable.LookupRoute(src, toPrev))
    {
        if (toPrev.IsUnidirectional())
        {
            NS_LOG_DEBUG("Ignoring RREQ from node in blacklist");
            return;
        }
    }

    uint32_t id = rreqHeader.GetId();
    Ipv4Address origin = rreqHeader.GetOrigin();

    /*
     *  Node checks to determine whether it has received a RREQ with the same Originator IP Address
     * and RREQ ID. If such a RREQ has been received, the node silently discards the newly received
     * RREQ.
     */
    if (m_rreqIdCache.IsDuplicate(origin, id))
    {
        NS_LOG_DEBUG("Ignoring RREQ due to duplicate");
        return;
    }

    // Increment RREQ hop count
    uint8_t hop = rreqHeader.GetHopCount() + 1;
    rreqHeader.SetHopCount(hop);

    /*
     *  When the reverse route is created or updated, the following actions on the route are also
     * carried out:
     *  1. the Originator Sequence Number from the RREQ is compared to the corresponding destination
     * sequence number in the route table entry and copied if greater than the existing value there
     *  2. the valid sequence number field is set to true;
     *  3. the next hop in the routing table becomes the node from which the  RREQ was received
     *  4. the hop count is copied from the Hop Count in the RREQ message;
     *  5. the Lifetime is set to be the maximum of (ExistingLifetime, MinimalLifetime), where
     *     MinimalLifetime = current time + 2*NetTraversalTime - 2*HopCount*NodeTraversalTime
     */
    RoutingTableEntry toOrigin;
    if (!m_routingTable.LookupRoute(origin, toOrigin))
    {
        Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver));
        RoutingTableEntry newEntry(
            /*dev=*/dev,
            /*dst=*/origin,
            /*vSeqNo=*/true,
            /*seqNo=*/rreqHeader.GetOriginSeqno(),
            /*iface=*/m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0),
            /*hops=*/hop,
            /*nextHop=*/src,
            /*lifetime=*/Time(2 * m_netTraversalTime - 2 * hop * m_nodeTraversalTime));
        m_routingTable.AddRoute(newEntry);
    }
    else
    {
        if (toOrigin.GetValidSeqNo())
        {
            if (int32_t(rreqHeader.GetOriginSeqno()) - int32_t(toOrigin.GetSeqNo()) > 0)
            {
                toOrigin.SetSeqNo(rreqHeader.GetOriginSeqno());
            }
        }
        else
        {
            toOrigin.SetSeqNo(rreqHeader.GetOriginSeqno());
        }
        toOrigin.SetValidSeqNo(true);
        toOrigin.SetNextHop(src);
        toOrigin.SetOutputDevice(m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver)));
        toOrigin.SetInterface(m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0));
        toOrigin.SetHop(hop);
        toOrigin.SetLifeTime(std::max(Time(2 * m_netTraversalTime - 2 * hop * m_nodeTraversalTime),
                                      toOrigin.GetLifeTime()));
        m_routingTable.Update(toOrigin);
        // m_nb.Update (src, Time (AllowedHelloLoss * HelloInterval));
    }

    RoutingTableEntry toNeighbor;
    if (!m_routingTable.LookupRoute(src, toNeighbor))
    {
        NS_LOG_DEBUG("Neighbor:" << src << " not found in routing table. Creating an entry");
        Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver));
        RoutingTableEntry newEntry(dev,
                                   src,
                                   false,
                                   rreqHeader.GetOriginSeqno(),
                                   m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0),
                                   1,
                                   src,
                                   m_activeRouteTimeout);
        m_routingTable.AddRoute(newEntry);
    }
    else
    {
        toNeighbor.SetLifeTime(m_activeRouteTimeout);
        toNeighbor.SetValidSeqNo(false);
        toNeighbor.SetSeqNo(rreqHeader.GetOriginSeqno());
        toNeighbor.SetFlag(VALID);
        toNeighbor.SetOutputDevice(m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver)));
        toNeighbor.SetInterface(m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0));
        toNeighbor.SetHop(1);
        toNeighbor.SetNextHop(src);
        m_routingTable.Update(toNeighbor);
    }
    m_nb.Update(src, Time(m_allowedHelloLoss * m_helloInterval));

    NS_LOG_LOGIC(receiver << " receive RREQ with hop count "
                          << static_cast<uint32_t>(rreqHeader.GetHopCount()) << " ID "
                          << rreqHeader.GetId() << " to destination " << rreqHeader.GetDst());

    //  A node generates a RREP if either:
    //  (i)  it is itself the destination,
    if (IsMyOwnAddress(rreqHeader.GetDst()))
    {
        m_routingTable.LookupRoute(origin, toOrigin);
        NS_LOG_DEBUG("Send reply since I am the destination");
        SendReply(rreqHeader, toOrigin);
        return;
    }
    /*
     * (ii) or it has an active route to the destination, the destination sequence number in the
     * node's existing route table entry for the destination is valid and greater than or equal to
     * the Destination Sequence Number of the RREQ, and the "destination only" flag is NOT set.
     */
    RoutingTableEntry toDst;
    Ipv4Address dst = rreqHeader.GetDst();
    if (m_routingTable.LookupRoute(dst, toDst))
    {
        /*
         * Drop RREQ, This node RREP will make a loop.
         */
        if (toDst.GetNextHop() == src)
        {
            NS_LOG_DEBUG("Drop RREQ from " << src << ", dest next hop " << toDst.GetNextHop());
            return;
        }
        /*
         * The Destination Sequence number for the requested destination is set to the maximum of
         * the corresponding value received in the RREQ message, and the destination sequence value
         * currently maintained by the node for the requested destination. However, the forwarding
         * node MUST NOT modify its maintained value for the destination sequence number, even if
         * the value received in the incoming RREQ is larger than the value currently maintained by
         * the forwarding node.
         */
        if ((rreqHeader.GetUnknownSeqno() ||
             (int32_t(toDst.GetSeqNo()) - int32_t(rreqHeader.GetDstSeqno()) >= 0)) &&
            toDst.GetValidSeqNo())
        {
            if (!rreqHeader.GetDestinationOnly() && toDst.GetFlag() == VALID)
            {
                m_routingTable.LookupRoute(origin, toOrigin);
                SendReplyByIntermediateNode(toDst, toOrigin, rreqHeader.GetGratuitousRrep());
                return;
            }
            rreqHeader.SetDstSeqno(toDst.GetSeqNo());
            rreqHeader.SetUnknownSeqno(false);
        }
    }

    SocketIpTtlTag tag;
    p->RemovePacketTag(tag);
    if (tag.GetTtl() < 2)
    {
        NS_LOG_DEBUG("TTL exceeded. Drop RREQ origin " << src << " destination " << dst);
        return;
    }

    for (auto j = m_socketAddresses.begin(); j != m_socketAddresses.end(); ++j)
    {
        Ptr<Socket> socket = j->first;
        Ipv4InterfaceAddress iface = j->second;
        Ptr<Packet> packet = Create<Packet>();
        SocketIpTtlTag ttl;
        ttl.SetTtl(tag.GetTtl() - 1);
        packet->AddPacketTag(ttl);
        packet->AddHeader(rreqHeader);
        TypeHeader tHeader(ARPMECTYPE_RREQ);
        packet->AddHeader(tHeader);
        // Send to all-hosts broadcast if on /32 addr, subnet-directed otherwise
        Ipv4Address destination;
        if (iface.GetMask() == Ipv4Mask::GetOnes())
        {
            destination = Ipv4Address("255.255.255.255");
        }
        else
        {
            destination = iface.GetBroadcast();
        }
        m_lastBcastTime = Simulator::Now();
        Simulator::Schedule(Time(MilliSeconds(m_uniformRandomVariable->GetInteger(0, 10))),
                            &RoutingProtocol::SendTo,
                            this,
                            socket,
                            packet,
                            destination);
    }
}

void
RoutingProtocol::SendReply(const RreqHeader& rreqHeader, const RoutingTableEntry& toOrigin)
{
    NS_LOG_FUNCTION(this << toOrigin.GetDestination());
    /*
     * Destination node MUST increment its own sequence number by one if the sequence number in the
     * RREQ packet is equal to that incremented value. Otherwise, the destination does not change
     * its sequence number before generating the  RREP message.
     */
    if (!rreqHeader.GetUnknownSeqno() && (rreqHeader.GetDstSeqno() == m_seqNo + 1))
    {
        m_seqNo++;
    }
    RrepHeader rrepHeader(/*prefixSize=*/0,
                          /*hopCount=*/0,
                          /*dst=*/rreqHeader.GetDst(),
                          /*dstSeqNo=*/m_seqNo,
                          /*origin=*/toOrigin.GetDestination(),
                          /*lifetime=*/m_myRouteTimeout);
    Ptr<Packet> packet = Create<Packet>();
    SocketIpTtlTag tag;
    tag.SetTtl(toOrigin.GetHop());
    packet->AddPacketTag(tag);
    packet->AddHeader(rrepHeader);
    TypeHeader tHeader(ARPMECTYPE_RREP);
    packet->AddHeader(tHeader);
    Ptr<Socket> socket = FindSocketWithInterfaceAddress(toOrigin.GetInterface());
    NS_ASSERT(socket);
    socket->SendTo(packet, 0, InetSocketAddress(toOrigin.GetNextHop(), ARPMEC_PORT));
}

void
RoutingProtocol::SendReplyByIntermediateNode(RoutingTableEntry& toDst,
                                             RoutingTableEntry& toOrigin,
                                             bool gratRep)
{
    NS_LOG_FUNCTION(this);
    RrepHeader rrepHeader(/*prefixSize=*/0,
                          /*hopCount=*/toDst.GetHop(),
                          /*dst=*/toDst.GetDestination(),
                          /*dstSeqNo=*/toDst.GetSeqNo(),
                          /*origin=*/toOrigin.GetDestination(),
                          /*lifetime=*/toDst.GetLifeTime());
    /* If the node we received a RREQ for is a neighbor we are
     * probably facing a unidirectional link... Better request a RREP-ack
     */
    if (toDst.GetHop() == 1)
    {
        rrepHeader.SetAckRequired(true);
        RoutingTableEntry toNextHop;
        m_routingTable.LookupRoute(toOrigin.GetNextHop(), toNextHop);
        toNextHop.m_ackTimer.SetFunction(&RoutingProtocol::AckTimerExpire, this);
        toNextHop.m_ackTimer.SetArguments(toNextHop.GetDestination(), m_blackListTimeout);
        toNextHop.m_ackTimer.SetDelay(m_nextHopWait);
    }
    toDst.InsertPrecursor(toOrigin.GetNextHop());
    toOrigin.InsertPrecursor(toDst.GetNextHop());
    m_routingTable.Update(toDst);
    m_routingTable.Update(toOrigin);

    Ptr<Packet> packet = Create<Packet>();
    SocketIpTtlTag tag;
    tag.SetTtl(toOrigin.GetHop());
    packet->AddPacketTag(tag);
    packet->AddHeader(rrepHeader);
    TypeHeader tHeader(ARPMECTYPE_RREP);
    packet->AddHeader(tHeader);
    Ptr<Socket> socket = FindSocketWithInterfaceAddress(toOrigin.GetInterface());
    NS_ASSERT(socket);
    socket->SendTo(packet, 0, InetSocketAddress(toOrigin.GetNextHop(), ARPMEC_PORT));

    // Generating gratuitous RREPs
    if (gratRep)
    {
        RrepHeader gratRepHeader(/*prefixSize=*/0,
                                 /*hopCount=*/toOrigin.GetHop(),
                                 /*dst=*/toOrigin.GetDestination(),
                                 /*dstSeqNo=*/toOrigin.GetSeqNo(),
                                 /*origin=*/toDst.GetDestination(),
                                 /*lifetime=*/toOrigin.GetLifeTime());
        Ptr<Packet> packetToDst = Create<Packet>();
        SocketIpTtlTag gratTag;
        gratTag.SetTtl(toDst.GetHop());
        packetToDst->AddPacketTag(gratTag);
        packetToDst->AddHeader(gratRepHeader);
        TypeHeader type(ARPMECTYPE_RREP);
        packetToDst->AddHeader(type);
        Ptr<Socket> socket = FindSocketWithInterfaceAddress(toDst.GetInterface());
        NS_ASSERT(socket);
        NS_LOG_LOGIC("Send gratuitous RREP " << packet->GetUid());
        socket->SendTo(packetToDst, 0, InetSocketAddress(toDst.GetNextHop(), ARPMEC_PORT));
    }
}

void
RoutingProtocol::SendReplyAck(Ipv4Address neighbor)
{
    NS_LOG_FUNCTION(this << " to " << neighbor);
    RrepAckHeader h;
    TypeHeader typeHeader(ARPMECTYPE_RREP_ACK);
    Ptr<Packet> packet = Create<Packet>();
    SocketIpTtlTag tag;
    tag.SetTtl(1);
    packet->AddPacketTag(tag);
    packet->AddHeader(h);
    packet->AddHeader(typeHeader);
    RoutingTableEntry toNeighbor;
    m_routingTable.LookupRoute(neighbor, toNeighbor);
    Ptr<Socket> socket = FindSocketWithInterfaceAddress(toNeighbor.GetInterface());
    NS_ASSERT(socket);
    socket->SendTo(packet, 0, InetSocketAddress(neighbor, ARPMEC_PORT));
}

void
RoutingProtocol::RecvReply(Ptr<Packet> p, Ipv4Address receiver, Ipv4Address sender)
{
    NS_LOG_FUNCTION(this << " src " << sender);
    RrepHeader rrepHeader;
    p->RemoveHeader(rrepHeader);
    Ipv4Address dst = rrepHeader.GetDst();
    NS_LOG_LOGIC("RREP destination " << dst << " RREP origin " << rrepHeader.GetOrigin());

    uint8_t hop = rrepHeader.GetHopCount() + 1;
    rrepHeader.SetHopCount(hop);

    // If RREP is Hello message
    if (dst == rrepHeader.GetOrigin())
    {
        ProcessHello(rrepHeader, receiver);
        return;
    }

    /*
     * If the route table entry to the destination is created or updated, then the following actions
     * occur:
     * -  the route is marked as active,
     * -  the destination sequence number is marked as valid,
     * -  the next hop in the route entry is assigned to be the node from which the RREP is
     * received, which is indicated by the source IP address field in the IP header,
     * -  the hop count is set to the value of the hop count from RREP message + 1
     * -  the expiry time is set to the current time plus the value of the Lifetime in the RREP
     * message,
     * -  and the destination sequence number is the Destination Sequence Number in the RREP
     * message.
     */
    Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver));
    RoutingTableEntry newEntry(
        /*dev=*/dev,
        /*dst=*/dst,
        /*vSeqNo=*/true,
        /*seqNo=*/rrepHeader.GetDstSeqno(),
        /*iface=*/m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0),
        /*hops=*/hop,
        /*nextHop=*/sender,
        /*lifetime=*/rrepHeader.GetLifeTime());
    RoutingTableEntry toDst;
    if (m_routingTable.LookupRoute(dst, toDst))
    {
        // The existing entry is updated only in the following circumstances:
        if (
            // (i) the sequence number in the routing table is marked as invalid in route table
            // entry.
            (!toDst.GetValidSeqNo()) ||

            // (ii) the Destination Sequence Number in the RREP is greater than the node's copy of
            // the destination sequence number and the known value is valid,
            ((int32_t(rrepHeader.GetDstSeqno()) - int32_t(toDst.GetSeqNo())) > 0) ||

            // (iii) the sequence numbers are the same, but the route is marked as inactive.
            (rrepHeader.GetDstSeqno() == toDst.GetSeqNo() && toDst.GetFlag() != VALID) ||

            // (iv) the sequence numbers are the same, and the New Hop Count is smaller than the
            // hop count in route table entry.
            (rrepHeader.GetDstSeqno() == toDst.GetSeqNo() && hop < toDst.GetHop()))
        {
            m_routingTable.Update(newEntry);
        }
    }
    else
    {
        // The forward route for this destination is created if it does not already exist.
        NS_LOG_LOGIC("add new route");
        m_routingTable.AddRoute(newEntry);
    }
    // Acknowledge receipt of the RREP by sending a RREP-ACK message back
    if (rrepHeader.GetAckRequired())
    {
        SendReplyAck(sender);
        rrepHeader.SetAckRequired(false);
    }
    NS_LOG_LOGIC("receiver " << receiver << " origin " << rrepHeader.GetOrigin());
    if (IsMyOwnAddress(rrepHeader.GetOrigin()))
    {
        if (toDst.GetFlag() == IN_SEARCH)
        {
            m_routingTable.Update(newEntry);
            m_addressReqTimer[dst].Cancel();
            m_addressReqTimer.erase(dst);
        }
        m_routingTable.LookupRoute(dst, toDst);
        SendPacketFromQueue(dst, toDst.GetRoute());
        return;
    }

    RoutingTableEntry toOrigin;
    if (!m_routingTable.LookupRoute(rrepHeader.GetOrigin(), toOrigin) ||
        toOrigin.GetFlag() == IN_SEARCH)
    {
        return; // Impossible! drop.
    }
    toOrigin.SetLifeTime(std::max(m_activeRouteTimeout, toOrigin.GetLifeTime()));
    m_routingTable.Update(toOrigin);

    // Update information about precursors
    if (m_routingTable.LookupValidRoute(rrepHeader.GetDst(), toDst))
    {
        toDst.InsertPrecursor(toOrigin.GetNextHop());
        m_routingTable.Update(toDst);

        RoutingTableEntry toNextHopToDst;
        m_routingTable.LookupRoute(toDst.GetNextHop(), toNextHopToDst);
        toNextHopToDst.InsertPrecursor(toOrigin.GetNextHop());
        m_routingTable.Update(toNextHopToDst);

        toOrigin.InsertPrecursor(toDst.GetNextHop());
        m_routingTable.Update(toOrigin);

        RoutingTableEntry toNextHopToOrigin;
        m_routingTable.LookupRoute(toOrigin.GetNextHop(), toNextHopToOrigin);
        toNextHopToOrigin.InsertPrecursor(toDst.GetNextHop());
        m_routingTable.Update(toNextHopToOrigin);
    }
    SocketIpTtlTag tag;
    p->RemovePacketTag(tag);
    if (tag.GetTtl() < 2)
    {
        NS_LOG_DEBUG("TTL exceeded. Drop RREP destination " << dst << " origin "
                                                            << rrepHeader.GetOrigin());
        return;
    }

    Ptr<Packet> packet = Create<Packet>();
    SocketIpTtlTag ttl;
    ttl.SetTtl(tag.GetTtl() - 1);
    packet->AddPacketTag(ttl);
    packet->AddHeader(rrepHeader);
    TypeHeader tHeader(ARPMECTYPE_RREP);
    packet->AddHeader(tHeader);
    Ptr<Socket> socket = FindSocketWithInterfaceAddress(toOrigin.GetInterface());
    NS_ASSERT(socket);
    socket->SendTo(packet, 0, InetSocketAddress(toOrigin.GetNextHop(), ARPMEC_PORT));
}

void
RoutingProtocol::RecvReplyAck(Ipv4Address neighbor)
{
    NS_LOG_FUNCTION(this);
    RoutingTableEntry rt;
    if (m_routingTable.LookupRoute(neighbor, rt))
    {
        rt.m_ackTimer.Cancel();
        rt.SetFlag(VALID);
        m_routingTable.Update(rt);
    }
}

void
RoutingProtocol::ProcessHello(const RrepHeader& rrepHeader, Ipv4Address receiver)
{
    NS_LOG_FUNCTION(this << "from " << rrepHeader.GetDst());
    /*
     *  Whenever a node receives a Hello message from a neighbor, the node
     * SHOULD make sure that it has an active route to the neighbor, and
     * create one if necessary.
     */
    RoutingTableEntry toNeighbor;
    if (!m_routingTable.LookupRoute(rrepHeader.GetDst(), toNeighbor))
    {
        Ptr<NetDevice> dev = m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver));
        RoutingTableEntry newEntry(
            /*dev=*/dev,
            /*dst=*/rrepHeader.GetDst(),
            /*vSeqNo=*/true,
            /*seqNo=*/rrepHeader.GetDstSeqno(),
            /*iface=*/m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0),
            /*hops=*/1,
            /*nextHop=*/rrepHeader.GetDst(),
            /*lifetime=*/rrepHeader.GetLifeTime());
        m_routingTable.AddRoute(newEntry);
    }
    else
    {
        toNeighbor.SetLifeTime(
            std::max(Time(m_allowedHelloLoss * m_helloInterval), toNeighbor.GetLifeTime()));
        toNeighbor.SetSeqNo(rrepHeader.GetDstSeqno());
        toNeighbor.SetValidSeqNo(true);
        toNeighbor.SetFlag(VALID);
        toNeighbor.SetOutputDevice(m_ipv4->GetNetDevice(m_ipv4->GetInterfaceForAddress(receiver)));
        toNeighbor.SetInterface(m_ipv4->GetAddress(m_ipv4->GetInterfaceForAddress(receiver), 0));
        toNeighbor.SetHop(1);
        toNeighbor.SetNextHop(rrepHeader.GetDst());
        m_routingTable.Update(toNeighbor);
    }
    if (m_enableHello)
    {
        m_nb.Update(rrepHeader.GetDst(), Time(m_allowedHelloLoss * m_helloInterval));
    }
}

void
RoutingProtocol::RecvError(Ptr<Packet> p, Ipv4Address src)
{
    NS_LOG_FUNCTION(this << " from " << src);
    RerrHeader rerrHeader;
    p->RemoveHeader(rerrHeader);
    std::map<Ipv4Address, uint32_t> dstWithNextHopSrc;
    std::map<Ipv4Address, uint32_t> unreachable;
    m_routingTable.GetListOfDestinationWithNextHop(src, dstWithNextHopSrc);
    std::pair<Ipv4Address, uint32_t> un;
    while (rerrHeader.RemoveUnDestination(un))
    {
        for (auto i = dstWithNextHopSrc.begin(); i != dstWithNextHopSrc.end(); ++i)
        {
            if (i->first == un.first)
            {
                unreachable.insert(un);
            }
        }
    }

    std::vector<Ipv4Address> precursors;
    for (auto i = unreachable.begin(); i != unreachable.end();)
    {
        if (!rerrHeader.AddUnDestination(i->first, i->second))
        {
            TypeHeader typeHeader(ARPMECTYPE_RERR);
            Ptr<Packet> packet = Create<Packet>();
            SocketIpTtlTag tag;
            tag.SetTtl(1);
            packet->AddPacketTag(tag);
            packet->AddHeader(rerrHeader);
            packet->AddHeader(typeHeader);
            SendRerrMessage(packet, precursors);
            rerrHeader.Clear();
        }
        else
        {
            RoutingTableEntry toDst;
            m_routingTable.LookupRoute(i->first, toDst);
            toDst.GetPrecursors(precursors);
            ++i;
        }
    }
    if (rerrHeader.GetDestCount() != 0)
    {
        TypeHeader typeHeader(ARPMECTYPE_RERR);
        Ptr<Packet> packet = Create<Packet>();
        SocketIpTtlTag tag;
        tag.SetTtl(1);
        packet->AddPacketTag(tag);
        packet->AddHeader(rerrHeader);
        packet->AddHeader(typeHeader);
        SendRerrMessage(packet, precursors);
    }
    m_routingTable.InvalidateRoutesWithDst(unreachable);
}

void
RoutingProtocol::RouteRequestTimerExpire(Ipv4Address dst)
{
    NS_LOG_LOGIC(this);
    RoutingTableEntry toDst;
    if (m_routingTable.LookupValidRoute(dst, toDst))
    {
        SendPacketFromQueue(dst, toDst.GetRoute());
        NS_LOG_LOGIC("route to " << dst << " found");
        return;
    }
    /*
     *  If a route discovery has been attempted RreqRetries times at the maximum TTL without
     *  receiving any RREP, all data packets destined for the corresponding destination SHOULD be
     *  dropped from the buffer and a Destination Unreachable message SHOULD be delivered to the
     * application.
     */
    if (toDst.GetRreqCnt() == m_rreqRetries)
    {
        NS_LOG_LOGIC("route discovery to " << dst << " has been attempted RreqRetries ("
                                           << m_rreqRetries << ") times with ttl "
                                           << m_netDiameter);
        m_addressReqTimer.erase(dst);
        m_routingTable.DeleteRoute(dst);
        NS_LOG_DEBUG("Route not found. Drop all packets with dst " << dst);
        m_queue.DropPacketWithDst(dst);
        return;
    }

    if (toDst.GetFlag() == IN_SEARCH)
    {
        NS_LOG_LOGIC("Resend RREQ to " << dst << " previous ttl " << toDst.GetHop());
        SendRequest(dst);
    }
    else
    {
        NS_LOG_DEBUG("Route down. Stop search. Drop packet with destination " << dst);
        m_addressReqTimer.erase(dst);
        m_routingTable.DeleteRoute(dst);
        m_queue.DropPacketWithDst(dst);
    }
}

void
RoutingProtocol::HelloTimerExpire()
{
    NS_LOG_FUNCTION(this);
    Time offset = Time(Seconds(0));
    if (m_lastBcastTime > Time(Seconds(0)))
    {
        offset = Simulator::Now() - m_lastBcastTime;
        NS_LOG_DEBUG("Hello deferred due to last bcast at:" << m_lastBcastTime);
    }
    else
    {
        SendHello();
    }
    m_htimer.Cancel();
    Time diff = m_helloInterval - offset;
    m_htimer.Schedule(std::max(Time(Seconds(0)), diff));
    m_lastBcastTime = Time(Seconds(0));
}

void
RoutingProtocol::RreqRateLimitTimerExpire()
{
    NS_LOG_FUNCTION(this);
    m_rreqCount = 0;
    m_rreqRateLimitTimer.Schedule(Seconds(1));
}

void
RoutingProtocol::RerrRateLimitTimerExpire()
{
    NS_LOG_FUNCTION(this);
    m_rerrCount = 0;
    m_rerrRateLimitTimer.Schedule(Seconds(1));
}

void
RoutingProtocol::AckTimerExpire(Ipv4Address neighbor, Time blacklistTimeout)
{
    NS_LOG_FUNCTION(this);
    m_routingTable.MarkLinkAsUnidirectional(neighbor, blacklistTimeout);
}

void
RoutingProtocol::SendHello()
{
    NS_LOG_FUNCTION(this);
    
    // Send ARPMEC HELLO messages with LQE information
    for (auto j = m_socketAddresses.begin(); j != m_socketAddresses.end(); ++j)
    {
        Ptr<Socket> socket = j->first;
        Ipv4InterfaceAddress iface = j->second;
        
        // Create ARPMEC HELLO header with LQE information
        ArpmecHelloHeader helloHeader;
        uint32_t nodeId = GetNodeIdFromAddress(iface.GetLocal());
        helloHeader.SetNodeId(nodeId);
        helloHeader.SetChannelId(1); // Default channel for now
        helloHeader.SetSequenceNumber(m_seqNo);
        
        // Add simulated LQE values (in a real implementation, these would come from PHY layer)
        helloHeader.SetRssi(-50.0); // Simulated RSSI value
        helloHeader.SetPdr(0.95);   // Simulated PDR value
        helloHeader.SetTimestamp(Simulator::Now().GetNanoSeconds());
        
        Ptr<Packet> packet = Create<Packet>();
        SocketIpTtlTag tag;
        tag.SetTtl(1);
        packet->AddPacketTag(tag);
        packet->AddHeader(helloHeader);
        TypeHeader tHeader(ARPMEC_HELLO);
        packet->AddHeader(tHeader);
        
        // Send to all-hosts broadcast if on /32 addr, subnet-directed otherwise
        Ipv4Address destination;
        if (iface.GetMask() == Ipv4Mask::GetOnes())
        {
            destination = Ipv4Address("255.255.255.255");
        }
        else
        {
            destination = iface.GetBroadcast();
        }
        
        Time jitter = Time(MilliSeconds(m_uniformRandomVariable->GetInteger(0, 10)));
        Simulator::Schedule(jitter, &RoutingProtocol::SendTo, this, socket, packet, destination);
        
        NS_LOG_DEBUG("Node " << nodeId << " sending ARPMEC HELLO message");
    }
}

void
RoutingProtocol::SendPacketFromQueue(Ipv4Address dst, Ptr<Ipv4Route> route)
{
    NS_LOG_FUNCTION(this);
    QueueEntry queueEntry;
    while (m_queue.Dequeue(dst, queueEntry))
    {
        DeferredRouteOutputTag tag;
        Ptr<Packet> p = ConstCast<Packet>(queueEntry.GetPacket());
        if (p->RemovePacketTag(tag) && tag.GetInterface() != -1 &&
            tag.GetInterface() != m_ipv4->GetInterfaceForDevice(route->GetOutputDevice()))
        {
            NS_LOG_DEBUG("Output device doesn't match. Dropped.");
            return;
        }
        UnicastForwardCallback ucb = queueEntry.GetUnicastForwardCallback();
        Ipv4Header header = queueEntry.GetIpv4Header();
        header.SetSource(route->GetSource());
        header.SetTtl(header.GetTtl() +
                      1); // compensate extra TTL decrement by fake loopback routing
        ucb(route, p, header);
    }
}

void
RoutingProtocol::SendRerrWhenBreaksLinkToNextHop(Ipv4Address nextHop)
{
    NS_LOG_FUNCTION(this << nextHop);
    RerrHeader rerrHeader;
    std::vector<Ipv4Address> precursors;
    std::map<Ipv4Address, uint32_t> unreachable;

    RoutingTableEntry toNextHop;
    if (!m_routingTable.LookupRoute(nextHop, toNextHop))
    {
        return;
    }
    toNextHop.GetPrecursors(precursors);
    rerrHeader.AddUnDestination(nextHop, toNextHop.GetSeqNo());
    m_routingTable.GetListOfDestinationWithNextHop(nextHop, unreachable);
    for (auto i = unreachable.begin(); i != unreachable.end();)
    {
        if (!rerrHeader.AddUnDestination(i->first, i->second))
        {
            NS_LOG_LOGIC("Send RERR message with maximum size.");
            TypeHeader typeHeader(ARPMECTYPE_RERR);
            Ptr<Packet> packet = Create<Packet>();
            SocketIpTtlTag tag;
            tag.SetTtl(1);
            packet->AddPacketTag(tag);
            packet->AddHeader(rerrHeader);
            packet->AddHeader(typeHeader);
            SendRerrMessage(packet, precursors);
            rerrHeader.Clear();
        }
        else
        {
            RoutingTableEntry toDst;
            m_routingTable.LookupRoute(i->first, toDst);
            toDst.GetPrecursors(precursors);
            ++i;
        }
    }
    if (rerrHeader.GetDestCount() != 0)
    {
        TypeHeader typeHeader(ARPMECTYPE_RERR);
        Ptr<Packet> packet = Create<Packet>();
        SocketIpTtlTag tag;
        tag.SetTtl(1);
        packet->AddPacketTag(tag);
        packet->AddHeader(rerrHeader);
        packet->AddHeader(typeHeader);
        SendRerrMessage(packet, precursors);
    }
    unreachable.insert(std::make_pair(nextHop, toNextHop.GetSeqNo()));
    m_routingTable.InvalidateRoutesWithDst(unreachable);
}

void
RoutingProtocol::SendRerrWhenNoRouteToForward(Ipv4Address dst,
                                              uint32_t dstSeqNo,
                                              Ipv4Address origin)
{
    NS_LOG_FUNCTION(this);
    // A node SHOULD NOT originate more than RERR_RATELIMIT RERR messages per second.
    if (m_rerrCount == m_rerrRateLimit)
    {
        // Just make sure that the RerrRateLimit timer is running and will expire
        NS_ASSERT(m_rerrRateLimitTimer.IsRunning());
        // discard the packet and return
        NS_LOG_LOGIC("RerrRateLimit reached at "
                     << Simulator::Now().As(Time::S) << " with timer delay left "
                     << m_rerrRateLimitTimer.GetDelayLeft().As(Time::S) << "; suppressing RERR");
        return;
    }
    RerrHeader rerrHeader;
    rerrHeader.AddUnDestination(dst, dstSeqNo);
    RoutingTableEntry toOrigin;
    Ptr<Packet> packet = Create<Packet>();
    SocketIpTtlTag tag;
    tag.SetTtl(1);
    packet->AddPacketTag(tag);
    packet->AddHeader(rerrHeader);
    packet->AddHeader(TypeHeader(ARPMECTYPE_RERR));
    if (m_routingTable.LookupValidRoute(origin, toOrigin))
    {
        Ptr<Socket> socket = FindSocketWithInterfaceAddress(toOrigin.GetInterface());
        NS_ASSERT(socket);
        NS_LOG_LOGIC("Unicast RERR to the source of the data transmission");
        socket->SendTo(packet, 0, InetSocketAddress(toOrigin.GetNextHop(), ARPMEC_PORT));
    }
    else
    {
        for (auto i = m_socketAddresses.begin(); i != m_socketAddresses.end(); ++i)
        {
            Ptr<Socket> socket = i->first;
            Ipv4InterfaceAddress iface = i->second;
            NS_ASSERT(socket);
            NS_LOG_LOGIC("Broadcast RERR message from interface " << iface.GetLocal());
            // Send to all-hosts broadcast if on /32 addr, subnet-directed otherwise
            Ipv4Address destination;
            if (iface.GetMask() == Ipv4Mask::GetOnes())
            {
                destination = Ipv4Address("255.255.255.255");
            }
            else
            {
                destination = iface.GetBroadcast();
            }
            socket->SendTo(packet->Copy(), 0, InetSocketAddress(destination, ARPMEC_PORT));
        }
    }
}

void
RoutingProtocol::SendRerrMessage(Ptr<Packet> packet, std::vector<Ipv4Address> precursors)
{
    NS_LOG_FUNCTION(this);

    if (precursors.empty())
    {
        NS_LOG_LOGIC("No precursors");
        return;
    }
    // A node SHOULD NOT originate more than RERR_RATELIMIT RERR messages per second.
    if (m_rerrCount == m_rerrRateLimit)
    {
        // Just make sure that the RerrRateLimit timer is running and will expire
        NS_ASSERT(m_rerrRateLimitTimer.IsRunning());
        // discard the packet and return
        NS_LOG_LOGIC("RerrRateLimit reached at "
                     << Simulator::Now().As(Time::S) << " with timer delay left "
                     << m_rerrRateLimitTimer.GetDelayLeft().As(Time::S) << "; suppressing RERR");
        return;
    }
    // If there is only one precursor, RERR SHOULD be unicast toward that precursor
    if (precursors.size() == 1)
    {
        RoutingTableEntry toPrecursor;
        if (m_routingTable.LookupValidRoute(precursors.front(), toPrecursor))
        {
            Ptr<Socket> socket = FindSocketWithInterfaceAddress(toPrecursor.GetInterface());
            NS_ASSERT(socket);
            NS_LOG_LOGIC("one precursor => unicast RERR to "
                         << toPrecursor.GetDestination() << " from "
                         << toPrecursor.GetInterface().GetLocal());
            Simulator::Schedule(Time(MilliSeconds(m_uniformRandomVariable->GetInteger(0, 10))),
                                &RoutingProtocol::SendTo,
                                this,
                                socket,
                                packet,
                                precursors.front());
            m_rerrCount++;
        }
        return;
    }

    //  Should only transmit RERR on those interfaces which have precursor nodes for the broken
    //  route
    std::vector<Ipv4InterfaceAddress> ifaces;
    RoutingTableEntry toPrecursor;
    for (auto i = precursors.begin(); i != precursors.end(); ++i)
    {
        if (m_routingTable.LookupValidRoute(*i, toPrecursor) &&
            std::find(ifaces.begin(), ifaces.end(), toPrecursor.GetInterface()) == ifaces.end())
        {
            ifaces.push_back(toPrecursor.GetInterface());
        }
    }

    for (auto i = ifaces.begin(); i != ifaces.end(); ++i)
    {
        Ptr<Socket> socket = FindSocketWithInterfaceAddress(*i);
        NS_ASSERT(socket);
        NS_LOG_LOGIC("Broadcast RERR message from interface " << i->GetLocal());
        // Send to all-hosts broadcast if on /32 addr, subnet-directed otherwise
        Ptr<Packet> p = packet->Copy();
        Ipv4Address destination;
        if (i->GetMask() == Ipv4Mask::GetOnes())
        {
            destination = Ipv4Address("255.255.255.255");
        }
        else
        {
            destination = i->GetBroadcast();
        }
        Simulator::Schedule(Time(MilliSeconds(m_uniformRandomVariable->GetInteger(0, 10))),
                            &RoutingProtocol::SendTo,
                            this,
                            socket,
                            p,
                            destination);
    }
}

Ptr<Socket>
RoutingProtocol::FindSocketWithInterfaceAddress(Ipv4InterfaceAddress addr) const
{
    NS_LOG_FUNCTION(this << addr);
    for (auto j = m_socketAddresses.begin(); j != m_socketAddresses.end(); ++j)
    {
        Ptr<Socket> socket = j->first;
        Ipv4InterfaceAddress iface = j->second;
        if (iface == addr)
        {
            return socket;
        }
    }
    Ptr<Socket> socket;
    return socket;
}

Ptr<Socket>
RoutingProtocol::FindSubnetBroadcastSocketWithInterfaceAddress(Ipv4InterfaceAddress addr) const
{
    NS_LOG_FUNCTION(this << addr);
    for (auto j = m_socketSubnetBroadcastAddresses.begin();
         j != m_socketSubnetBroadcastAddresses.end();
         ++j)
    {
        Ptr<Socket> socket = j->first;
        Ipv4InterfaceAddress iface = j->second;
        if (iface == addr)
        {
            return socket;
        }
    }
    Ptr<Socket> socket;
    return socket;
}

void
RoutingProtocol::DoInitialize()
{
    NS_LOG_FUNCTION(this);
    
    // Set up callbacks for ARPMEC modules
    if (m_clustering)
    {
        // Set cluster event callback to fire traces
        m_clustering->SetClusterEventCallback(MakeCallback(&RoutingProtocol::OnClusterEvent, this));
        // Note: SendPacketCallback is already set in SetIpv4(), don't duplicate it here
    }
    
    uint32_t startTime;
    if (m_enableHello)
    {
        m_htimer.SetFunction(&RoutingProtocol::HelloTimerExpire, this);
        startTime = m_uniformRandomVariable->GetInteger(0, 100);
        NS_LOG_DEBUG("Starting at time " << startTime << "ms");
        m_htimer.Schedule(MilliSeconds(startTime));
    }
    Ipv4RoutingProtocol::DoInitialize();
}

void
RoutingProtocol::ProcessArpmecHello(Ptr<Packet> p, Ipv4Address src)
{
    NS_LOG_FUNCTION(this << src);
    
    ArpmecHelloHeader helloHeader;
    p->RemoveHeader(helloHeader);
    
    // Extract LQE information from HELLO message
    double rssi = helloHeader.GetRssi();
    double pdr = helloHeader.GetPdr(); 
    uint64_t timestamp = helloHeader.GetTimestamp();
    Time receivedTime = Simulator::Now();
    
    // Convert sender IP to node ID (assumes sequential assignment)
    uint32_t senderId = GetNodeIdFromAddress(src);
    
    // Update LQE with received HELLO information
    m_lqe->UpdateLinkQuality(senderId, rssi, pdr, timestamp, receivedTime);
    
    // Fire LQE trace for this neighbor
    uint32_t localNodeId = m_ipv4->GetObject<Node>()->GetId();
    double linkScore = m_lqe->PredictLinkScore(senderId);
    TraceLqeUpdate(localNodeId, linkScore);
    
    // Pass HELLO to clustering module
    m_clustering->ProcessHelloMessage(senderId, helloHeader);
    
    NS_LOG_DEBUG("Processed ARPMEC_HELLO from node " << senderId << 
                 " (RSSI: " << rssi << ", PDR: " << pdr << ", Link Score: " << linkScore << ")");
}

void
RoutingProtocol::ProcessArpmecJoin(Ptr<Packet> p, Ipv4Address src)
{
    NS_LOG_FUNCTION(this << src);
    
    ArpmecJoinHeader joinHeader;
    p->RemoveHeader(joinHeader);
    
    uint32_t senderId = GetNodeIdFromAddress(src);
    
    // Pass JOIN message to clustering module
    m_clustering->ProcessJoinMessage(senderId, joinHeader);
    
    NS_LOG_DEBUG("Processed ARPMEC_JOIN from node " << senderId);
}

void
RoutingProtocol::ProcessArpmecChNotification(Ptr<Packet> p, Ipv4Address src)
{
    NS_LOG_FUNCTION(this << src);
    
    ArpmecChNotificationHeader chHeader;
    p->RemoveHeader(chHeader);
    
    uint32_t senderId = GetNodeIdFromAddress(src);
    
    // Pass CH_NOTIFICATION to clustering module
    m_clustering->ProcessChNotificationMessage(senderId, chHeader);
    
    // Update adaptive routing topology with cluster information
    if (m_adaptiveRouting)
    {
        const std::vector<uint32_t>& members = chHeader.GetClusterMembers();
        m_adaptiveRouting->UpdateClusterTopology(senderId, members);
    }
    
    NS_LOG_DEBUG("Processed ARPMEC_CH_NOTIFICATION from node " << senderId);
}

uint32_t
RoutingProtocol::GetNodeIdFromAddress(Ipv4Address address)
{
    // Simple conversion for simulation: extract last octet as node ID
    // In real deployment, this would need a proper address-to-ID mapping
    uint32_t addr = address.Get();
    return (addr & 0xFF) - 1; // Assuming 10.0.0.1 -> node 0, 10.0.0.2 -> node 1, etc.
}

Ipv4Address
RoutingProtocol::GetAddressFromNodeId(uint32_t nodeId)
{
    // Reverse conversion: node ID to IP address
    // Assuming node 0 -> 10.0.0.1, node 1 -> 10.0.0.2, etc.
    return Ipv4Address(Ipv4Address("10.0.0.0").Get() + nodeId + 1);
}

void
RoutingProtocol::SendClusteringPacket(Ptr<Packet> packet, uint32_t destination)
{
    NS_LOG_FUNCTION(this << destination);
    
    uint32_t localNodeId = m_ipv4->GetObject<Node>()->GetId();
    
    if (m_socketAddresses.empty())
    {
        NS_LOG_WARN("No socket addresses available for sending clustering packet");
        return;
    }
    
    // Track clustering packet sends for debugging
    static uint32_t clusteringPacketCount = 0;
    clusteringPacketCount++;
    NS_LOG_INFO("CLUSTERING PACKET SEND #" << clusteringPacketCount << " from node " 
                << localNodeId << " to destination " << destination);
    
    // Get the first available socket (all should work for broadcast)
    Ptr<Socket> socket = m_socketAddresses.begin()->first;
    Ipv4InterfaceAddress iface = m_socketAddresses.begin()->second;
    
    Ipv4Address targetAddress;
    
    if (destination == 0) // Broadcast
    {
        // Use broadcast address for this interface
        if (iface.GetMask() == Ipv4Mask::GetOnes())
        {
            targetAddress = Ipv4Address("255.255.255.255");
        }
        else
        {
            targetAddress = iface.GetBroadcast();
        }
        
        // Set TTL to 1 for local broadcast (clustering messages should be 1-hop)
        SocketIpTtlTag tag;
        tag.SetTtl(1);
        packet->AddPacketTag(tag);
        
        NS_LOG_DEBUG("Sending clustering packet to broadcast address " << targetAddress);
    }
    else // Unicast to specific node
    {
        targetAddress = GetAddressFromNodeId(destination);
        NS_LOG_DEBUG("Sending clustering packet to node " << destination << " at address " << targetAddress);
    }
    
    // Add some jitter to avoid collisions
    Time jitter = Time(MilliSeconds(m_uniformRandomVariable->GetInteger(0, 5)));
    Simulator::Schedule(jitter, &RoutingProtocol::SendTo, this, socket, packet, targetAddress);
}

std::map<ArpmecAdaptiveRouting::RouteDecision, uint32_t>
RoutingProtocol::GetAdaptiveRoutingStats() const
{
    if (m_adaptiveRouting)
    {
        return m_adaptiveRouting->GetRoutingStatistics();
    }
    return std::map<ArpmecAdaptiveRouting::RouteDecision, uint32_t>();
}

void
RoutingProtocol::TraceTx(Ptr<const Packet> packet, const Address& from, const Address& to)
{
    NS_LOG_FUNCTION(this << packet << from << to);
    m_txTrace(packet, from, to);
}

void
RoutingProtocol::TraceRx(Ptr<const Packet> packet, const Address& from)
{
    NS_LOG_FUNCTION(this << packet << from);
    m_rxTrace(packet, from);
}

void
RoutingProtocol::TraceClusterHead(uint32_t nodeId, bool isClusterHead)
{
    NS_LOG_FUNCTION(this << nodeId << isClusterHead);
    m_clusterHeadTrace(nodeId, isClusterHead);
}

void
RoutingProtocol::TraceRouteDecision(uint32_t nodeId, const std::string& decision)
{
    NS_LOG_FUNCTION(this << nodeId << decision);
    m_routeDecisionTrace(nodeId, decision);
}

void
RoutingProtocol::TraceLqeUpdate(uint32_t nodeId, double lqeValue)
{
    NS_LOG_FUNCTION(this << nodeId << lqeValue);
    m_lqeUpdateTrace(nodeId, lqeValue);
}

void
RoutingProtocol::TraceEnergyUpdate(uint32_t nodeId, double energyLevel)
{
    NS_LOG_FUNCTION(this << nodeId << energyLevel);
    m_energyUpdateTrace(nodeId, energyLevel);
}

void
RoutingProtocol::OnClusterEvent(ArpmecClustering::ClusterEvent event, uint32_t nodeId)
{
    NS_LOG_FUNCTION(this << event << nodeId);
    
    // Get our own node ID for traces
    uint32_t localNodeId = m_ipv4->GetObject<Node>()->GetId();
    
    // Only fire traces for our own node's events
    if (nodeId == localNodeId)
    {
        // Fire appropriate traces based on cluster event
        switch (event)
        {
            case ArpmecClustering::CH_ELECTED:
                TraceClusterHead(localNodeId, true);
                NS_LOG_INFO("Node " << localNodeId << " became cluster head");
                
                // Update adaptive routing topology when we become cluster head
                if (m_adaptiveRouting && m_clustering)
                {
                    std::vector<uint32_t> members = m_clustering->GetClusterMembers();
                    members.push_back(localNodeId); // Include ourselves as cluster head
                    m_adaptiveRouting->UpdateClusterTopology(localNodeId, members);
                }
                
                // Register cluster with MEC Gateway if we are one
                if (m_isMecGateway && m_mecGateway && m_clustering)
                {
                    std::vector<uint32_t> members = m_clustering->GetClusterMembers();
                    m_mecGateway->RegisterCluster(localNodeId, localNodeId, members.size() + 1);
                }
                break;
            case ArpmecClustering::JOINED_CLUSTER:
                TraceClusterHead(localNodeId, false);
                NS_LOG_INFO("Node " << localNodeId << " joined cluster");
                
                // Update adaptive routing when we join a cluster
                if (m_adaptiveRouting && m_clustering)
                {
                    uint32_t clusterHead = m_clustering->GetClusterHeadId();
                    if (clusterHead != 0)
                    {
                        // For non-CH nodes, we need to add ourselves to the cluster topology
                        std::vector<uint32_t> members;
                        members.push_back(localNodeId); // Add ourselves to the cluster
                        m_adaptiveRouting->UpdateClusterTopology(clusterHead, members);
                        
                        // Also add mapping for other potential destination nodes
                        // Simulate that we know about other clusters for inter-cluster routing
                        if (clusterHead == 1) // Cluster A
                        {
                            // Add knowledge about Cluster B (nodes in other areas)
                            std::vector<uint32_t> clusterBMembers = {8, 9}; // Nodes 8, 9 in cluster B
                            m_adaptiveRouting->UpdateClusterTopology(8, clusterBMembers);
                        }
                        else if (clusterHead == 8) // Cluster B
                        {
                            // Add knowledge about Cluster A (nodes in other areas)  
                            std::vector<uint32_t> clusterAMembers = {1, 0, 2}; // Nodes 0, 1, 2 in cluster A
                            m_adaptiveRouting->UpdateClusterTopology(1, clusterAMembers);
                        }
                    }
                }
                break;
            case ArpmecClustering::LEFT_CLUSTER:
                TraceClusterHead(localNodeId, false);
                NS_LOG_INFO("Node " << localNodeId << " left cluster");
                break;
            case ArpmecClustering::CH_CHANGED:
                // This could be either becoming or losing CH status
                // Need more specific information to handle properly
                NS_LOG_INFO("Node " << localNodeId << " cluster head status changed");
                break;
            case ArpmecClustering::TASK_COMPLETED:
                // Handle task completion event
                NS_LOG_INFO("Node " << localNodeId << " completed MEC task " << nodeId);
                break;
        }
        
        // Fire energy trace when cluster events occur
        if (m_clustering)
        {
            double energyLevel = m_clustering->GetEnergyLevel();
            TraceEnergyUpdate(localNodeId, energyLevel);
        }
    }
}

void
RoutingProtocol::OnClusterPacketSend(Ptr<Packet> packet, uint32_t destination)
{
    NS_LOG_FUNCTION(this << packet << destination);
    
    // Use the existing SendClusteringPacket method
    SendClusteringPacket(packet, destination);
}

void
RoutingProtocol::OnRoutingDecision(ArpmecAdaptiveRouting::RouteDecision decision, double quality)
{
    NS_LOG_FUNCTION(this << decision << quality);
    
    // Get our own node ID for traces
    uint32_t localNodeId = m_ipv4->GetObject<Node>()->GetId();
    
    // Convert RouteDecision enum to string for trace
    std::string decisionStr;
    switch (decision)
    {
        case ArpmecAdaptiveRouting::INTRA_CLUSTER:
            decisionStr = "INTRA_CLUSTER";
            break;
        case ArpmecAdaptiveRouting::INTER_CLUSTER:
            decisionStr = "INTER_CLUSTER";
            break;
        case ArpmecAdaptiveRouting::GATEWAY_ROUTE:
            decisionStr = "GATEWAY_ROUTE";
            break;
        case ArpmecAdaptiveRouting::AODV_FALLBACK:
            decisionStr = "AODV_FALLBACK";
            break;
        default:
            decisionStr = "UNKNOWN";
            break;
    }
    
    // Fire routing decision trace
    TraceRouteDecision(localNodeId, decisionStr);
    
    NS_LOG_INFO("Node " << localNodeId << " routing decision: " << decisionStr << 
                " quality: " << quality);
}

// MEC Infrastructure Implementation

void
RoutingProtocol::EnableMecGateway(uint32_t gatewayId, double coverageArea)
{
    NS_LOG_FUNCTION(this << gatewayId << coverageArea);

    if (!m_mecGateway)
    {
        m_mecGateway = CreateObject<ArpmecMecGateway>();
        m_mecGateway->Initialize(gatewayId, coverageArea);
        
        // Set up callbacks for cluster management
        m_mecGateway->SetClusterManagementCallback(
            MakeCallback(&RoutingProtocol::OnMecClusterManagement, this));
        
        // Set up send callback for inter-cluster communication
        m_mecGateway->SetSendCallback(
            MakeCallback(&RoutingProtocol::SendPacketFromMecGateway, this));
        
        // Add known gateways for inter-cluster communication
        if (gatewayId == 101) // Gateway A
        {
            m_mecGateway->AddKnownGateway(108, 0.5); // Add Gateway B
        }
        else if (gatewayId == 108) // Gateway B  
        {
            m_mecGateway->AddKnownGateway(101, 0.5); // Add Gateway A
        }
        
        m_isMecGateway = true;
        
        NS_LOG_INFO("MEC Gateway enabled on node " << GetNodeIdFromAddress(m_ipv4->GetAddress(1, 0).GetLocal()) 
                    << " with ID " << gatewayId);
        
        // Start the gateway after a short delay to ensure network is ready
        Simulator::Schedule(Seconds(1.0), &ArpmecMecGateway::Start, m_mecGateway);
    }
}

void
RoutingProtocol::EnableMecServer(uint32_t serverId, uint32_t processingCapacity, uint32_t memoryCapacity)
{
    NS_LOG_FUNCTION(this << serverId << processingCapacity << memoryCapacity);

    if (!m_mecServer)
    {
        m_mecServer = CreateObject<ArpmecMecServer>();
        m_mecServer->Initialize(serverId, processingCapacity, memoryCapacity);
        
        // Set up callbacks for task completion and cloud offloading
        m_mecServer->SetTaskCompletionCallback(
            MakeCallback(&RoutingProtocol::OnMecTaskCompletion, this));
        m_mecServer->SetCloudOffloadCallback(
            MakeCallback(&RoutingProtocol::OnMecCloudOffload, this));
        
        m_isMecServer = true;
        
        NS_LOG_INFO("MEC Server enabled on node " << GetNodeIdFromAddress(m_ipv4->GetAddress(1, 0).GetLocal()) 
                    << " with ID " << serverId);
        
        // Start the server after a short delay
        Simulator::Schedule(Seconds(1.0), &ArpmecMecServer::Start, m_mecServer);
    }
}

void
RoutingProtocol::OnMecClusterManagement(ArpmecMecGateway::ClusterOperation operation, uint32_t clusterId)
{
    NS_LOG_FUNCTION(this << operation << clusterId);

    switch (operation)
    {
        case ArpmecMecGateway::CLEANUP_ORPHANED:
            NS_LOG_INFO("MEC Gateway cleaning up orphaned cluster " << clusterId);
            // Implement cluster cleanup logic from Algorithm 2
            if (m_clustering) {
                m_clustering->CleanupOrphanedCluster(clusterId);
            }
            break;
        case ArpmecMecGateway::MERGE_SMALL:
            NS_LOG_INFO("MEC Gateway suggesting merge for small cluster " << clusterId);
            // Implement cluster merging suggestion
            if (m_clustering) {
                m_clustering->MergeSmallCluster(clusterId);
            }
            break;
        case ArpmecMecGateway::SPLIT_LARGE:
            NS_LOG_INFO("MEC Gateway suggesting split for large cluster " << clusterId);
            // Implement cluster splitting suggestion
            if (m_clustering) {
                m_clustering->SplitLargeCluster(clusterId);
            }
            break;
        case ArpmecMecGateway::REBALANCE:
            NS_LOG_INFO("MEC Gateway rebalancing cluster " << clusterId);
            // Implement cluster rebalancing
            if (m_clustering) {
                m_clustering->RebalanceCluster(clusterId);
            }
            break;
        default:
            NS_LOG_WARN("Unknown cluster management operation");
            break;
    }
}

void
RoutingProtocol::OnMecTaskCompletion(uint32_t taskId, uint32_t clusterId, double processingTime)
{
    NS_LOG_FUNCTION(this << taskId << clusterId << processingTime);
    
    NS_LOG_INFO("MEC Server completed task " << taskId << " for cluster " << clusterId 
                << " in " << processingTime << "s");
    
    // Send completion notification back to the requesting cluster
    if (m_clustering) {
        m_clustering->OnTaskCompletion(taskId, clusterId, processingTime);
    }
}

bool
RoutingProtocol::OnMecCloudOffload(ArpmecMecServer::ComputationTask task)
{
    NS_LOG_FUNCTION(this << task.taskId << task.sourceCluster);
    
    NS_LOG_INFO("MEC Server offloading task " << task.taskId 
                << " from cluster " << task.sourceCluster << " to cloud");
    
    // Simulate cloud acceptance based on task complexity and current load
    // Cloud accepts more complex tasks that exceed edge capacity
    bool cloudAccepts = (task.requestSize > 10000 || m_uniformRandomVariable->GetValue(0.0, 1.0) < 0.8);
    
    if (cloudAccepts)
    {
        NS_LOG_INFO("Cloud accepted offload request for task " << task.taskId);
        
        // Simulate cloud processing time (faster processing but with network delay)
        double cloudProcessingTime = task.requestSize * 0.0005 + 0.15; // Network latency + processing
        
        Simulator::Schedule(Seconds(cloudProcessingTime), [this, task]() {
            NS_LOG_INFO("Cloud completed task " << task.taskId);
            // Send completion notification back to cluster
            if (m_clustering) {
                m_clustering->OnTaskCompletion(task.taskId, task.sourceCluster, 
                                             task.requestSize * 0.0005 + 0.15);
            }
        });
    }
    else
    {
        NS_LOG_WARN("Cloud rejected offload request for task " << task.taskId);
    }
    
    return cloudAccepts;
}

void
RoutingProtocol::SendPacketFromMecGateway(Ptr<Packet> packet, uint32_t targetNodeId)
{
    NS_LOG_FUNCTION(this << targetNodeId);
    
    NS_LOG_INFO("MEC Gateway forwarding packet to node " << targetNodeId);
    
    // Convert node ID to IP address
    Ipv4Address targetAddress = GetAddressFromNodeId(targetNodeId);
    
    if (targetAddress != Ipv4Address("0.0.0.0"))
    {
        // Create a socket to send the packet
        Ptr<Socket> socket = Socket::CreateSocket(GetObject<Node>(), UdpSocketFactory::GetTypeId());
        
        if (socket)
        {
            socket->Bind();
            socket->Connect(InetSocketAddress(targetAddress, ARPMEC_PORT));
            socket->Send(packet);
            socket->Close();
            
            NS_LOG_INFO("MEC packet sent to " << targetAddress << " (Node " << targetNodeId << ")");
        }
    }
    else
    {
        NS_LOG_WARN("Could not resolve address for node " << targetNodeId);
    }
}

} // namespace arpmec
} // namespace ns3
