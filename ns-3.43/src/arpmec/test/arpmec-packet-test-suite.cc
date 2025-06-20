#include "ns3/test.h"
#include "ns3/arpmec-packet.h"
#include "ns3/packet.h"
#include <sstream>
#include <cmath> // For std::abs in LQE field testing

namespace ns3 {
namespace arpmec {

class ArpmecPacketTest : public TestCase
{
public:
    ArpmecPacketTest() : TestCase("ARPMEC Packet Test") {}
    void DoRun() override
    {
        // Test pour ArpmecHelloHeader with LQE fields
        ArpmecHelloHeader hello;
        hello.SetNodeId(123);
        hello.SetChannelId(5);
        hello.SetSequenceNumber(42);
        hello.SetRssi(-65.5);        // RSSI in dBm
        hello.SetPdr(0.95);          // PDR value (95%)
        hello.SetTimestamp(1000000); // Timestamp in microseconds
        
        Ptr<Packet> packet = Create<Packet>();
        packet->AddHeader(hello);
        ArpmecHelloHeader helloDeserialized;
        packet->RemoveHeader(helloDeserialized);
        
        NS_TEST_ASSERT_MSG_EQ(helloDeserialized.GetNodeId(), 123, "Node ID mismatch in HELLO");
        NS_TEST_ASSERT_MSG_EQ(helloDeserialized.GetChannelId(), 5, "Channel ID mismatch in HELLO");
        NS_TEST_ASSERT_MSG_EQ(helloDeserialized.GetSequenceNumber(), 42, "Sequence number mismatch in HELLO");
        
        // Test RSSI with tolerance
        double rssiDiff = std::abs(helloDeserialized.GetRssi() - (-65.5));
        NS_TEST_ASSERT_MSG_LT(rssiDiff, 0.001, "RSSI mismatch in HELLO");
        
        // Test PDR with tolerance  
        double pdrDiff = std::abs(helloDeserialized.GetPdr() - 0.95);
        NS_TEST_ASSERT_MSG_LT(pdrDiff, 0.001, "PDR mismatch in HELLO");
        
        NS_TEST_ASSERT_MSG_EQ(helloDeserialized.GetTimestamp(), 1000000, "Timestamp mismatch in HELLO");

        // Test pour ArpmecJoinHeader
        ArpmecJoinHeader join;
        join.SetNodeId(456);
        join.SetChId(789);
        packet = Create<Packet>();
        packet->AddHeader(join);
        ArpmecJoinHeader joinDeserialized;
        packet->RemoveHeader(joinDeserialized);
        NS_TEST_ASSERT_MSG_EQ(joinDeserialized.GetNodeId(), 456, "Node ID mismatch in JOIN");
        NS_TEST_ASSERT_MSG_EQ(joinDeserialized.GetChId(), 789, "CH ID mismatch in JOIN");

        // Test pour ArpmecChNotificationHeader
        ArpmecChNotificationHeader chNotif;
        chNotif.SetChId(101);
        std::vector<uint32_t> members = {102, 103, 104};
        chNotif.SetClusterMembers(members);
        packet = Create<Packet>();
        packet->AddHeader(chNotif);
        ArpmecChNotificationHeader chNotifDeserialized;
        packet->RemoveHeader(chNotifDeserialized);
        NS_TEST_ASSERT_MSG_EQ(chNotifDeserialized.GetChId(), 101, "CH ID mismatch in CH_NOTIFICATION");
        const auto& deserializedMembers = chNotifDeserialized.GetClusterMembers();
        NS_TEST_ASSERT_MSG_EQ(deserializedMembers.size(), members.size(), "Number of members mismatch in CH_NOTIFICATION");
        for (size_t j = 0; j < members.size(); ++j)
        {
            NS_TEST_ASSERT_MSG_EQ(deserializedMembers[j], members[j], "Member mismatch at index " + std::to_string(j));
        }

        // Test pour ArpmecClusterListHeader
        ArpmecClusterListHeader clusterList;
        std::vector<std::pair<uint32_t, std::vector<uint32_t>>> clusters;
        clusters.emplace_back(201, std::vector<uint32_t>{202, 203});
        clusters.emplace_back(204, std::vector<uint32_t>{205});
        clusterList.SetClusters(clusters);
        packet = Create<Packet>();
        packet->AddHeader(clusterList);
        ArpmecClusterListHeader clusterListDeserialized;
        packet->RemoveHeader(clusterListDeserialized);
        const auto& deserializedClusters = clusterListDeserialized.GetClusters();
        NS_TEST_ASSERT_MSG_EQ(deserializedClusters.size(), clusters.size(), "Number of clusters mismatch in CLUSTER_LIST");
        for (size_t j = 0; j < clusters.size(); ++j)
        {
            NS_TEST_ASSERT_MSG_EQ(deserializedClusters[j].first, clusters[j].first, "Cluster CH ID mismatch at index " + std::to_string(j));
            NS_TEST_ASSERT_MSG_EQ(deserializedClusters[j].second.size(), clusters[j].second.size(), "Cluster members size mismatch at index " + std::to_string(j));
            for (size_t k = 0; k < clusters[j].second.size(); ++k)
            {
                NS_TEST_ASSERT_MSG_EQ(deserializedClusters[j].second[k], clusters[j].second[k], "Cluster member mismatch at index " + std::to_string(j) + ", " + std::to_string(k));
            }
        }

        // Test pour ArpmecDataHeader
        ArpmecDataHeader data;
        data.SetSourceId(301);
        data.SetDestId(302);
        data.SetChId(303);
        packet = Create<Packet>();
        packet->AddHeader(data);
        ArpmecDataHeader dataDeserialized;
        packet->RemoveHeader(dataDeserialized);
        NS_TEST_ASSERT_MSG_EQ(dataDeserialized.GetSourceId(), 301, "Source ID mismatch in DATA");
        NS_TEST_ASSERT_MSG_EQ(dataDeserialized.GetDestId(), 302, "Destination ID mismatch in DATA");
        NS_TEST_ASSERT_MSG_EQ(dataDeserialized.GetChId(), 303, "CH ID mismatch in DATA");

        // Test pour ArpmecAbdicateHeader
        ArpmecAbdicateHeader abdicate;
        abdicate.SetChId(401);
        packet = Create<Packet>();
        packet->AddHeader(abdicate);
        ArpmecAbdicateHeader abdicateDeserialized;
        packet->RemoveHeader(abdicateDeserialized);
        NS_TEST_ASSERT_MSG_EQ(abdicateDeserialized.GetChId(), 401, "CH ID mismatch in ABDICATE");

        // Test pour TypeHeader
        TypeHeader typeHeader(ARPMEC_HELLO);
        packet = Create<Packet>();
        packet->AddHeader(typeHeader);
        TypeHeader typeHeaderDeserialized;
        packet->RemoveHeader(typeHeaderDeserialized);
        NS_TEST_ASSERT_MSG_EQ(typeHeaderDeserialized.Get(), ARPMEC_HELLO, "TypeHeader ARPMEC_HELLO mismatch");
        NS_TEST_ASSERT_MSG_EQ(typeHeaderDeserialized.IsValid(), true, "TypeHeader ARPMEC_HELLO invalid");
    }
};

class ArpmecPacketTestSuite : public TestSuite
{
public:
    ArpmecPacketTestSuite() : TestSuite("arpmec-packet-test-suite", Type::UNIT)
    {
        AddTestCase(new ArpmecPacketTest, TestCase::Duration::QUICK);
    }
};

static ArpmecPacketTestSuite g_arpmecPacketTestSuite;

} // namespace arpmec
} // namespace ns3