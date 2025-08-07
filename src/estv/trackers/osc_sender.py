from pythonosc.udp_client import SimpleUDPClient


class VRChatOscTrackerSender:

    def __init__(self, ip="127.0.0.1", port=9000):
        self.client = SimpleUDPClient(ip, port)


    def send_tracker(self, tracker_index, position, rotation):
        # 位置(x, y, z)はメートル単位、回転(x, y, z)はクォータニオンで送信
        # tracker_indexは1～8（1=腰, 2=左足, 3=右足, ...など）
        self.client.send_message(
            f"/tracking/trackers/{tracker_index}/position",
            [float(position[0]), float(position[1]), float(position[2])]
        )
        self.client.send_message(
            f"/tracking/trackers/{tracker_index}/rotation",
            [float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])]
        )
