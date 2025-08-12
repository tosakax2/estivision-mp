from pythonosc.udp_client import SimpleUDPClient


# Unityの座標系と同じ、左利き座標系、+yが上
# 位置(x, y, z)は 1.0f = 1m
# 回転(z, y, x)はクォータニオン、単位は度(degree)、回転値はワールド座標系
# VRChatでサポートされている部位は、腰・胸・肘×2・膝×2・足×2の8か所
# 1から8までのどの番号がどの部位になるのかは決まっておらず、VRChat内でキャリブレーションした際にOSCにより入力された座標のIKポインターとVRChat上のアバターのトラッカーに追従する部位が自動的に対応付けられる仕様なので、どの番号にどの部位を割り当てるかは送信元のアプリケーションの任意


class VRChatOscTrackerSender:

    def __init__(self, ip="127.0.0.1", port=9000):
        self.client = SimpleUDPClient(ip, port)


    def send_tracker(self, tracker_index, position, rotation):
        self.client.send_message(
            f"/tracking/trackers/{tracker_index}/position",
            [float(position[0]), float(position[1]), float(position[2])]
        )
        self.client.send_message(
            f"/tracking/trackers/{tracker_index}/rotation",
            [float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])]
        )
