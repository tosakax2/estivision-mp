from pythonosc.udp_client import SimpleUDPClient


class OSCSender:

    def __init__(self, ip="127.0.0.1", port=39539):
        self.client = SimpleUDPClient(ip, port)


    def send_tracker(self, tracker_results):
        for result in tracker_results:
            self.client.send_message(
                f"/VMC/Ext/Tra/Pos",
                [
                    result.name,
                    float(result.position[0]),
                    float(result.position[1]),
                    float(result.position[2]),
                    float(result.rotation[0]),
                    float(result.rotation[1]),
                    float(result.rotation[2]),
                    float(result.rotation[3]),
                ],
            )
