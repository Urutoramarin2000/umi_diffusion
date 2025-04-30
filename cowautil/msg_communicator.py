import zmq

class MsgPublisher:
    def __init__(self, port) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f'tcp://*:{port}')
        print(f"Publisher bound to port {port}")  # 添加调试信息

    def send(self, topic:str, msg:dict):
        # print(f"Sending message on topic '{topic}' with content: {msg}")  # 添加调试信息
        self.socket.send_string(topic, flags=zmq.SNDMORE)
        self.socket.send_json(msg)
        # print(f"after Sending message on topic '{topic}' with content: {msg}")  # 添加调试信息


class MsgSubscriber:
    def __init__(self, ip, port, topic) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f'tcp://{ip}:{port}')
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)


    def recv(self):
        topic = self.socket.recv_string()
        msg = self.socket.recv_json()
        return msg