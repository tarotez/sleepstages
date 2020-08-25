import time
from connect_laser_device import connect_laser_device

class ConnectionCheckClient:
    # def __init__(self, host, port):
    def __init__(self):

        # 出力デバイスとの接続
        # self.connected2serialClient = False
        # print('in __init__ of classifierClient, self.connected2serialClient = False')
        self.serialClient, self.connected2serialClient = connect_laser_device()

        if self.connected2serialClient:
            serialClient = self.serialClient
            print('in listen, serialClient = self.serialClient')

        print('started listening at connectionCheckClient')
        stagePrediction = 'n'
        for i in range(8):
                # stageEstimate is one of ['w', 'n', 'r']
                if self.connected2serialClient:
                    serialClient.write(b'c')
                    # print('clear sent to serialClient to reset')

                # シリアル通信ではバイナリ列でやりとりする必要があるので
                # バイナリ形式にエンコードする
                if self.connected2serialClient:
                    #serialClient.write(b'c')
                    serialClient.write(stagePrediction.encode('utf-8'))

                # wait 0.5 seconds
                time.sleep(0.5)
                # print('slept for 0.5 seconds.')

                # change label
                if stagePrediction == 'w':
                    stagePrediction = 'n'
                elif stagePrediction == 'n':
                    stagePrediction = 'r'
                elif stagePrediction == 'r':
                    stagePrediction = 'w'
