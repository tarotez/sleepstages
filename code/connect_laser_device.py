import serial

def connect_laser_device():
    """
    RobotDyn UNOはブートのタイミングによってwebserial1420だったりwebserial1410だったりする
    この他のIDでマウントされることは確認していないが、あり得る
    そこで、IDの候補をあらかじめリストとして保持し、
    リスト中のID全てでSerialのインスタンスを作るよう試行を行う
    """
    id_candidate_list = ['/dev/tty.usbserial-1410',
                         '/dev/tty.usbserial-1420',
                         '/dev/tty.wchusbserial1410',
                         '/dev/tty.wchusbserial1420']

    connected2serialClient = False
    for id_candidate in id_candidate_list:
        # print('trying to connect to ', id_candidate)
        try:
            # serial_module = importlib.import_module("serial")
            # self.serialClient = serial_module.Serial(id_candidate, 9600)
            serialClient = serial.Serial(id_candidate, 9600)
            print('connected to serial client:', id_candidate)
            connected2serialClient = True
            # print('connected2serialClient = True')
            return serialClient, connected2serialClient

        except Exception as e:
            # print(e)
            # print('could not connect to', id_candidate)
            pass

    return None, connected2serialClient
