import numpy as np


def imp_str2hex(imp_str):
    """阻抗原始字符串数据转换为int，并切分合并两个字符为一组"""
    imp_data = []
    for i in range(0, len(imp_str), 2):
        imp_data.append(int(imp_str[i : i + 2], base=16))
    return imp_data


def cal_imp(imp_data_buf):
    """计算阻抗-单通道-80个采样点，输出一个阻抗值"""
    # Constants
    leadOffDrive_amps = 6.0e-9  # 6 nA
    eeg_res = {"databuf": [0] * 4, "data": 0}  # Simulated structure
    imp_data_raw = [0] * 80
    imp_data_convert_uv = [0.0] * 80

    for i in range(80):
        if (imp_data_buf[i * 3] & 0x80) == 0x80:
            eeg_res["databuf"][3] = 0xFF
            eeg_res["databuf"][2] = imp_data_buf[i * 3]
            eeg_res["databuf"][1] = imp_data_buf[i * 3 + 1]
            eeg_res["databuf"][0] = imp_data_buf[i * 3 + 2]
            # eeg_res["data"] = (eeg_res["databuf"][3] << 24) | (eeg_res["databuf"][2] << 16) | (eeg_res["databuf"][1] << 8) | eeg_res["databuf"][0]
            eeg_res["data"] = (
                (eeg_res["databuf"][3])
                | (eeg_res["databuf"][2] << 16)
                | (eeg_res["databuf"][1] << 8)
                | eeg_res["databuf"][0]
            )

            imp_data_raw[i] = eeg_res["data"]
            imp_data_convert_uv[i] = imp_data_raw[i] * 4.5 / 1 / 8388607 * 1e6
        else:
            # eeg_res["databuf"][3] = 0xFF
            eeg_res["databuf"][3] = 0x0
            eeg_res["databuf"][2] = imp_data_buf[i * 3]
            eeg_res["databuf"][1] = imp_data_buf[i * 3 + 1]
            eeg_res["databuf"][0] = imp_data_buf[i * 3 + 2]
            # eeg_res["data"] = (eeg_res["databuf"][3] << 24) | (eeg_res["databuf"][2] << 16) | (eeg_res["databuf"][1] << 8) | eeg_res["databuf"][0]
            eeg_res["data"] = (
                (eeg_res["databuf"][3])
                | (eeg_res["databuf"][2] << 16)
                | (eeg_res["databuf"][1] << 8)
                | eeg_res["databuf"][0]
            )
            imp_data_raw[i] = eeg_res["data"]
            imp_data_convert_uv[i] = imp_data_raw[i] * 4.5 / 1 / 8388607 * 1e6

    data_std_uv = np.std(np.array(imp_data_convert_uv))
    impedance_uvrms = 1.414 * data_std_uv / leadOffDrive_amps / 1e6 / 1e3

    return impedance_uvrms



# 入口
def imp_conversion(raw_data):
    r_list = imp_str2hex(raw_data)
    imp_data = cal_imp(r_list)
    return imp_data


if __name__ == "__main__":
    # device = ['16_wifi', 'd8_ble', 'd4_ble', '4_wifi']

    # 4通道ble的阻抗
    # file = r'./data\exp_data\bluetooth_4.channel.txt'
    # # 设备类型
    # device = 'd4_ble'

    # # 4通道wifi的阻抗
    # file = r'./data\exp_data\wifi_4.channel.txt'
    # # 设备类型
    # device = '4_wifi'

    # # 8通道ble的阻抗
    # file = r'./data\exp_data\bluetooth_8.channel.txt'
    # # 设备类型
    # device = 'd8_ble'

    # 16通道wifi的阻抗
    file = r"./data\exp_data\wifi_16_channel.txt"
    # 设备类型
    device = "16_wifi"

    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            ch_name, ch_imp_hex = line.strip().split(",")
            res = imp_conversion(device, ch_name, ch_imp_hex)
            print(res)
