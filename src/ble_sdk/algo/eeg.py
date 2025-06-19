"""_summary_
本代码处理EEG信号数据。此数据只包含有效数据（不含帧头帧尾，共64次采样，192字节数据）
"""
import numpy as np
from scipy import signal


def one_ch_one_sampling_process(data):
    """用来处理单通道1次采样的三字节数据
    Args:
        data (str): 三字节数据 (DA43F2)

    Returns:
        int: 经过解析后的单通道1次采样的数据
    """
    EEGHEX = int(data, 16)  # 16进制转10进制
    tmp = int("0x800000", 16)  # 阈值
    if EEGHEX < tmp:
        EEGR = EEGHEX
        EEGV = EEGHEX * (4.5 / 8388607) / 8 * 1000000
    else:
        EEGR = EEGHEX - 16777216
        EEGV = EEGR * (4.5 / 8388607) / 8 * 1000000
    return int(EEGV)


def get_eeg_signal(input_list, channel_num):
    """获取eeg原始16进制数据，解析成10进制数

    Args:
        input_path (str): 数据地址
        channel_num (int): 通道数

    Raises:
        ValueError: 输入channel_num错误
        ValueError: 输入数据行长度错误

    Returns:
        ndarray: EEG解析后二维的ndarray，有多少个输入通道就有多少个维度——[[ch1-data],[ch2-data],[],[]……]，size为(channel_num × data_length)
    """

    # 验证channel_num合法性
    if channel_num not in {1, 2, 4, 8, 16}:
        raise ValueError("channel_num must be 1, 2, 4, 8, 16")

    # 常量定义
    BYTES_PER_SAMPLE = 3  # 每个采样点占3字节
    BYTES_PER_LINE = 192  # 每行192字节
    HEX_CHARS_PER_BYTE = 2  # 每个字节用2个十六进制字符表示

    # 初始化结果列表(先收集再转换，避免频繁数组操作)
    channels_data = [[] for _ in range(channel_num)]

    # with open(input_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(input_list):
        line = (
            line.strip()
        )  # 去除行末尾的换行符和空白字符，line就是192字节的纯净eeg数据
        if not line:  # 跳过空行
            continue

        # 验证行长度
        if len(line) != BYTES_PER_LINE * HEX_CHARS_PER_BYTE:
            # raise ValueError(f"行 {line_num+1} 数据错误: 应为 {BYTES_PER_LINE * HEX_CHARS_PER_BYTE} 字符，实际为 {len(line)}")
            continue

        # 计算每行包含的样本块数
        samples_per_channel = BYTES_PER_LINE // (BYTES_PER_SAMPLE * channel_num)

        # 遍历每个样本块
        for block in range(samples_per_channel):
            start_pos = block * BYTES_PER_SAMPLE * channel_num * HEX_CHARS_PER_BYTE

            # 提取每个通道的6字符(3字节)十六进制数据
            for ch in range(channel_num):
                chunk_start = start_pos + ch * BYTES_PER_SAMPLE * HEX_CHARS_PER_BYTE
                chunk_end = chunk_start + BYTES_PER_SAMPLE * HEX_CHARS_PER_BYTE
                hex_str = line[chunk_start:chunk_end]

                uv_value = one_ch_one_sampling_process(
                    hex_str
                )  # 解析单个通道单次采样数据

                channels_data[ch].append(uv_value)

    channels_data = np.array(channels_data, dtype=np.int32)

    # 计算PSD和FFT
    psd_result = compute_psd(channels_data, fs=500, n_fft=256, fmax=100)
    fft_result = compute_fft_amplitude(channels_data, fs=500, n_fft=256, fmax=100)

    return channels_data, psd_result, fft_result


def compute_psd(channels_data, fs=500, n_fft=256, fmax=100):
    """
    计算多通道PSD（功率谱密度）
    输出形状: (channel_num, 2, 256)
    维度2: 0=频率轴(0~100Hz), 1=功率值
    """
    channel_num = channels_data.shape[0]
    freqs, psd = signal.welch(channels_data, fs=fs, nperseg=n_fft, axis=1)
    # 截取0~100Hz并插值到256点
    idx_max = np.argmax(freqs >= fmax)  # 找到100Hz对应的索引
    freq_axis = np.linspace(0, fmax, 256)  # 插值后的频率轴

    psd_output = np.zeros((channel_num, 2, 256))
    for ch in range(channel_num):
        psd_output[ch, 0, :] = freq_axis  # 频率轴
        psd_output[ch, 1, :] = np.interp(
            freq_axis, freqs[:idx_max], psd[ch, :idx_max]
        )  # 功率值

    return psd_output


def compute_fft_amplitude(channels_data, fs=500, n_fft=256, fmax=100):
    """
    计算多通道FFT幅值谱（取模值）
    输出形状: (channel_num, 2, 256)
    维度2: 0=频率轴(0~100Hz), 1=幅值
    """
    channel_num = channels_data.shape[0]
    fft_result = np.fft.rfft(
        channels_data, n=n_fft, axis=1
    )  # 实数FFT（自动计算正频率）
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)  # 原始频率轴

    # 截取0~100Hz并插值到256点
    idx_max = np.argmax(freqs >= fmax)
    freq_axis = np.linspace(0, fmax, 256)  # 插值后的频率轴

    fft_output = np.zeros((channel_num, 2, 256))
    for ch in range(channel_num):
        fft_output[ch, 0, :] = freq_axis  # 频率轴
        amplitude = np.abs(fft_result[ch, :idx_max])  # 取模
        fft_output[ch, 1, :] = np.interp(freq_axis, freqs[:idx_max], amplitude)  # 幅值

    return fft_output


if __name__ == "__main__":

    nddata, psddd, ffttt = get_eeg_signal("./1-bci-eeg.txt", 1)
    print(nddata.shape, psddd.shape, ffttt.shape)

#     import matplotlib.pyplot as plt  # 只用来测试一下解析结果的正确性
#     plt.figure(figsize=(12, 4))
#     plt.plot(nddata[0, :])
#     plt.show()

#     # 绘制通道0的PSD和FFT结果
#     plt.figure(figsize=(12, 4))
#     plt.subplot(121)
#     plt.plot(psddd[0, 0, :], psddd[0, 1, :])
#     plt.title("PSD (0~100Hz)")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Power")
#     plt.subplot(122)
#     plt.plot(ffttt[0, 0, :], ffttt[0, 1, :])
#     plt.title("FFT Amplitude (0~100Hz)")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Amplitude")
#     plt.tight_layout()
#     plt.show()
