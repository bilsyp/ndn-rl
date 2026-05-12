import os
import numpy as np

FILE_PATH = './belgium/'
OUTPUT_PATH = './mahimahi_traces/'
BYTES_PER_PKT = 1500.0
MILLISEC_IN_SEC = 1000.0
BITS_IN_BYTE = 8.0

def main():
    # Pastikan folder output ada
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    files = os.listdir(FILE_PATH)
    for trace_file in files:
        # Gunakan 'r' dan 'w' (mode teks), bukan 'rb'/'wb'
        with open(FILE_PATH + trace_file, 'r') as f, open(OUTPUT_PATH + trace_file, 'w') as mf:
            time_ms = []
            bytes_recv = []
            recv_time = []
            
            for line in f:
                parse = line.split()
                if len(parse) < 6: continue # Hindari baris kosong/rusak
                time_ms.append(float(parse[1]))
                bytes_recv.append(float(parse[4]))
                recv_time.append(float(parse[5]))

            time_ms = np.array(time_ms)
            bytes_recv = np.array(bytes_recv)
            recv_time = np.array(recv_time)
            throughput_all = bytes_recv / recv_time

            millisec_time = 0
            mf.write(str(millisec_time) + '\n')

            # Di Python 3, gunakan range(), bukan xrange()
            for i in range(len(throughput_all)):
                throughput = throughput_all[i]
                pkt_per_millisec = throughput / BYTES_PER_PKT 

                millisec_count = 0
                pkt_count = 0

                while True:
                    millisec_count += 1
                    millisec_time += 1
                    to_send = (millisec_count * pkt_per_millisec) - pkt_count
                    to_send = np.floor(to_send)

                    for _ in range(int(to_send)):
                        mf.write(str(millisec_time) + '\n')

                    pkt_count += to_send

                    if millisec_count >= recv_time[i]:
                        break

if __name__ == '__main__':
    main()