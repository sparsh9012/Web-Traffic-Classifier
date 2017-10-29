import numpy as np
np.set_printoptions(threshold=np.nan)
from os import walk
import random

def create_dataset(path):

    # arrays that save source and destination IPv4 strings for every flow.
    src_ip = []
    dst_ip = []

    # arrays that save protocol number(in string format) and array of flags that indicate the use of a specific protocol for every flow.
    prot_num = []
    prot_flag = []

    # array of flags that indicate the use of specific icmp code for every flow.
    icmp_code_flags = []

    # array of flags that indicate the use of specific tcp flags for every flow.
    tcp_flags = []

    # array of flags that indicate the use of a specific L4 source and destination port for every flow.
    src_port = []
    dst_port = []
    dst_port_num = [] # array that holds the L4 destination port number for every flow.
    
    # arrays that save number of incoming and outgoing packets for every flow.
    packets_in = []
    packets_out = []
    
    # arrays that save average byte size of incoming and outgoing packets for every flow.
    packet_in_bytes = []
    packet_out_bytes = []

    # array that saves duration of each flow.
    duration = []

    # arrays that save source and destination ASNs strings for every flow.
    src_AS = []
    dst_AS = []

    # array that holds all the extracted informations for every flow.
    data = []

    ip_dstport = {} # hash table that counts total flow packets from source ip to L4 destination port
    ip_dstport_ret = {} # hash table that counts total flow packets returned from our AS to specific source ip (with specific L4 destination port)
    subnet_dstport = {} # hash table that counts total flow packets from source subnet to L4 destination port
    

    for (dirpath, dirname, file_names) in walk(path):
        for file_name in sorted(file_names):
            f = open(path + file_name)
            f.readline() # skip the first line which shows the template of the flows
            for line in f:
                col = line.split("|")

                # save source and destination IPv4 strings for current flow
                src_ip.append(col[0])
                dst_ip.append(col[1])


                # save the number of incoming and outgoing packets for current flow.
                packets_in.append([ int(float(col[2])) ])
                packets_out.append([ int(float(col[3])) ])


                # calculate average number of bytes of an incoming packet for current flow.
                try:
                    inc = int(float(col[4]))//packets_in[-1]
                except:
                    inc = 0
                packet_in_bytes.append([inc])


                # calculate average number of bytes of an outgoing packet for current flow.
                try:
                    outg = int(float(col[5]))//packets_out[-1]
                except:
                    outg = 0
                packet_out_bytes.append([outg])


                # extract protocol number and flag from current flow
                t = col[6]
                prot_num.append(t)
                if t=='1': # ICMP
                    prot_flag.append([1, 0, 0])
                elif t=='6': # TCP
                    prot_flag.append([0, 1, 0])
                elif t=='17': # UDP
                    prot_flag.append([0, 0, 1])
                else: # other protocol
                    prot_flag.append([0, 0, 0])


                # extract icmp code from current flow. nProbe field is ICMP_Type * 256 + ICMP code.
                IC = int(float(col[7]))%256
                icmp_code_flags.append([ int(IC==0), int(IC==1), int(IC==2), int(IC==3), int(IC==4), int(IC==5), int(IC==6), int(IC==7), int(IC==8), int(IC==9), int(IC==10), int(IC==11), int(IC==12), int(IC==13), int(IC==14), int(IC==15) ])


                # extract tcp flags from current flow. 'tcp_flags' list format: [URG, ACK, PSH, RST, SYN, FIN].
                TF = '{0:06b}'.format(int(col[8])) # convert flow field which contains cumulative of all flow TCP flags to binary format
                tcp_flags.append([ int(TF[0]), int(TF[1]), int(TF[2]), int(TF[3]), int(TF[4]), int(TF[5]) ])


                # extract L4 source port from current flow. 'src_port' list format:
                # [FTP data,FTP control,SSH,TELNET,SMTP,HOST NAME SERVER,DNS,BOOTP server,BOOTP client,TFTP,HTTP,POP3,SFTP,SQL Services,NTP,SQL Service,SNMP,BGP,HTTPS,ports>1023]
                sp = col[9]
                src_port.append([ int(sp=='20'), int(sp=='21'), int(sp=='22'), int(sp=='23'), int(sp=='25'), int(sp=='42'), int(sp=='53'), int(sp=='67'), int(sp=='68'), int(sp=='69'), int(sp=='80'), int(sp=='110'), int(sp=='115'), int(sp=='118'), int(sp=='123'), int(sp=='156'), int(sp=='161'), int(sp=='179'), int(sp=='443'), int(int(sp)>1023) ])


                # extract L4 destination port from current flow. 'dst_port' list format:
                # [FTP data,FTP control,SSH,TELNET,SMTP,HOST NAME SERVER,DNS,BOOTP server,BOOTP client,TFTP,HTTP,POP3,SFTP,SQL Services,NTP,SQL Service,SNMP,BGP,HTTPS,ports>1023]
                dp = col[10]
                dst_port_num.append(dp)
                dst_port.append([ int(dp=='20'), int(dp=='21'), int(dp=='22'), int(dp=='23'), int(dp=='25'), int(dp=='42'), int(dp=='53'), int(dp=='67'), int(dp=='68'), int(dp=='69'), int(dp=='80'), int(dp=='110'), int(dp=='115'), int(dp=='118'), int(dp=='123'), int(dp=='156'), int(dp=='161'), int(dp=='179'), int(dp=='443'), int(int(dp)>1023) ])


                # calculate the duration(in milliseconds) of the current flow. (FLOW_END_MILLISECONDS - FLOW_START_MILLISECONDS)
                duration.append([int(float(col[14]) - float(col[13]))])


                # calculate flag that indicates if source and destination ASNs are 3323 (Ntua ASN, which is the AS where we did our experiments)
                src_AS.append(int(col[11]=="3323"))
                dst_AS.append(int( (col[12]=="3323") or (col[11]!="3323" and col[12]!= "3323") ))


                # fill ip_dstport and ip_dstport_ret hash tables
                key = col[0],col[1],col[6],col[10] # create hash key for this flow
                if key in ip_dstport: # add extra packets to this existing flow
                    ip_dstport[key] += int(float(col[2]))
                    ip_dstport_ret[key] += int(float(col[3]))
                else: # this flow does not exist in the hash table so we add it
                   ip_dstport[key] = int(float(col[2]))
                   ip_dstport_ret[key] = int(float(col[3]))

                
                # fill subnet_dstport hash table
                key = col[0][:11],col[1],col[6],col[10] # create hash key for this flow
                if key in subnet_dstport: # add extra packets to this existing flow
                    subnet_dstport[key] += int(float(col[2]))
                else: # this flow does not exist in the hash table so we add it
                   subnet_dstport[key] = int(float(col[2]))


                # final data format: [protocol flags, number of incoming packets, number of outgoing packets, average bytes of an incoming packet, average bytes of an outgoing packet, src_port, dst_port, tcp_flags, icmp_code, duration, total number of packets from source ip to L4 dst port for all the flows, flag that indicates if source ASN is 3323, flag that indicates if destination ASN is 3323]

                # data with no aggregation
                #data.append( prot_flag[-1] + packets_in[-1] + packets_out[-1] + packet_in_bytes[-1] + packet_out_bytes[-1] + src_port[-1] + dst_port[-1] + tcp_flags[-1] + icmp_code_flags[-1] + duration[-1] + [src_AS[-1], dst_AS[-1]] )

                # data with no aggregation and no outgoing fields
                #data.append( prot_flag[-1] + packets_in[-1] + packet_in_bytes[-1] + src_port[-1] + dst_port[-1] + tcp_flags[-1] + icmp_code_flags[-1] + duration[-1] + [src_AS[-1], dst_AS[-1]] )

                # data with progressive aggregation
                #data.append( prot_flag[-1] + packets_in[-1] + packets_out[-1] + packet_in_bytes[-1] + packet_out_bytes[-1] + src_port[-1] + dst_port[-1] + tcp_flags[-1] + icmp_code_flags[-1] + duration[-1] + [ip_dstport[src_ip[-1], dst_ip[-1], prot_num[-1], dst_port_num[-1]]] + [ip_dstport_ret[src_ip[-1], dst_ip[-1], prot_num[-1], dst_port_num[-1]]] + [src_AS[-1], dst_AS[-1]] )

    # data with aggregation and no outgoing packets
    for i in range(len(prot_flag)):
        #data.append( prot_flag[i] + packets_in[i] + packet_in_bytes[i] + src_port[i] + dst_port[i] + tcp_flags[i] + icmp_code_flags[i] + duration[i] + [ip_dstport[src_ip[i], dst_ip[i], prot_num[i], dst_port_num[i]]]  + [src_AS[i], dst_AS[i]] )

        # data with aggregation but also outgoing fields
        data.append( prot_flag[i] + packets_in[i] + packets_out[i] + packet_in_bytes[i] + packet_out_bytes[i] + src_port[i] + dst_port[i] + tcp_flags[i] + icmp_code_flags[i] + duration[i] + [ip_dstport[src_ip[i], dst_ip[i], prot_num[i], dst_port_num[i]]] + [ip_dstport_ret[src_ip[i], dst_ip[i], prot_num[i], dst_port_num[i]]] + [src_AS[i], dst_AS[i]] )


####[ip_dstport[src_ip[i], dst_ip[i], prot_num[i], dst_port_num[i]]]
####[ip_dstport_ret[src_ip[i], dst_ip[i], prot_num[i], dst_port_num[i]]]
####[subnet_dstport[src_ip[i][:11], dst_ip[i], prot_num[i], dst_port_num[i]]]

    return data


def pre_normalization(x):
    xmin = np.amin(x, 0)
    xmax = np.amax(x, 0)
    scale = xmax - xmin
    (i,j) = x.shape
    #print(x.shape)
    for i in range(0,j):
        if scale[i]==0:
            scale[i] = 1
    #print(scale)
    return scale,xmin

def normalize(x, scale, xmin): 
    x_norm = (x-xmin)/(scale)
    return x_norm

def create_input_rand_pattern(samples_per_categ, check_icmp, check_tcp, check_udp, check_ps, check_lg):
    icmp_flood_data = create_dataset("./icmp_flood_flows/")
    tcp_syn_flood_data = create_dataset("./tcp_syn_flood_flows/")
    udp_flood_data = create_dataset("./udp_flood_flows/")
    port_scan_data = create_dataset("./port_scan_flows/")
    legit_data = create_dataset("./legit_traffic_flows/")
    
    inp = []
    out = []

    legit = 0
    icmp = 0
    tcp = 0
    udp = 0
    ps = 0

    if (check_icmp):
        for i in range(0, samples_per_categ):
            out.append([1])
    if (check_tcp):
        for i in range(0, samples_per_categ):
            out.append([2])
    if (check_udp):
        for i in range(0, samples_per_categ):
            out.append([3])
    if (check_ps):
        for i in range(0, samples_per_categ):
            out.append([4])
    if (check_lg):
        for i in range(0, samples_per_categ):
            out.append([0])

    #print(out)
    random.shuffle(out)
    #print(out)

    for i in range(0, len(out)):
        if(out[i]==[1]):
            inp.append(icmp_flood_data[icmp])
            icmp = icmp + 1
        if(out[i]==[2]):
            inp.append(tcp_syn_flood_data[tcp])
            tcp = tcp + 1
        if(out[i]==[3]):
            inp.append(udp_flood_data[udp])
            udp = udp + 1
        if(out[i]==[4]):
            inp.append(port_scan_data[ps])
            ps = ps + 1
        if(out[i]==[0]):
            inp.append(legit_data[legit])
            legit = legit + 1

    samples = icmp + tcp + udp + ps + legit # total number of samples

    write_to_file("./data/data_all_attacks_rand_pattern.txt", inp, samples, 1)
    write_to_file("./data/labels_all_attacks_rand_pattern.txt", out,  samples, 1)

    x = np.array(inp)
    y = np.array(out)
    
    return (x, y, samples)

def create_block_input(sample_length, sequence_length, check_icmp, check_tcp, check_udp, check_ps, check_lg):
    icmp_flood_data = create_dataset("./icmp_flood_flows/")
    tcp_syn_flood_data = create_dataset("./tcp_syn_flood_flows/")
    udp_flood_data = create_dataset("./udp_flood_flows/")
    port_scan_data = create_dataset("./port_scan_flows/")
    legit_data = create_dataset("./legit_traffic_flows/")

    inp = []
    output = []

    samples = sample_length
    seq_len = sequence_length

    legit = 0
    icmp = 0
    tcp = 0
    udp = 0
    port_scan = 0

    loops = int(samples/seq_len)

    for i in range(0, loops):
        if (i%5 == 0) and check_icmp:
            for k in range(icmp, icmp+seq_len):
                inp.append(icmp_flood_data[k])
                output.append([1])
            icmp = icmp + seq_len
        elif (i%5 == 1) and check_tcp:
            for k in range(tcp, tcp+seq_len):
                inp.append(tcp_syn_flood_data[k])
                output.append([2])
            tcp = tcp + seq_len
        elif (i%5 == 2) and check_udp:
            for k in range(udp, udp+seq_len):
                inp.append(udp_flood_data[k])
                output.append([3])
            udp = udp + seq_len
        elif (i%5 == 3) and check_ps:
            for k in range(port_scan, port_scan+seq_len):
                inp.append(port_scan_data[k])
                output.append([4])
            port_scan = port_scan + seq_len
        elif (i%5 == 4) and check_lg:
            for k in range(legit, legit+seq_len):
                inp.append(legit_data[k])
                output.append([0])
            legit = legit + seq_len
    
    samples = icmp + tcp + udp + port_scan + legit # total number of samples

    write_to_file("./data/eee.txt", inp, samples, seq_len)
    write_to_file("./data/ee.txt", output,  samples, 1)

    x = np.array(inp)
    y = np.array(output)
    
    return (x, y, samples)


def write_to_file(filename, x, samples, seq_length):
    out_file = open(filename, 'w')

    out_file.write(str(samples) + "," + str(seq_length) + "\n") # write sample|seq_length

    for i in range(0, len(x)):
        for item in x[i]:
            out_file.write(str(item)+"|")
        out_file.write("\n")
    out_file.close()


def read_from_file(filename):
    in_file = open(filename, 'r')

    col = in_file.readline().split(",")
    samples = int(float(col[0]))
    seq_length = int(float(col[1]))

    inp = []
    for line in in_file:
        col = line.split("|")
        temp = []
        for i in range(0, len(col)-1):
            temp = temp + [int(float(col[i]))]
        inp.append(temp)

    return (samples, seq_length, inp)
