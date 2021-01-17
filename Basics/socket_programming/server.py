# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 2020

@author: Kwong Cheong Ng
@filename: server.py
@coding: utf-8
========================
Date          Comment
========================
02052020      First revision 
02132020      Buffer streaming data and conversion
02142020      Use pickle library to send Python Objects
"""
import socket
import time 
import pickle # 02142020

# create the socket
# AF_INET == ipv4
# SOCK_STREAM == TCP
HEADERSIZE = 10 # 02132020

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1243))
s.listen(5)

while True:
    # now our endpoint knows about the OTHER endpoint.
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established")

    # 02132020
    # 02142020 convert msg into dictionary list
    #msg = "Welcome to the server!"
    #msg = f"{len(msg):<{HEADERSIZE}}" + msg
    d = {1: "Hello", 2: "World"}
    msg = pickle.dumps(d)
    msg = bytes(f"{len(msg):<{HEADERSIZE}}", 'utf-8') + msg
    print(msg)
    clientsocket.send(msg) # no need to convert msg into bytes
    #clientsocket.send(bytes(msg, "utf-8"))
    
    while True:
        time.sleep(3)
        msg = f"The time is {time.time()}"
        msg = f"{len(msg):{HEADERSIZE}}" + msg 
        
        print(msg)
        clientsocket.send(bytes(msg, "utf-8"))
    
    clientsocket.close()
