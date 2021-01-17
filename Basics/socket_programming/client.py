# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 2020

@author: Kwong Cheong Ng
@filename: client.py
@coding: utf-8
========================
Date          Comment
========================
02042020      First revision 
02132020      Buffer streaming data and conversion
02142020      Use pickle library to send Python Objects
"""
HEADERSIZE = 10

import pickle # 02142020
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((socket.gethostname(), 1243))

while True:
#    full_msg = ''
    full_msg = b'' # 02142020
    new_msg = True
    while True:
        msg = s.recv(16)
        if new_msg:
            print("new msg len:", msg[:HEADERSIZE])
            msglen = int(msg[:HEADERSIZE])
            new_msg = False

        print(f"full message length: {msglen}")

        #full_msg += msg.decode("utf-8")
        full_msg += msg
        
        print(len(full_msg))

    if len(full_msg)-HEADERSIZE == msglen:
        print("full msg recvd")    
        print(full_msg[HEADERSIZE:])
        print(pickle.loads(full_msg[HEADERSIZE:]))
        new_msg = True
        full_msg = b""