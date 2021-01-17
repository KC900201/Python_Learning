# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 2020

@author: Kwong Cheong Ng
@filename: chat_client.py
@coding: utf-8
========================
Date          Comment
========================
02152020      First revision (chat bot programming)
"""
import sys
import socket
import select
import errno

HEADER_LENGTH = 10 # buffer streaming and conversion
IP = "127.0.0.1"
PORT = 1234

my_username = input("Username: ")

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((IP, PORT))

# set recv method to not block
client_socket.setblocking(False)

username = my_username.encode('utf-8')
username_header = f"{len(username):<{HEADER_LENGTH}}".encode('utf-8')
client_socket.send(username_header + username)

while True:
    message = input(f'{my_username} > ')

    if message:
        # Encode message to bytes, prepare header and convert to bytes
        message = message.encode('utf-8')
        message_header = f"{len(message):<{HEADER_LENGTH}}".encode('utf-8')
        client_socket.send(message_header + message)
    
    try:
        # Now we want to loop over received messages
        while True:
            username_header = client_socket.recv(HEADER_LENGTH)
            
            if not len(username_header):
                print('Connection closed by the server')
                sys.exit()
            # Convert header to int value
            username_length: int(username_header.decode('utf-8').strip())
            # Receive and decode username 
            username = client_socket.recv(username_length).decode('utf-8')
            # Now do the same for message (as we received username, we received)
            message_header = client_socket.recv(HEADER_LENGTH)
            message_length = int(message_header.decode('utf-8').strip())
            message = client_socket.recv(message_length).decode('utf-8')

            # Print message
            print(f'{username} > {message}')

    except IOError as e:
        if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
            print('Reading error: {}'.format(str(e)))
            sys.exit()
        # Did no receive anything
        continue            

    except Exception as e:
        # Any other exception - something happened, exit
        print('Reading error: '.format(str(e)))
        sys.exit()