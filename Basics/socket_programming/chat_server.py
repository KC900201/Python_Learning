# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 2020

@author: Kwong Cheong Ng
@filename: chat_server.py
@coding: utf-8
========================
Date          Comment
========================
02152020      First revision (chat bot programming)
"""

import socket
import select

HEADER_LENGTH = 10 # buffer streaming and conversion
IP = "127.0.0.1"
PORT = 1234

# setup socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# setup fixed address to overcome "address already in use"
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# bind IP to PORT
server_socket.bind((IP, PORT))
# make server to listen to new connections
server_socket.listen()
# create a list of sockets to keep track of
sockets_list = [server_socket]

clients = {}

# debug info
print(f'Listening for connections on {IP}:{PORT}...')

# Receive msg
def receive_msg(client_socket):
    try:
        msg_header = client_socket.recv(HEADER_LENGTH)

        if not len(msg_header):
            return False

        msg_length = int(msg_header.decode('utf-8').strip())
        
        return {'header': msg_header, 'data': client_socket.recv(msg_length)}
    except:
        # Something went wrong like empty message or client exited abruptly
        return False

while True:
    read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list)

    # Iterate over notified sockets
    for notified_socket in read_sockets:
        # If notified socket is a server socket - new connection, accept it
        if notified_socket == server_socket:
            client_socket, client_address = server_socket.accept()
            user = receive_msg(client_socket)
            if user is False:
                continue
            sockets_list.append(client_socket)
            # continue 02152020
            clients[client_socket] = user
            
            print('Accepted new connection from {}:{}, username: {}'.format(*client_address, user['data'].decode('utf-8')))
        
        # Else existing socket is sending a message
        else:
            # Receive message 
            message = receive_msg(notified_socket)
            
            # If False, client disconnected, cleanup
            if message is False:
                print('Closed connection from: {}'.format(clients[notified_socket]))

                # Remove from list for socket.socket()
                sockets_list.remove(notified_socket)
                
                # Remove from list of users
                del clients[notified_socket]
                
                continue
            
            # Get user by notified socket
            user = clients[notified_socket]

            print(f'Received message from {user["data"].decode("utf-8")}: {message["data"].decode("utf-8")}')

            # Iterate over connected clients and broadcast message
            for client_socket in clients:
                if client_socket != notified_socket:
                    client_socket.send(user['header'] + user['data'] + message['header'] + message['data'])

    for notified_socket in exception_sockets:
        # Remove from list for socket.socket()
        sockets_list.remove(notified_socket)
        # Remove from list of users
        del clients[notified_socket]
