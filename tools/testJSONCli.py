from web.jsonsocket import Client

host = '127.0.0.1'
port = 8008

# Client code:
client = Client()
client.connect(host, port).send("disgust/web1461854292144.png\n")
print "client send:"
response = client.recv(1)
print "response", response

client.close()
