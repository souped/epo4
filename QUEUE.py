
class Node():
    def __init__(self, data=None, nextnode=None):
        self.data = data
        self.next = nextnode

    def set_data(self, data):
        self.data = data
    
class QList():
    def __init__(self, size=0):
        self.size = size
        self.head = None
        self.tail = None

    def add(self, data):
        newnode = Node(data)

        # if queue is empty, head = tail = newnode
        if self.tail == None:
            self.tail = newnode
            self.head = newnode
        # else append new node at the end.
        else:
            self.tail.next = newnode
            self.tail = newnode
        self.size += 1

    def pop(self):
        if self.head is None: return None
        temp = self.head
        self.head = self.head.next

        if self.head is None: self.tail = None
        self.size -= 1
        return temp.data
    
    def peek(self):
        return self.head.data

    def __sizeof__(self) -> int:
        return self.size
    
    def __str__(self) -> str:
        temp = self.head
        res = []
        for i in range(self.size):
            res.append(temp.data)
            temp = temp.next
        return res.__str__()

