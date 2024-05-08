from threading import Thread, Event
from queue import Queue
from time import sleep, time

command_queue = Queue()