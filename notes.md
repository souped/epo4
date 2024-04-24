## TODO:
1. alle functionaliteiten van controle over de auto opdelen in aparte functies, 
1. ook vooral in aparte bestanden typen en dan importeren aangezien dat de leesbaarheid van code aanzienlijk verbeterd.


# DEADLINES:
⋅⋅⋅[brightspace link naar volledige planning](https://brightspace.tudelft.nl/d2l/le/content/595573/viewContent/3413888/View)
1. 8 mei: MODULES 1,2,3 AF
1. 15 mei: COMplete module 4 + midterm report writing + sign off modules 1,2,3
1. 24 mei: deadline midterm report ( 22:00 ) + MOdule 5 
1. 12 juni: complete challenge A
1. 19 juni: complete final challenge
1. 20 juni: deadline final report (22:00 (**STRICT!**))
1. 24 juni: final presentation and discussion

### TODO concreet:
- modify the \_\_init\_\_ so that when the communication with KITT is started, the beacon is initialized with the correct set of parameters. Use the existing serial port and send_command. 
- add two methods start_beacon and stop_beacon that turn the beacon on or off. Note that you should have set the beacon parameters during the \_\_init\_\_, so there is no need to resend them every time you turn the beacon on

- TODO: wasd Function:
• The wasd function is designed to be a continuous loop that reads keyboard events using the keyboard library which you loaded using import keyboard.
• When a key is pressed (KEY_DOWN), the function interprets the key and adjusts KITT’s speed and steering angle or toggles the beacon accordingly.
• The ’w’ key accelerates KITT forward, ’s’ stops KITT, ’a’ turns KITT left, and ’d’ turns KITT right.
• The ’e’ key turns on the beacon, and the ’q’ key turns off the beacon.
• When a key is released (KEY_UP), you could define appropriate actions, e.g. stop KITT or reset the steering angle.

#### driving commands:
- motor commands:
- neutral: 150
- forward: 165
- backward: 135
\n   
- steering commands (angle)
- hard left: 200
- hard right: 100
general format:   
`serial_port.write(b'code\n')`
motor: `M{speed}`
angle: `D{angle}`

audio beacon commands:
- turn on: `A1`
- turn off: `A0`
! note: 
```python
# Be aware that the default code word for the beacon is 0x00000000, which means KITT will not start making noise on its own when the beacon is turned on. You should specify a code as described below before you can hear the beacon make noise. The beacon signal is similar to what was used in EE2T11 Telecommunication A practicum, except that now it is possible to use an arbitrary carrier frequency, bit frequency, and repetition count.
#You can set the carrier frequency onto which a code is transmitted to a maximum frequency of 30 kHz.
# It can be set to, for example, 10000 Hz by using
carrier_frequency = 10000.to_bytes(2, byteorder='big')
serial_port.write(b'F' + carrier_frequency + b'\n')
# Furthermore, you can set the bit-frequency of a code that is transmitted with OOK on the carrier frequency to, for example, 5000 Hz by using:
bit_frequency = 5000.to_bytes(2, byteorder='big')
serial_port.write(b'B' + bit_frequency + b'\n')
#The repetition count, which sets the time between consecutive transmissions according to the formula repetition_count = bit_frequency/repetition_frequency and has a minimum of 32, can be set to, for example, 2500 by using:
repetition_count = 2500.to_bytes(2, byteorder='big')
serial_port.write(b'R' + repetition_count + b'\n')
# The 32 bits code pattern is transmitted bit-wise over the beacon. This code must be specified in hexadecimal; say you use the hexadecimal 0xDEADBEEF as an example; the command to do this is:
code = 0xDEADBEEF.to_bytes(4, byteorder='big')
serial_port.write(b'C' + code + b'\n')
```

