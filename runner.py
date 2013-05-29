

import os.path
import sys

if len(sys.argv) < 2 :
	print("provide path of executable")	
	sys.exit()

if(not os.path.isfile(sys.argv[1]) ):
	print("executable has not been created. build has been unsucessful")	
	sys.exit()


doubleQuote = ' " '
singleQuote = " ' "

echoMsg = 'press any key to continue..'
echoCmd = "echo " + singleQuote + echoMsg + singleQuote

waitCmd = "read -n 1 c"

xTermCmd = "xterm -e"


command = xTermCmd + doubleQuote +  sys.argv[1] + ";" + echoCmd + ";" + waitCmd + doubleQuote
os.system(command)


		

