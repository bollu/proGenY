#pragma once
#include "core/Messaging/eventMgr.h"

class mainLoopListener : public Observer{
private:
	bool windowClosed;

public:

	mainLoopListener(eventMgr &eventManager) : windowClosed(false){

		eventManager.Register(Hash::getHash("windowClosed"), this);
		//dear lord using vim is hard >_<
	}

	void recieveEvent(const Hash *eventName, baseProperty *eventData){
		static const Hash *windowClosed = Hash::getHash("windowClosed"); 
		
		if(eventName == windowClosed){
			this->windowClosed = true;
		}
	}

	bool isWindowClosed(){
		return this->windowClosed;
	}
};
