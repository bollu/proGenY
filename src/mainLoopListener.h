#pragma once
#include "core/Messaging/eventMgr.h"

class mainLoopListener : public Observer{
private:
	bool windowClosed;

public:

	mainLoopListener(eventMgr &eventManager) : windowClosed(false){

		eventManager.Register(Hash::getHash("windowClosed"), this);
		eventManager.Register(Hash::getHash("keyPressed"), this);
	}

	void recieveEvent(const Hash *eventName, baseProperty *eventData){
		static const Hash *windowClosed = Hash::getHash("windowClosed"); 
		static const Hash *keyPressed = Hash::getHash("keyPressed");
		
		if(eventName == windowClosed){
			this->windowClosed = true;
		}
		if(eventName == keyPressed){
			Prop<sf::Event::KeyEvent> *eventProp = 
				dynamic_cast< Prop<sf::Event::KeyEvent> *>(eventData); 
			assert(eventProp != NULL && "\nunable to receive event data\n");

			sf::Event::KeyEvent *event = eventProp->getVal();
			if(event->code == sf::Keyboard::Key::Escape){
				this->windowClosed = true;
			}

			
		}
	}

	bool isWindowClosed(){
		return this->windowClosed;
	}
};
