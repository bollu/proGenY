#pragma once
#include <string>
#include "../Hash.h"

#include "../Process/processMgr.h"
#include "../Messaging/eventMgr.h"
#include "../Settings.h"
#include "stateSaveLoader.h"


class State{
public:

	
	void Init(processMgr &_processManager, Settings &_settings, eventMgr &_eventManager){
		this->processManager = &_processManager;
		this->settings = &_settings;
		this->eventManager = &_eventManager;

		this->_Init();

	}
	
	virtual ~State(){};

	//create the saver and loader for this particular state
	virtual	stateSaveLoader *createSaveLoader() = 0;

	virtual void Update(float dt) = 0;
	virtual void Draw() = 0;

	const Hash* getHashedName(){
		return this->hashedName;
	}

	bool shouldChangeState(){
		return this->changingState;
	}

	std::string getNextStateName(){
		return this->nextStateName;
	}

protected:
	State(std::string name){
		hashedName = Hash::getHash(name);
		this->changingState = true;
		
	};

	void _setStateTransition(std::string _nextStateName){
		this->nextStateName = _nextStateName;
		this->changingState = true;
	}

	virtual void _Init(){};

	processMgr *processManager;
	eventMgr *eventManager;
	Settings *settings;
private:
	 const Hash* hashedName;
	 
	 std::string nextStateName; 
	 
	 //whether a state change must take place. 
	 bool changingState;

};


