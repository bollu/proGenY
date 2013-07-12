#pragma once
#include <string>
#include "../Hash.h"

#include "../Process/processMgr.h"
#include "../Messaging/eventMgr.h"
#include "../Settings.h"
#include "stateSaveLoader.h"


/*!Used to create and manage States.

this is an implementation of the State Pattern. Each State represents
a logical "step" in an  FSM. It allows encapsulation of different stages in the
program. 

\sa stateMgr
*/
class State{
public:

	/*!initlaizes the State*/
	void Init(processMgr &_processManager, Settings &_settings, eventMgr &_eventManager){
		this->processManager = &_processManager;
		this->settings = &_settings;
		this->eventManager = &_eventManager;

		this->_Init();

	}
	
	virtual ~State(){};

	/*!instructs the State to provide it's saveLoader for saving / loading */ 
	virtual	stateSaveLoader *createSaveLoader() = 0;

	/*!instructs the State to update 
	@param [in] dt time between last frame and this frame
	*/
	virtual void Update(float dt) = 0;

	/*!instructs the State to Draw data*/
	virtual void Draw() = 0;

	/*!returns the name of the state
	\return the name of the state as a Hash *
	*/
	const Hash* getHashedName(){
		return this->hashedName;
	}

	/*!returns whether the FSM should change the state to a new state
	*/
	bool shouldChangeState(){
		return this->changingState;
	}

	/*!returns the name of the next state to which the FSM should switch to 
	*/
	std::string getNextStateName(){
		return this->nextStateName;
	}

protected:
	State(std::string name){
		hashedName = Hash::getHash(name);
		this->changingState = true;
		
	};

	/*!set the next state to which the FSM should switch to 
	*/
	void _setStateTransition(std::string _nextStateName){
		//TODO: include animations
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


