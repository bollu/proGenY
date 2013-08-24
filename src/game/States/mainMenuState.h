#pragma once
#include "../../core/State/State.h"
#include "../../core/State/dummyStateSaveLoader.h"

class mainMenuState : public State{
public:
	mainMenuState() : State("mainMenuState"){};
	
	void Update(float dt){};
	
	void Draw(){};

	stateSaveLoader *createSaveLoader(){
		return new dummyStateSaveLoader();
	}

private:
	
};