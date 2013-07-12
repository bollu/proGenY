#pragma once
#include "../../core/State/State.h"
#include "../../core/State/dummyStateSaveLoader.h"



#include "../level/gameSegment.h"

class gameSegmentLoader : public State{
private:
	gameSegment *segment;

	std::string segmentName;

	bool doneLoading;
public:
	gameSegmentLoader() : State("gameSegmentLoader"){}
	
	void Update(float dt){};
	void Draw(){};

	
	stateSaveLoader *createSaveLoader(){
		return new dummyStateSaveLoader();
	}

	gameSegment *getLoadedSegment();
};