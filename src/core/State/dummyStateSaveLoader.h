#pragma once
#include "stateSaveLoader.h"


class dummyStateSaveLoader : public stateSaveLoader{
public:
	dummyStateSaveLoader() : stateSaveLoader(NULL){};
	~dummyStateSaveLoader(){};
	 void Save(){ this->doneSaving = true; };
	 void Load(){ this->doneLoading = true; };

};