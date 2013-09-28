#pragma once
#include "stateSaveLoader.h"


/*!a stateSaveLoader that does nothing
can be used along with states that don't need to save or load 
any data. It can also be used while testing
*/
class dummyStateSaveLoader : public stateSaveLoader{
public:
	dummyStateSaveLoader() : stateSaveLoader(NULL){};
	~dummyStateSaveLoader(){};
	 void Save(){ this->doneSaving = true; };
	 void Load(){ this->doneLoading = true; };

};