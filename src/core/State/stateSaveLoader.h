#pragma once


class State;

class stateSaveLoader{
public:
	bool isDoneSaving(){
		return this->doneSaving;
	};

	bool isDoneLoading(){
		return this->doneLoading;
	};

	virtual void Save() = 0;
	virtual void Load() = 0;

	virtual ~stateSaveLoader(){};
protected:
	stateSaveLoader(State *state){
		this->doneSaving = this->doneLoading = false;
	};


	bool doneSaving;
	bool doneLoading;
};