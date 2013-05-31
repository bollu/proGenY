#pragma once
#include <string>
#include "../Hash.h"
#include "../Messaging/eventMgr.h"
#include "../Settings.h"


class Process{
protected:
	const Hash *nameHash;

	Process(std::string _name){
		this->nameHash = Hash::getHash(_name);
	};

public:
	virtual ~Process(){};

	virtual void Pause(){};
	virtual void Resume(){};

	virtual void preUpdate(){};
	virtual void Update(float dt){};
	virtual void Draw(){};
	virtual void postDraw(){};


	//virtual void Startup() = 0;
	virtual void Shutdown(){};

	const Hash *getNameHash(){
		return this->nameHash;
	}

};