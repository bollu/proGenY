#pragma once
#include "../../core/componentSys/processor/ObjectProcessor.h"
#include "../../core/componentSys/Object.h"

#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/EventManager.h"



struct HealthData {
private:
	friend class healthProcessor;
	//HP may go below zero.
	int currentHP;

public:
	bool invul;
	unsigned int maxHP;

	HealthData(){
		this->maxHP = this->currentHP = -1;
		this->invul = false;
	}

	int getHP(){
		return this->currentHP;
	}
	
};

class healthProcessor : public ObjectProcessor {
public:
	healthProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager);
protected:
	void _Process(Object *obj, float dt);
	void _onObjectAdd(Object *obj);
	bool _shouldProcess(Object *obj){
		return obj->hasProperty("HealthData");
	};
};
