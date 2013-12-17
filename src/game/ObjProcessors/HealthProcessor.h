#pragma once
#include "../../core/componentSys/processor/ObjectProcessor.h"
#include "../../core/componentSys/Object.h"

#include "../../core/controlFlow/processMgr.h"
#include "../../core/IO/Settings.h"
#include "../../core/controlFlow/eventMgr.h"



struct healthData {
private:
	bool invul;

	unsigned int maxHP;
	//HP may go below zero.
	int currentHP;
public:

	healthData(){
		this->maxHP = this->currentHP = -1;
		this->invul = false;
	}


	void setHP(unsigned int HP){
		this->currentHP = this->maxHP = HP;

	}

	void makeInvulnerable(){
		this->invul = true;
	}

	void makeVulnerable(){
		this->invul = false;
	}

	void Damage(unsigned int damage){

		if(!this->invul){
			this->currentHP -= damage;
		}
	}

	void Heal(unsigned int heal){
		this->currentHP += heal;
	}

	int getHP(){
		return this->currentHP;
	};


};

class healthProcessor : public ObjectProcessor {

public:
	healthProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager);
protected:
	void _Process(Object *obj, float dt);

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("healthData");
	};
};
