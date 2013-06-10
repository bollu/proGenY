#pragma once
#include "../../core/objectProcessor.h"
#include "../../core/Object.h"

#include "../../core/Process/processMgr.h"
#include "../../core/Settings.h"
#include "../../core/Messaging/eventMgr.h"



struct healthData {
private:
	unsigned int maxHP;
	//HP may go below zero.
	int currentHP;
public:
	void setHP(unsigned int HP){
		this->currentHP = HP;

	}

	void Damage(unsigned int damage){
		this->currentHP -= damage;
	}

	void Heal(unsigned int heal){
		this->currentHP += heal;
	}

	int getHP(){
		return this->currentHP;
	};


};

class healthProcessor : public objectProcessor {
private:

public:
	healthProcessor(processMgr &processManager, Settings &settings, eventMgr &_eventManager){};
	void Process(float dt){};
	void postProcess();
};