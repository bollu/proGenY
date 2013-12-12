#pragma once
#include "../../core/objectProcessor.h"
#include "../../core/Object.h"

#include "../../core/Process/processMgr.h"
#include "../../core/Settings.h"
#include "../../core/Messaging/eventMgr.h"



struct HealthData {
private:
	friend class healthProcessor;

	bool invul;

	float maxHP;
	//HP may go below zero.
	float currentHP;

public:

	HealthData(){
		this->maxHP = this->currentHP = -1;
		this->invul = false;
	}

	void setHP(float HP){
		this->currentHP = this->maxHP = HP;

	}

	void makeInvulnerable(){
		this->invul = true;
	}

	void makeVulnerable(){
		this->invul = false;
	}

	void Damage(float damage){
		if(!this->invul){
			this->currentHP -= damage;
			util::infoLog<<"currentHP = "<<currentHP;
		}
	}

	void Heal(float heal){
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