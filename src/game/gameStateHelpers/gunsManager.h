#pragma once
#include "../ObjProcessors/GunProcessor.h"
#include "../../core/controlFlow/EventManager.h"

class ObjectManager;

class gunsManager : public Observer{
public:
	gunsManager(EventManager &eventManager, ObjectManager &objectManager, viewProcess *viewProc,  Object *player);
	void recieveEvent(const Hash *eventName, baseProperty *eventData);

private:
	Object *player_;
	
	std::vector<Object *> guns_;
	Object *currentGun_;

	ObjectManager &objectManager_;
	viewProcess *viewProc_;

	//0 to guns.size() - 1
	int currentGunIndex_;

	//reloads the currentGun pointer and currentGunData pointer
	void reloadGunPtrs_();

	void gotoNextGun_(int skip);
	void gotoPrevGun_(int skip);

	
	void addGun_(Object *gun, bool currentGun=false);
	void switchGuns_(Object *prevGun, Object *newGun);

	void updateGunAngle_(util::Angle &angle);
	void fireGun_();

};
