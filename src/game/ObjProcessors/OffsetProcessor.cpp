#include "OffsetProcessor.h"

void OffsetProcessor::_Process(Object *obj, float dt){

	OffsetData *data = obj->getPrimitive<OffsetData>(Hash::getHash("OffsetData"));

	Object *parent = obj->getParent();
	assert(parent != NULL);

	if(data->offsetPos){
		vector2 *parentPos = parent->getPrimitive<vector2>(Hash::getHash("position"));
		vector2 *pos = obj->getPrimitive<vector2>(Hash::getHash("position"));

		(*pos) = *parentPos + data->posOffset;
	}

	if(data->offsetAngle){
		util::Angle *facing = obj->getPrimitive<util::Angle>(Hash::getHash("facing"));
		util::Angle *parentFacing = parent->getPrimitive<util::Angle>(
			Hash::getHash("facing"));

		*facing = *parentFacing + data->angleOffset;
	}
}

bool OffsetProcessor::_shouldProcess(Object *obj){
	return obj->hasProperty("OffsetData");
};