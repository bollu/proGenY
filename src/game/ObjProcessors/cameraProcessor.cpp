#pragma once
#include "cameraProcessor.h"


void cameraProcessor::onObjectAdd(Object *obj){
	cameraData *data = obj->getProp<cameraData>(Hash::getHash("cameraData"));

	if(!data){
		return;
	}

	if(!data->enabled){
		return;
	}	
	data->cameraCenter = view->getCenter();
	data->accumilator = 0;


};

void cameraProcessor::Process(float dt){
	for(auto it=  objMap->begin(); it != objMap->end(); ++it){
		Object *obj = it->second;

		cameraData *data = obj->getProp<cameraData>(Hash::getHash("cameraData"));

		if(!data){
			continue;
		}

		if(!data->enabled){
			continue;
		}	

		vector2 cameraMoveAmt = this->_calcCameraMoveAmt(obj, data);
		this->_simulateCamera(cameraMoveAmt, dt, data);
	}

}

vector2 cameraProcessor::_limitCameraCoord(vector2 cameraCoord, cameraData *data){
	vector2 limitedCameraCoord = cameraCoord;

	if(cameraCoord.x < data->minCoord.x){
		limitedCameraCoord.x = data->minCoord.x;
	}
	if(cameraCoord.y < data->minCoord.y){
		limitedCameraCoord.y = data->minCoord.y;
	}

	if(cameraCoord.x > data->maxCoord.x){
		limitedCameraCoord.x = data->maxCoord.x;
	}
	if(cameraCoord.y > data->maxCoord.y){
		limitedCameraCoord.y = data->maxCoord.y;
	}

	return limitedCameraCoord;

};	

vector2 cameraProcessor::_limitMoveAmt(vector2 moveAmt, vector2 maxMoveAmt){
	vector2 limitedMoveAmt = moveAmt;

	if(moveAmt.x > maxMoveAmt.x){
		limitedMoveAmt.x = maxMoveAmt.x;
	}

	if(moveAmt.x < -maxMoveAmt.x){
		limitedMoveAmt.x = -maxMoveAmt.x;
	}

	if(moveAmt.y > maxMoveAmt.y){
		limitedMoveAmt.y = maxMoveAmt.y;
	}
	
	if(moveAmt.y < -maxMoveAmt.y){
		limitedMoveAmt.y = -maxMoveAmt.y;
	}

	return limitedMoveAmt;
};

vector2 cameraProcessor::_calcCameraMoveAmt(Object *obj, cameraData *data){

	vector2 windowDim = vector2::cast(window->getSize());
	vector2 *gamePos = obj->getProp<vector2>(Hash::getHash("position"));
	vector2 screenPos = this->view->game2ScreenCoord(*gamePos);

	vector2 currentCameraCenter = view->getCenter();
	vector2 boxBottomLeft = currentCameraCenter - vector2(data->boxHalfW, data->boxHalfH);
	vector2 boxtopRight = currentCameraCenter + vector2(data->boxHalfW, data->boxHalfH);

	vector2 cameraMoveAmt = vector2(0, 0);
	if(screenPos.x < boxBottomLeft.x){
		cameraMoveAmt.x = screenPos.x - boxBottomLeft.x;
	}
	if(screenPos.x > boxtopRight.x){
		cameraMoveAmt.x = screenPos.x - boxtopRight.x;
	}

	if(screenPos.y < boxBottomLeft.y){
		cameraMoveAmt.y = screenPos.y - boxBottomLeft.y;
	}

	if(screenPos.y > boxtopRight.y){
		cameraMoveAmt.y = screenPos.y - boxtopRight.y;
	}




	return cameraMoveAmt;
};

void cameraProcessor::_simulateCamera(vector2 cameraMoveAmt, float dt, cameraData *data){
	vector2 newCameraCenter = view->getCenter();

	data->accumilator += dt;

	if(data->accumilator > (this->maxAccumilation)){
		data->accumilator = this->stepSize;
	}

	while(data->accumilator >= this->stepSize){

		//use a damped spring equation to move the camera
		//science, bitch!
		float k = 0.3;
		//q = 1 => critically damped spring
		float q = 1;

		vector2 acc = k * cameraMoveAmt - q * data->v;
		data->v += acc ;

		newCameraCenter += data->v;

		data->accumilator -= this->stepSize;
	};

	newCameraCenter = _limitCameraCoord(newCameraCenter, data);
	view->setCenter(newCameraCenter);
	//view->setRotation(util::Angle::Deg(0));
};
