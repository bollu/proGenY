#include "CameraProcessor.h"


void cameraProcessor::_onObjectAdd(Object *obj){

	IO::infoLog<<"\nObject added to cameraProcessor";
	
	CameraData *data = obj->getPrimitive<CameraData>(Hash::getHash("CameraData"));

	if(!data){
		return;
	}

	if(!data->enabled){
		return;
	}	
	data->cameraCenter = view->getCenter();
	data->accumilator = 0;

	vector2 windowDim = vector2::cast(window->getSize());
	data->minCoord += windowDim * 0.5;
};

void cameraProcessor::_Process(Object *obj, float dt){
	CameraData *data = obj->getPrimitive<CameraData>(Hash::getHash("CameraData"));

	/*
	if(!data){
		continue;
	}
	*/

	if(!data->enabled){
		return;
	}	

	vector2 cameraMoveAmt = this->_calcCameraMoveAmt(obj, data);
	this->_simulateCamera(cameraMoveAmt, dt, data);
}

vector2 cameraProcessor::_limitCameraCoord(vector2 cameraCoord, CameraData *data){
	vector2 limitedCameraCoord = cameraCoord;

	if(cameraCoord.x < data->minCoord.x){
		limitedCameraCoord.x = data->minCoord.x;
	}

	/*
	if(cameraCoord.y < data->minCoord.y){
		limitedCameraCoord.y = data->minCoord.y;
	}*/

	if(cameraCoord.x > data->maxCoord.x){
		limitedCameraCoord.x = data->maxCoord.x;
	}

	/*
	if(cameraCoord.y > data->maxCoord.y){
		limitedCameraCoord.y = data->maxCoord.y;
	}*/

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

vector2 cameraProcessor::_calcCameraMoveAmt(Object *obj, CameraData *data){

	vector2 windowDim = vector2::cast(window->getSize());
	vector2 *gamePos = obj->getPrimitive<vector2>(Hash::getHash("position"));
	
	vector2 viewPos = this->view->view2RenderCoord(
		this->view->game2ViewCoord(*gamePos));


	vector2 currentCameraCenter = view->getCenter();
	vector2 boxBottomLeft = currentCameraCenter - vector2(data->boxHalfW, data->boxHalfH);
	vector2 boxtopRight = currentCameraCenter + vector2(data->boxHalfW, data->boxHalfH);

	vector2 cameraMoveAmt = vector2(0, 0);
	if(viewPos.x < boxBottomLeft.x){
		cameraMoveAmt.x = viewPos.x - boxBottomLeft.x;
	}
	if(viewPos.x > boxtopRight.x){
		cameraMoveAmt.x = viewPos.x - boxtopRight.x;
	}

	if(viewPos.y < boxBottomLeft.y){
		cameraMoveAmt.y = viewPos.y - boxBottomLeft.y;
	}

	if(viewPos.y > boxtopRight.y){
		cameraMoveAmt.y = viewPos.y - boxtopRight.y;
	}

	return cameraMoveAmt;
};

void cameraProcessor::_simulateCamera(vector2 cameraMoveAmt, float dt, CameraData *data){
	vector2 newCameraCenter = view->getCenter();

	data->accumilator += dt;

	if(data->accumilator > (this->maxAccumilation)){
		data->accumilator = this->stepSize;
	}

	while(data->accumilator >= this->stepSize){

		//use a damped spring equation to move the camera
		//science, bitch!
		float k = 1;
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
