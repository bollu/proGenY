#pragma once
#include "../Hash.h"
#include "../../util/logObject.h"


class Process;

class processMgr{
private:
	std::map<const Hash*, Process *>processes;

	Process *_getProcess(const Hash* processName);
public:
	void addProcess(Process *p);

	void preUpdate();
	void Update(float dt);
	void Draw();
	void postDraw();

	void Shutdown();
	

	void PauseProcess(const Hash* processName);
	void ResumePorcess(const Hash* processName);

	template <typename processType>
	processType *getProcess(const Hash* processName){
		return dynamic_cast< processType *>(this->_getProcess(processName));
	};
};